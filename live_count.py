"""
live_count.py

Main app: live webcam feed -> YOLOv8 (.pt) or ONNX Runtime (.onnx) detects notes ->
IoU tracking -> automatic (AUTO) or manual (SINGLE/MULTI) counting -> running total 
receipt tracked and spoken aloud -> fake-note heuristic flags suspicious notes.

Controls:
    Mouse  - Click a button in the on-screen control bar to trigger any
             action below without touching the keyboard
    M      - Cycle count modes (SINGLE -> MULTI -> AUTO)
    C      - Reset/clear current session counts
    P      - Pause/resume camera tracking
    R      - Speak the running total
    G      - Toggle voice gender filter (Female -> Male -> All)
    V      - Toggle TTS engine (Sarvam AI <-> Offline pyttsx3)
    N      - Next Sarvam AI voice (38 verified working voices)
    B      - Previous Sarvam AI voice
    +/-    - Increase/decrease confidence threshold by 0.05
    SPACE  - In SINGLE mode: count the highest confidence note
             In MULTI mode: batch-count all currently tracked notes at once
             In AUTO mode: (disabled, counting is automatic)
    Q/ESC  - Quit and hear/print the session summary
"""

import argparse
import os
import time
import threading
import signal
import sys
from collections import defaultdict, Counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.tts_engine import TTSEngine
from utils.fake_check import check_note
from utils.tracker import NoteTracker
from utils.onnx_runner import ONNXRunner

# Updated denominations list: added "5" and removed "2000" as requested
DENOMINATIONS = ["5", "10", "20", "50", "100", "200", "500"]

# Per-denomination minimum confidence, applied ON TOP of the global --conf
# threshold. When a note is unclear the detector tends to emit a low-confidence
# Rs 5 guess, which inflated the running total, so Rs 5 must clear a higher bar
# before it is accepted. Denominations not listed use the global threshold only.
MIN_CONF_PER_DENOM = {
    "5": 0.60,
}

# Colour mapping in BGR for visual bounding boxes.  These values are sampled
# from the dominant colours of the current Indian note series, so the overlay
# immediately communicates the denomination being detected.
COLOR_MAP = {
    "5": (116, 156, 94),      # green
    "10": (130, 153, 191),    # chocolate brown
    "20": (29, 219, 202),     # yellow-green
    "50": (186, 197, 104),    # fluorescent blue/turquoise
    "100": (190, 126, 134),   # lavender
    "200": (81, 159, 230),    # bright orange
    "500": (137, 175, 169),   # stone grey
}

WINDOW_NAME = "INR Cash Counter"
CAM_WIDTH = 1280
CAM_HEIGHT = 720
BUTTON_BAR_HEIGHT = 64

# On-screen buttons mirroring every keyboard shortcut, in (label, key_code) form
# so a click can be funnelled through the exact same key-handling logic below.
CONTROL_BUTTONS = [
    ("MODE", ord("m")),
    ("CLEAR", ord("c")),
    ("PAUSE", ord("p")),
    ("CONF -", ord("-")),
    ("CONF +", ord("+")),
    ("GENDER", ord("g")),
    ("ENGINE", ord("v")),
    ("VOICE -", ord("b")),
    ("VOICE +", ord("n")),
    ("COUNT", ord(" ")),
    ("TOTAL", ord("r")),
    ("QUIT", 27),
]


def layout_buttons(specs, bar_y, bar_width, bar_height, margin=8, gap=4):
    """Compute fixed pixel rects for each button, evenly spaced across the bar"""
    n = len(specs)
    usable = bar_width - 2 * margin - gap * (n - 1)
    btn_w = usable // n
    btn_h = bar_height - 16
    buttons = []
    x = margin
    for label, code in specs:
        rect = (x, bar_y + 8, x + btn_w, bar_y + 8 + btn_h)
        buttons.append({"label": label, "code": code, "rect": rect})
        x += btn_w + gap
    return buttons


def draw_button_bar(canvas, buttons, mouse_pos, dynamic_labels=None, active_codes=None):
    """Render the control bar, highlighting the hovered button and any active toggle"""
    dynamic_labels = dynamic_labels or {}
    active_codes = active_codes or set()
    mx, my = mouse_pos
    for btn in buttons:
        x1, y1, x2, y2 = btn["rect"]
        hovered = x1 <= mx <= x2 and y1 <= my <= y2
        label = dynamic_labels.get(btn["code"], btn["label"])

        base_color = (55, 55, 55)
        if btn["code"] in active_codes:
            base_color = (0, 120, 80)
        if hovered:
            base_color = tuple(min(255, c + 45) for c in base_color)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), base_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (100, 100, 100), 1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = x1 + max(2, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)


def make_mouse_callback(buttons, pending_key, mouse_pos):
    """Build an OpenCV mouse callback that queues a key code when a button is clicked"""
    def on_mouse(event, x, y, flags, userdata):
        mouse_pos[0] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in buttons:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    pending_key[0] = btn["code"]
                    break
    return on_mouse


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C signal"""
    global shutdown_requested
    shutdown_requested = True
    print("\nShutdown requested...")

signal.signal(signal.SIGINT, signal_handler)


class SessionState:
    def __init__(self):
        self.counts = defaultdict(int)
        self.flagged_suspicious = []  # list of (denomination, confidence)

    @property
    def total(self):
        return sum(int(denom) * count for denom, count in self.counts.items())

    def add(self, denomination):
        self.counts[denomination] += 1

    def reset(self):
        self.counts.clear()
        self.flagged_suspicious.clear()

    def summary_text(self):
        if not self.counts:
            return "No notes were counted this session."
        parts = [f"{c} note{'s' if c != 1 else ''} of {d} rupees"
                 for d, c in sorted(self.counts.items(), key=lambda x: int(x[0]))]
        text = "Session summary. " + ", ".join(parts) + f". Grand total: {self.total} rupees."
        if self.flagged_suspicious:
            text += f" Warning: {len(self.flagged_suspicious)} note(s) were flagged as suspicious."
        return text


def draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=2, length=15):
    """Draw stylish HUD corner brackets instead of basic boxes"""
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness * 2)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness * 2)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness * 2)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness * 2)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness * 2)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness * 2)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness * 2)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness * 2)


def get_safe_crop(frame, bbox):
    """Clip coordinates to frame bounds and return the cropped note image"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    # Ensure coordinates are within valid image dimensions
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Return empty slice if invalid
    if x1 >= x2 or y1 >= y2:
        return None
    return frame[y1:y2, x1:x2]


def process_detection_async(frame, model, is_onnx, conf_threshold, detection_lock, detection_results, iou_threshold=0.45):
    """Run YOLO or ONNX detection in a background thread to keep camera display smooth"""
    try:
        if is_onnx:
            # ONNX inference (predict directly returns (denom_str, conf, xyxy) based on metadata)
            raw_dets = model.predict(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)
            detections = []
            for denom, conf, xyxy in raw_dets:
                if conf < MIN_CONF_PER_DENOM.get(denom, 0.0):
                    continue
                detections.append((denom, conf, xyxy))
        else:
            # PyTorch inference
            results = model.predict(frame, imgsz=640, verbose=False)
            result = results[0]
            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    conf = float(boxes.conf[i])
                    if conf >= conf_threshold:
                        cls_idx = int(boxes.cls[i])
                        denom = result.names[cls_idx]
                        if conf < MIN_CONF_PER_DENOM.get(denom, 0.0):
                            continue
                        xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                        detections.append((denom, conf, xyxy))

        with detection_lock:
            detection_results["items"] = detections
            detection_results["version"] += 1
    except Exception as e:
        print(f"[!] Detection thread error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt",
                         help="path to model weights (.pt or .onnx)")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.35,
                         help="initial detection confidence threshold")
    parser.add_argument("--threads", type=int, default=2,
                         help="torch/cpu thread count for inference acceleration")
    parser.add_argument("--fps", type=int, default=30,
                         help="target frame rate of camera capture")
    parser.add_argument("--detect_every", type=int, default=2,
                         help="run inference every N frames to conserve CPU/NPU resources")
    parser.add_argument("--mode", type=str, default="auto",
                         choices=["single", "multi", "auto"],
                         help="single=manual confirm best, multi=manual batch, auto=automatic track/count")
    parser.add_argument("--verify_time", type=float, default=0.5,
                         help="seconds to verify note detection before counting (prevents flickering)")
    args = parser.parse_args()

    # Configure PyTorch CPU limits
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(args.threads)

    # Initialize selected inference engine
    is_onnx = args.model.lower().endswith(".onnx")
    if is_onnx:
        print(f"[*] Loading model via ONNX Runtime: {args.model}")
        model = ONNXRunner(args.model, imgsz=640)
        # Determine the selected execution provider name for HUD
        active_providers = model.session.get_providers()
        if "QNNExecutionProvider" in active_providers:
            hardware_backend = "ONNX QNN (NPU)"
        elif "DmlExecutionProvider" in active_providers:
            hardware_backend = "ONNX DirectML (GPU)"
        else:
            hardware_backend = "ONNX CPU"
    else:
        print(f"[*] Loading model via PyTorch YOLOv8: {args.model}")
        model = YOLO(args.model)
        hardware_backend = "PyTorch CPU/CUDA"

    tts = TTSEngine(mode="sarvam", voice="priya")
    state = SessionState()
    tracker = NoteTracker(
        iou_threshold=0.25,
        max_lost_frames=10,
        min_active_frames=3,
        min_active_seconds=args.verify_time,
    )

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[-] Could not open camera {args.cam}.")
        return
    
    # Configure high-speed camera properties
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_time = 1.0 / args.fps

    # On-screen control bar: buttons are laid out once (fixed pixel rects) and
    # a click funnels its key_code through the same handling as a real keypress.
    control_buttons = layout_buttons(CONTROL_BUTTONS, CAM_HEIGHT, CAM_WIDTH, BUTTON_BAR_HEIGHT)
    pending_key = [None]
    mouse_pos = [(-1, -1)]
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_callback(control_buttons, pending_key, mouse_pos))

    # Background Thread Variables for Inference
    detection_results = {"items": [], "version": 0}
    detection_lock = threading.Lock()
    detection_thread = None
    processed_detection_version = 0
    
    mode = args.mode
    conf_threshold = args.conf
    is_paused = False
    
    tts.say(f"Cash counter ready. {mode} mode enabled.")
    print(f"[*] Launching INR Cash Counter | Engine: {hardware_backend} | Mode: {mode.upper()}")
    
    # Speech Queuing variables
    speech_queue = []
    last_speech_added_time = 0
    SPEECH_COOLDOWN = 0.2  # seconds of silence before announcing the batch
    
    frame_count = 0
    
    try:
        while not shutdown_requested:
            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break
            
            # Flip camera frame horizontally to act as a mirror
            frame = cv2.flip(frame, 1)
            
            # 1. Trigger background inference thread
            if not is_paused:
                frame_count += 1
                if frame_count % args.detect_every == 0:
                    if detection_thread is None or not detection_thread.is_alive():
                        detection_thread = threading.Thread(
                            target=process_detection_async,
                            args=(frame, model, is_onnx, conf_threshold, detection_lock, detection_results),
                            daemon=True
                        )
                        detection_thread.start()
            
            # 2. Get latest detections
            detections = []
            detection_version = processed_detection_version
            with detection_lock:
                detections = list(detection_results["items"])
                detection_version = detection_results["version"]
            
            current_time = time.time()
            
            # 3. Update tracking only for a newly completed inference result.
            # Reprocessing the last result on every camera frame made notes become
            # "stable" without three actual detections and broke multi mode.
            if not is_paused and detection_version != processed_detection_version:
                processed_detection_version = detection_version
                newly_stable = tracker.update(detections, current_time)

                if mode == "auto":
                    for tracked in newly_stable:
                        tracked.counted = True
                        state.add(tracked.denom)
                        speech_queue.append((tracked.denom, True))
                        last_speech_added_time = current_time
                        print(f"[Auto] Verified & Counted Rs {tracked.denom} "
                              f"(conf {tracked.conf:.2f}) | total={state.total}")

            # 4. Handle Batch Speech Announcements
            if speech_queue and (time.time() - last_speech_added_time) > SPEECH_COOLDOWN:
                # Compile denominations
                denoms = [item[0] for item in speech_queue]
                any_suspicious = any(not item[1] for item in speech_queue)
                
                note_counts = Counter(denoms)
                summary_parts = [f"{count}x{d}" if count > 1 else f"{d}" for d, count in sorted(note_counts.items(), key=lambda x: int(x[0]))]
                
                speech_text = f"Counted {', '.join(summary_parts)}. Total {state.total}."
                if any_suspicious:
                    speech_text += " Warning, check note."
                
                tts.say(speech_text)
                speech_queue.clear()

            # 5. Render HUD Display
            display = frame.copy()
            
            # Interactive Laser Scanline effect (renders inside active camera window only)
            scanline_y = int((time.time() * 150) % 720)
            overlay_scan = display.copy()
            # Draw laser line from x=340 (end of receipt sidebar) to right edge
            cv2.line(overlay_scan, (340, scanline_y), (1280, scanline_y), (0, 255, 0), 2)
            cv2.line(overlay_scan, (340, max(0, scanline_y - 2)), (1280, max(0, scanline_y - 2)), (0, 200, 0), 1)
            cv2.line(overlay_scan, (340, min(719, scanline_y + 2)), (1280, min(719, scanline_y + 2)), (0, 200, 0), 1)
            cv2.addWeighted(overlay_scan, 0.25, display, 0.75, 0, display)
            
            # Draw tracked bounding boxes on camera view
            for tid, tracked in tracker.tracked_notes.items():
                x1, y1, x2, y2 = tracked.bbox
                denom = tracked.denom
                conf = tracked.conf
                
                # Keep every detected note in its denomination's real-world
                # colour, including while it is still being verified.
                color = COLOR_MAP.get(denom, (0, 255, 0))
                status = "Counted" if tracked.counted else "Scanning"
                
                # Draw corner brackets (Target locking HUD design)
                draw_corner_brackets(display, x1, y1, x2, y2, color, thickness=2, length=15)
                # Outer thin box
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
                
                # Draw metadata tag above note box
                label = f"#{tid} Rs {denom} ({conf:.2f}) [{status}]"
                (w_label, h_label), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                
                # Dynamic Label Tag Positioning: place inside box if too close to top edge
                tag_y1 = y1 - 20 if y1 - 20 >= 0 else y1 + 5
                tag_y2 = y1 if y1 - 20 >= 0 else y1 + h_label + 15
                text_y = y1 - 6 if y1 - 20 >= 0 else y1 + h_label + 10
                
                # Tag Background
                cv2.rectangle(display, (x1, tag_y1), (x1 + w_label + 10, tag_y2), color, -1)
                # Tag Text
                cv2.putText(display, label, (x1 + 5, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Render Semi-Transparent Receipt HUD Sidebar (Left side, width 340px)
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (340, 720), (18, 18, 18), -1)
            cv2.addWeighted(overlay, 0.88, display, 0.12, 0, display)
            
            # Draw Sidebar Elements
            # Header Title
            cv2.putText(display, "INR CASH COUNTER", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(display, (20, 55), (320, 55), (60, 60, 60), 1)
            
            # System Status Section
            cv2.putText(display, "SYSTEM STATUS", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
            
            # Mode Indicator
            mode_color = (0, 255, 0) if mode == "auto" else ((0, 230, 255) if mode == "multi" else (0, 0, 255))
            cv2.putText(display, f"Mode: {mode.upper()}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_color, 2, cv2.LINE_AA)
            
            # Engine Accelerator info
            accel_color = (0, 255, 0) if "NPU" in hardware_backend else ((0, 230, 255) if "GPU" in hardware_backend else (150, 150, 150))
            cv2.putText(display, f"Engine: {hardware_backend}", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, accel_color, 1, cv2.LINE_AA)
            
            # Current conf threshold
            cv2.putText(display, f"Conf Threshold: {conf_threshold:.2f}", (20, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Tracking status
            t_status = "PAUSED" if is_paused else "ACTIVE"
            t_color = (0, 0, 255) if is_paused else (0, 255, 0)
            cv2.putText(display, f"Tracking state: {t_status}", (20, 172),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1, cv2.LINE_AA)
            
            # Voice / TTS engine info
            voice_label = tts.engine_display
            voice_color = (0, 230, 255) if tts.mode == "sarvam" else (150, 150, 150)
            cv2.putText(display, f"Voice: {voice_label}", (20, 192),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, voice_color, 1, cv2.LINE_AA)
            
            cv2.line(display, (20, 207), (320, 207), (60, 60, 60), 1)
            
            # Session Receipt Counts
            cv2.putText(display, "SESSION RECEIPT", (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
            
            y_offset = 260
            for denom in DENOMINATIONS:
                count = state.counts[denom]
                color = COLOR_MAP[denom]
                
                # Denomination Color Dot
                cv2.circle(display, (32, y_offset - 5), 5, color, -1)
                
                # Denomination text label
                cv2.putText(display, f"Rs {denom}:", (50, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                
                # Quantity Count
                q_color = (255, 255, 255) if count > 0 else (100, 100, 100)
                q_thick = 2 if count > 0 else 1
                cv2.putText(display, f"x {count}", (180, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, q_color, q_thick, cv2.LINE_AA)
                
                # Subtotal
                subtotal = count * int(denom)
                cv2.putText(display, f"Rs {subtotal:,}", (250, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
                
                y_offset += 25
                
            cv2.line(display, (20, y_offset + 5), (320, y_offset + 5), (60, 60, 60), 1)
            
            # Total Notes count
            total_notes = sum(state.counts.values())
            cv2.putText(display, f"Notes Counted: {total_notes}", (20, y_offset + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
            
            # Grand Total Receipt Footer
            cv2.putText(display, "GRAND TOTAL", (20, y_offset + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(display, f"Rs {state.total:,}", (20, y_offset + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
            
            # Controls legend at bottom
            cv2.line(display, (20, 600), (320, 600), (60, 60, 60), 1)
            cv2.putText(display, "CONTROLS", (20, 615),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(display, "M:Mode C:Clear P:Pause +/-:Conf", (20, 633),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display, "G:Gender V:Engine N/B:Voice", (20, 651),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 255), 1, cv2.LINE_AA)
            
            # Space shortcut explanation based on mode
            if mode == "single":
                space_info = "SPACE: Count single best note"
            elif mode == "multi":
                space_info = "SPACE: Commit current batch"
            else:
                space_info = "SPACE: (Disabled in Auto mode)"
            cv2.putText(display, space_info, (20, 669),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display, "R:Read Total  Q/ESC:Quit", (20, 687),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.putText(display, "Or click a button below", (20, 705),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 200, 255), 1, cv2.LINE_AA)

            # Render window with an added control bar for mouse-driven buttons
            canvas = np.zeros((CAM_HEIGHT + BUTTON_BAR_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            canvas[:CAM_HEIGHT] = display

            space_label = "COMMIT" if mode == "multi" else "COUNT"
            dynamic_labels = {
                ord(" "): space_label,
                ord("p"): "RESUME" if is_paused else "PAUSE",
                ord("m"): f"MODE:{mode.upper()}",
            }
            active_codes = {ord("p")} if is_paused else set()
            draw_button_bar(canvas, control_buttons, mouse_pos[0], dynamic_labels, active_codes)

            cv2.imshow(WINDOW_NAME, canvas)

            # Handle FPS delay
            elapsed = time.time() - frame_start
            delay = max(1, int((frame_time - elapsed) * 1000))

            raw_key = cv2.waitKey(delay)
            key = None
            if raw_key != -1:
                key = raw_key & 0xFF
            if pending_key[0] is not None:
                key = pending_key[0]
                pending_key[0] = None

            if key is not None:

                # ESC or 'q'
                if key == 27 or key == ord("q"):
                    break

                # SPACE bar
                elif key == ord(" "):
                    if mode == "single" and detections:
                        # Count the highest confidence note in current frame
                        best_det = max(detections, key=lambda x: x[1])
                        denom, conf, bbox = best_det
                        
                        crop = get_safe_crop(frame, bbox)
                        is_genuine = True
                        fake_conf = 1.0
                        if crop is not None:
                            is_genuine, fake_conf, details = check_note(crop, denom)
                        
                        state.add(denom)
                        if not is_genuine:
                            state.flagged_suspicious.append((denom, fake_conf))
                            tts.say(f"Warning suspicious {denom}. Total {state.total}.")
                        else:
                            tts.say(f"Counted {denom}. Total {state.total}.")
                        print(f"[Single] Manual counted Rs {denom} (conf={conf:.2f}) | total={state.total}")
                        
                    elif mode == "multi":
                        # Fix: Manual batch count commits all stable notes that haven't been counted yet
                        newly_committed = []
                        for tid, tracked in tracker.tracked_notes.items():
                            # Check if the note has reached stable threshold but hasn't been counted yet
                            stable_for_long_enough = (
                                time.time() - tracked.start_time >= tracker.min_active_seconds
                            )
                            if (tracked.frames_active >= tracker.min_active_frames
                                    and stable_for_long_enough
                                    and not tracked.counted):
                                tracked.counted = True
                                state.add(tracked.denom)
                                newly_committed.append(tracked)
                                
                                # Apply fake check
                                crop = get_safe_crop(frame, tracked.bbox)
                                if crop is not None:
                                    is_genuine, fake_conf, details = check_note(crop, tracked.denom)
                                    if not is_genuine:
                                        state.flagged_suspicious.append((tracked.denom, fake_conf))
                                    
                        if newly_committed:
                            notes_counted = [n.denom for n in newly_committed]
                            note_counts = Counter(notes_counted)
                            summary_str = " + ".join([f"{count}x{d}" for d, count in sorted(note_counts.items(), key=lambda x: int(x[0]))])
                            tts.say(f"Counted {len(newly_committed)} notes. Total {state.total}.")
                            print(f"[Multi] Committed batch of {len(newly_committed)} notes: {summary_str} | total={state.total}")
                        else:
                            tts.say("No notes to count.")
                            print("[Multi] No new notes to commit.")
                
                # Change mode (M)
                elif key == ord("m") or key == ord("M"):
                    if mode == "single":
                        mode = "multi"
                    elif mode == "multi":
                        mode = "auto"
                    else:
                        mode = "single"
                    tracker.clear()
                    speech_queue.clear()
                    tts.say(f"{mode} mode enabled.")
                    print(f"[*] Changed mode to: {mode.upper()}")
                    
                # Reset Session (C or Backspace)
                elif key == ord("c") or key == ord("C") or key == 8:
                    state.reset()
                    tracker.clear()
                    speech_queue.clear()
                    tts.say("Receipt cleared. Total zero rupees.")
                    print("[*] Session counts cleared.")
                    
                # Pause/Resume (P)
                elif key == ord("p") or key == ord("P"):
                    is_paused = not is_paused
                    p_state = "paused" if is_paused else "resumed"
                    tts.say(f"Tracking {p_state}.")
                    print(f"[*] Tracking {p_state.upper()}")
                    
                # Speak Total (R)
                elif key == ord("r") or key == ord("R"):
                    tts.say(f"Total is {state.total}.")
                
                # Toggle voice gender filter: Female -> Male -> All (G)
                elif key == ord("g") or key == ord("G"):
                    new_gender = tts.toggle_gender()
                    v_name = tts.voice_display
                    tts.say(f"{new_gender.title()} voices. {v_name}.")
                    print(f"[*] Voice gender filter: {new_gender.upper()} | Voice: {v_name}")

                # Toggle TTS engine: Sarvam AI <-> Offline (V)
                elif key == ord("v") or key == ord("V"):
                    new_mode = tts.toggle_engine()
                    label = tts.engine_display
                    tts.say(f"{label}.")
                    print(f"[*] TTS engine: {label}")
                
                # Next Sarvam voice (N)
                elif key == ord("n") or key == ord("N"):
                    new_voice = tts.next_voice()
                    tts.say(f"{new_voice}.")
                    print(f"[*] Sarvam voice: {new_voice}")
                
                # Previous Sarvam voice (B)
                elif key == ord("b") or key == ord("B"):
                    new_voice = tts.prev_voice()
                    tts.say(f"{new_voice}.")
                    print(f"[*] Sarvam voice: {new_voice}")
                    
                # Adjust confidence threshold (+ / -)
                elif key == ord("+") or key == ord("="):
                    conf_threshold = min(0.95, conf_threshold + 0.05)
                    print(f"[*] Confidence threshold increased to: {conf_threshold:.2f}")
                elif key == ord("-"):
                    conf_threshold = max(0.10, conf_threshold - 0.05)
                    print(f"[*] Confidence threshold decreased to: {conf_threshold:.2f}")
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user...")
        
    finally:
        # Cleanup
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=0.5)
        cap.release()
        cv2.destroyAllWindows()
        
    # Announce session summary
    summary = state.summary_text()
    print("\n" + summary)
    tts.say(summary)
    time.sleep(4)
    tts.stop()


if __name__ == "__main__":
    main()
