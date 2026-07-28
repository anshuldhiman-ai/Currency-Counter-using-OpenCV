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
from collections import defaultdict, Counter
from enum import Enum
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils.tts_engine import TTSEngine
from utils.fake_check import check_note
from utils.tracker import NoteTracker
from utils.onnx_runner import ONNXRunner

# ─── Constants ─────────────────────────────────────────────────────────────────
DENOMINATIONS = ["5", "10", "20", "50", "100", "200", "500"]

MIN_CONF_PER_DENOM: Dict[str, float] = {"5": 0.60}

COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
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
SIDEBAR_WIDTH = 340
BUTTON_BAR_HEIGHT = 64
CONTROLS_LEGEND_Y = 600
SPEECH_COOLDOWN = 0.2

# ─── Enums ─────────────────────────────────────────────────────────────────────

class CountMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
    AUTO = "auto"

# ─── UI Helpers ────────────────────────────────────────────────────────────────

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


def layout_buttons(
    specs: list, bar_y: int, bar_width: int, bar_height: int,
    margin: int = 8, gap: int = 4
) -> list:
    """Compute fixed pixel rects for each button, evenly spaced across the bar."""
    n = len(specs)
    usable = bar_width - 2 * margin - gap * (n - 1)
    btn_w = usable // n
    btn_h = bar_height - 16
    buttons = []
    x = margin
    for label, code in specs:
        buttons.append({
            "label": label, "code": code,
            "rect": (x, bar_y + 8, x + btn_w, bar_y + 8 + btn_h)
        })
        x += btn_w + gap
    return buttons


def draw_button_bar(
    canvas: np.ndarray, buttons: list, mouse_pos: Tuple[int, int],
    dynamic_labels: Optional[dict] = None,
    active_codes: Optional[set] = None
) -> None:
    """Render the control bar with hover highlights and active-state indicators."""
    dynamic_labels = dynamic_labels or {}
    active_codes = active_codes or set()
    mx, my = mouse_pos

    for btn in buttons:
        x1, y1, x2, y2 = btn["rect"]
        hovered = x1 <= mx <= x2 and y1 <= my <= y2
        label = dynamic_labels.get(btn["code"], btn["label"])

        # Base color: active toggles get green; hover gets lighter
        base_color = (55, 55, 55)
        if btn["code"] in active_codes:
            base_color = (0, 100, 60)
        if hovered:
            base_color = tuple(min(255, c + 50) for c in base_color)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), base_color, -1)
        # Rounded-corner effect: subtle border
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 80), 1)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx = x1 + max(2, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)


def make_mouse_callback(
    buttons: list, pending_key: list, mouse_pos: list
):
    """Build an OpenCV mouse callback that queues a key code on button click."""
    def on_mouse(event, x, y, *_):
        mouse_pos[0] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            for btn in buttons:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    pending_key[0] = btn["code"]
                    break
    return on_mouse


# ─── Graceful Shutdown ─────────────────────────────────────────────────────────
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    """Handle Ctrl+C signal via a thread-safe Event."""
    shutdown_event.set()
    print("\nShutdown requested...")

signal.signal(signal.SIGINT, signal_handler)


class SessionState:
    """Thread-safe session state for counts and flagged notes."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counts: Dict[str, int] = defaultdict(int)
        self.flagged_suspicious: list = []

    @property
    def total(self) -> int:
        with self._lock:
            return sum(int(d) * c for d, c in self.counts.items())

    def add(self, denomination: str) -> None:
        with self._lock:
            self.counts[denomination] += 1

    def reset(self) -> None:
        with self._lock:
            self.counts.clear()
            self.flagged_suspicious.clear()

    @property
    def total_notes(self) -> int:
        with self._lock:
            return sum(self.counts.values())

    def summary_text(self) -> str:
        with self._lock:
            if not self.counts:
                return "No notes were counted this session."
            parts = [
                f"{c} note{'s' if c != 1 else ''} of {d} rupees"
                for d, c in sorted(self.counts.items(), key=lambda x: int(x[0]))
            ]
        text = "Session summary. " + ", ".join(parts) + f". Grand total: {self.total} rupees."
        if self.flagged_suspicious:
            text += f" Warning: {len(self.flagged_suspicious)} note(s) were flagged as suspicious."
        return text


def draw_corner_brackets(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int], thickness: int = 2, length: int = 15
) -> None:
    """Draw HUD corner brackets instead of basic boxes."""
    for (ox, oy, dx, dy) in [
        (x1, y1, length, 0), (x1, y1, 0, length),
        (x2, y1, -length, 0), (x2, y1, 0, length),
        (x1, y2, length, 0), (x1, y2, 0, -length),
        (x2, y2, -length, 0), (x2, y2, 0, -length),
    ]:
        cv2.line(img, (ox, oy), (ox + dx, oy + dy), color, thickness * 2)


def get_safe_crop(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """Clip bbox to frame bounds and return the cropped note image."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x1 >= x2 or y1 >= y2:
        return None
    return frame[y1:y2, x1:x2]


def format_speech_summary(denom_list: list) -> str:
    """Build a compact spoken summary from a list of denomination strings."""
    counts = Counter(denom_list)
    parts = []
    for d in sorted(counts, key=lambda x: int(x[0])):
        c = counts[d]
        parts.append(f"{c}x{d}" if c > 1 else d)
    return ", ".join(parts)


def get_hardware_backend(model, is_onnx: bool) -> str:
    """Determine the display name for the active inference engine."""
    if is_onnx:
        providers = model.session.get_providers()
        if "QNNExecutionProvider" in providers:
            return "ONNX QNN (NPU)"
        elif "DmlExecutionProvider" in providers:
            return "ONNX DirectML (GPU)"
        return "ONNX CPU"
    return "PyTorch CPU/CUDA"


def run_yolo_inference(
    model, frame: np.ndarray, conf_threshold: float
) -> list:
    """Run PyTorch YOLOv8 inference and return filtered detections."""
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
    return detections


def process_detection_async(
    frame: np.ndarray, model, is_onnx: bool, conf_threshold: float,
    detection_lock: threading.Lock, detection_results: dict,
    iou_threshold: float = 0.45
) -> None:
    """Run YOLO or ONNX inference in a background thread."""
    try:
        if is_onnx:
            raw_dets = model.predict(frame, conf_threshold=conf_threshold,
                                     iou_threshold=iou_threshold)
            detections = []
            for denom, conf, xyxy in raw_dets:
                if conf < MIN_CONF_PER_DENOM.get(denom, 0.0):
                    continue
                detections.append((denom, conf, xyxy))
        else:
            detections = run_yolo_inference(model, frame, conf_threshold)

        with detection_lock:
            detection_results["items"] = detections
            detection_results["version"] += 1
    except Exception as e:
        print(f"[!] Detection thread error: {e}")


def render_scanline(display: np.ndarray, current_time: float) -> None:
    """Draw a subtle animated scanline effect on the camera view area."""
    scanline_y = int((current_time * 150) % CAM_HEIGHT)
    overlay = display.copy()
    cv2.line(overlay, (SIDEBAR_WIDTH, scanline_y), (CAM_WIDTH, scanline_y),
             (0, 255, 0), 2)
    for dy in (-2, 2):
        cv2.line(overlay, (SIDEBAR_WIDTH, max(0, scanline_y + dy)),
                 (CAM_WIDTH, min(CAM_HEIGHT - 1, scanline_y + dy)),
                 (0, 200, 0), 1)
    cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)


def render_detection_overlay(
    display: np.ndarray, tracker: NoteTracker
) -> None:
    """Draw bounding boxes, corner brackets, and metadata tags for tracked notes."""
    for tid, tracked in tracker.tracked_notes.items():
        x1, y1, x2, y2 = tracked.bbox
        denom = tracked.denom
        conf = tracked.conf
        color = COLOR_MAP.get(denom, (0, 255, 0))
        status = "Counted" if tracked.counted else "Scanning"

        draw_corner_brackets(display, x1, y1, x2, y2, color, thickness=2, length=15)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)

        label = f"#{tid} Rs {denom} ({conf:.2f}) [{status}]"
        (w_label, h_label), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )

        # Tag position: above box if room, inside box otherwise
        above = y1 - 20 >= 0
        tag_y1 = y1 - 20 if above else y1 + 5
        tag_y2 = y1 if above else y1 + h_label + 15
        text_y = y1 - 6 if above else y1 + h_label + 10

        cv2.rectangle(display, (x1, tag_y1), (x1 + w_label + 10, tag_y2), color, -1)
        cv2.putText(display, label, (x1 + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def render_sidebar(
    display: np.ndarray, state: SessionState, mode: str,
    hardware_backend: str, conf_threshold: float, is_paused: bool,
    tts: TTSEngine
) -> None:
    """Draw the semi-transparent receipt sidebar with session info."""
    sw = SIDEBAR_WIDTH

    # Semi-transparent dark overlay
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (sw, CAM_HEIGHT), (12, 12, 14), -1)
    cv2.addWeighted(overlay, 0.90, display, 0.10, 0, display)

    x, w = 20, sw - 40

    # ── Header ────────────────────────────────────────────────────────────
    cv2.putText(display, "₹ INR CASH COUNTER", (x, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(display, (x, 50), (x + w, 50), (60, 60, 60), 1)

    # ── Status section ─────────────────────────────────────────────────────
    row_y = 75
    cv2.putText(display, "SYSTEM STATUS", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
    row_y += 22

    # Mode
    mode_colors = {"auto": (0, 255, 0), "multi": (0, 230, 255), "single": (0, 0, 255)}
    mode_color = mode_colors.get(mode, (255, 255, 255))
    cv2.putText(display, f"Mode: {mode.upper()}", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, mode_color, 2, cv2.LINE_AA)
    row_y += 24

    # Engine
    accel_color = (0, 255, 0) if "NPU" in hardware_backend else (
        (0, 230, 255) if "GPU" in hardware_backend else (150, 150, 150))
    cv2.putText(display, f"Engine: {hardware_backend}", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, accel_color, 1, cv2.LINE_AA)
    row_y += 20

    # Conf threshold
    cv2.putText(display, f"Conf Threshold: {conf_threshold:.2f}", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200, 200, 200), 1, cv2.LINE_AA)
    row_y += 20

    # Tracking state
    t_status = "ACTIVE" if not is_paused else "PAUSED"
    t_color = (0, 255, 0) if not is_paused else (0, 0, 255)
    cv2.putText(display, f"Tracking: {t_status}", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, t_color, 1, cv2.LINE_AA)
    row_y += 22

    # Voice engine
    voice_color = (0, 230, 255) if tts.mode == "sarvam" else (150, 150, 150)
    cv2.putText(display, f"Voice: {tts.engine_display}", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, voice_color, 1, cv2.LINE_AA)
    row_y += 22

    cv2.line(display, (x, row_y), (x + w, row_y), (60, 60, 60), 1)

    # ── Receipt section ────────────────────────────────────────────────────
    row_y += 22
    cv2.putText(display, "SESSION RECEIPT", (x, row_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 140, 140), 1, cv2.LINE_AA)
    row_y += 22

    receipt_start = row_y
    for denom in DENOMINATIONS:
        count = state.counts[denom]
        color = COLOR_MAP[denom]

        cv2.circle(display, (x + 10, row_y - 4), 5, color, -1)
        cv2.putText(display, f"Rs {denom}:", (x + 24, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

        q_color = (255, 255, 255) if count > 0 else (100, 100, 100)
        cv2.putText(display, f"x {count}", (x + 130, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, q_color, 2 if count > 0 else 1, cv2.LINE_AA)

        subtotal = count * int(denom)
        cv2.putText(display, f"₹ {subtotal:,}", (x + 200, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150, 150, 150), 1, cv2.LINE_AA)

        row_y += 24

    cv2.line(display, (x, row_y), (x + w, row_y), (60, 60, 60), 1)

    # ── Totals ─────────────────────────────────────────────────────────────
    total_notes = state.total_notes
    cv2.putText(display, f"Notes Counted: {total_notes}", (x, row_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(display, "GRAND TOTAL", (x, row_y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(display, f"₹ {state.total:,}", (x, row_y + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # ── Controls legend ────────────────────────────────────────────────────
    legend_y = CONTROLS_LEGEND_Y
    cv2.line(display, (x, legend_y), (x + w, legend_y), (60, 60, 60), 1)
    cv2.putText(display, "CONTROLS", (x, legend_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    control_lines = [
        (f"M:{mode.upper()[:3]}  C:Clear  P:Pause  +/-:Conf", (180, 180, 180)),
        ("G:Gender  V:Engine  N/B:Voice", (0, 230, 255)),
    ]
    for i, (line, clr) in enumerate(control_lines):
        cv2.putText(display, line, (x, legend_y + 34 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, clr, 1, cv2.LINE_AA)

    space_hints = {
        "single": "SPACE: Count best note",
        "multi": "SPACE: Commit batch",
        "auto": "SPACE: (auto-counting)",
    }
    space_info = space_hints.get(mode, "")
    cv2.putText(display, space_info, (x, legend_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(display, "R:Read Total  Q/ESC:Quit", (x, legend_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)


def process_speech_queue(
    speech_queue: list, last_speech_added_time: float, tts: TTSEngine, state: SessionState
) -> float:
    """Flush queued speech items after cooldown period. Returns updated last_speech_added_time."""
    if not speech_queue or (time.time() - last_speech_added_time) <= SPEECH_COOLDOWN:
        return last_speech_added_time

    denoms = [item[0] for item in speech_queue]
    any_suspicious = any(not item[1] for item in speech_queue)

    speech_text = f"Counted {format_speech_summary(denoms)}. Total {state.total}."
    if any_suspicious:
        speech_text += " Warning, check note."

    tts.say(speech_text)
    speech_queue.clear()
    return time.time()


def handle_space_bar(
    mode: str, detections: list, frame: np.ndarray,
    tracker: NoteTracker, state: SessionState, tts: TTSEngine
) -> None:
    """Handle SPACE key press based on current mode."""
    if mode == "single" and detections:
        best_det = max(detections, key=lambda x: x[1])
        denom, conf, bbox = best_det
        crop = get_safe_crop(frame, bbox)
        is_genuine, fake_conf = True, 1.0
        if crop is not None:
            is_genuine, fake_conf, _ = check_note(crop, denom)
        state.add(denom)
        if not is_genuine:
            state.flagged_suspicious.append((denom, fake_conf))
            tts.say(f"Warning suspicious {denom}. Total {state.total}.")
        else:
            tts.say(f"Counted {denom}. Total {state.total}.")
        print(f"[Single] Manual counted Rs {denom} (conf={conf:.2f}) | total={state.total}")

    elif mode == "multi":
        newly_committed = []
        for tracked in tracker.tracked_notes.values():
            stable_long = time.time() - tracked.start_time >= tracker.min_active_seconds
            if tracked.frames_active >= tracker.min_active_frames and stable_long and not tracked.counted:
                tracked.counted = True
                state.add(tracked.denom)
                newly_committed.append(tracked)
                crop = get_safe_crop(frame, tracked.bbox)
                if crop is not None:
                    is_genuine, fake_conf, _ = check_note(crop, tracked.denom)
                    if not is_genuine:
                        state.flagged_suspicious.append((tracked.denom, fake_conf))

        if newly_committed:
            notes = [n.denom for n in newly_committed]
            summary_str = format_speech_summary(notes)
            tts.say(f"Counted {len(newly_committed)} notes. Total {state.total}.")
            print(f"[Multi] Committed batch: {summary_str} | total={state.total}")
        else:
            tts.say("No notes to count.")
            print("[Multi] No new notes to commit.")


def cycle_mode(mode: str) -> str:
    """Cycle count mode: single -> multi -> auto -> single."""
    cycle = {"single": "multi", "multi": "auto", "auto": "single"}
    return cycle.get(mode, "auto")


def configure_torch_threads(threads: int) -> None:
    """Set CPU thread limits for PyTorch and related libraries."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(threads)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best.pt",
                   help="path to model weights (.pt or .onnx)")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.35,
                   help="initial detection confidence threshold")
    p.add_argument("--threads", type=int, default=2,
                   help="torch/cpu thread count for inference acceleration")
    p.add_argument("--fps", type=int, default=30,
                   help="target frame rate of camera capture")
    p.add_argument("--detect_every", type=int, default=2,
                   help="run inference every N frames")
    p.add_argument("--mode", type=str, default="auto",
                   choices=["single", "multi", "auto"],
                   help="single=manual, multi=batch, auto=automatic")
    p.add_argument("--verify_time", type=float, default=0.5,
                   help="seconds to verify a note before counting")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch_threads(args.threads)

    # ── Initialise inference engine ──────────────────────────────────────────
    is_onnx = args.model.lower().endswith(".onnx")
    if is_onnx:
        print(f"[*] Loading model via ONNX Runtime: {args.model}")
        model = ONNXRunner(args.model, imgsz=640)
    else:
        print(f"[*] Loading model via PyTorch YOLOv8: {args.model}")
        model = YOLO(args.model)
    hardware_backend = get_hardware_backend(model, is_onnx)

    tts = TTSEngine(mode="sarvam", voice="priya")
    state = SessionState()
    tracker = NoteTracker(
        iou_threshold=0.25, max_lost_frames=10,
        min_active_frames=3, min_active_seconds=args.verify_time,
    )

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[-] Could not open camera {args.cam}.")
        return

    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    frame_time = 1.0 / args.fps

    # ── UI controls ──────────────────────────────────────────────────────────
    control_buttons = layout_buttons(CONTROL_BUTTONS, CAM_HEIGHT, CAM_WIDTH,
                                     BUTTON_BAR_HEIGHT)
    pending_key = [None]
    mouse_pos = [(-1, -1)]
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME,
                         make_mouse_callback(control_buttons, pending_key, mouse_pos))

    # ── State ────────────────────────────────────────────────────────────────
    detection_results: dict = {"items": [], "version": 0}
    detection_lock = threading.Lock()
    detection_thread: Optional[threading.Thread] = None
    processed_detection_version = 0

    mode = args.mode
    conf_threshold = args.conf
    is_paused = False
    frame_count = 0

    speech_queue: list = []
    last_speech_added_time: float = 0.0

    tts.say(f"Cash counter ready. {mode} mode enabled.")
    print(f"[*] Launching INR Cash Counter | Engine: {hardware_backend} | Mode: {mode.upper()}")

    # ── Main loop ────────────────────────────────────────────────────────────
    try:
        while not shutdown_event.is_set():
            frame_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            # Force-resize (many cameras ignore CAP_PROP_FRAME_*)
            if frame.shape[1] != CAM_WIDTH or frame.shape[0] != CAM_HEIGHT:
                frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)

            # 1. Background inference
            if not is_paused:
                frame_count += 1
                if (frame_count % args.detect_every == 0
                        and (detection_thread is None or not detection_thread.is_alive())):
                    detection_thread = threading.Thread(
                        target=process_detection_async,
                        args=(frame, model, is_onnx, conf_threshold,
                              detection_lock, detection_results),
                        daemon=True
                    )
                    detection_thread.start()

            # 2. Latest detections
            detections: list = []
            detection_version = processed_detection_version
            with detection_lock:
                detections = list(detection_results["items"])
                detection_version = detection_results["version"]

            current_time = time.time()

            # 3. Update tracking on fresh inference only
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

            # 4. Speech queue
            last_speech_added_time = process_speech_queue(
                speech_queue, last_speech_added_time, tts, state
            )

            # 5. HUD rendering
            display = frame.copy()
            render_scanline(display, current_time)
            render_detection_overlay(display, tracker)
            render_sidebar(display, state, mode, hardware_backend,
                           conf_threshold, is_paused, tts)

            # 6. Control bar
            canvas = np.zeros((CAM_HEIGHT + BUTTON_BAR_HEIGHT, CAM_WIDTH, 3),
                              dtype=np.uint8)
            canvas[:CAM_HEIGHT] = display
            space_label = "COMMIT" if mode == "multi" else "COUNT"
            dynamic_labels = {
                ord(" "): space_label,
                ord("p"): "RESUME" if is_paused else "PAUSE",
                ord("m"): f"MODE:{mode.upper()}",
            }
            active_codes = {ord("p")} if is_paused else set()
            draw_button_bar(canvas, control_buttons, mouse_pos[0],
                            dynamic_labels, active_codes)
            cv2.imshow(WINDOW_NAME, canvas)

            # 7. FPS control
            elapsed = time.time() - frame_start
            delay = max(1, int((frame_time - elapsed) * 1000))

            raw_key = cv2.waitKey(delay)
            key = None
            if raw_key != -1:
                key = raw_key & 0xFF
            if pending_key[0] is not None:
                key = pending_key[0]
                pending_key[0] = None

            if key is None:
                continue

            # ── Keyboard handling ────────────────────────────────────────────
            if key == 27 or key == ord("q"):
                break

            elif key == ord(" "):
                handle_space_bar(mode, detections, frame, tracker, state, tts)

            elif key in (ord("m"), ord("M")):
                mode = cycle_mode(mode)
                tracker.clear()
                speech_queue.clear()
                tts.say(f"{mode} mode enabled.")
                print(f"[*] Changed mode to: {mode.upper()}")

            elif key in (ord("c"), ord("C"), 8):
                state.reset()
                tracker.clear()
                speech_queue.clear()
                tts.say("Receipt cleared. Total zero rupees.")
                print("[*] Session counts cleared.")

            elif key in (ord("p"), ord("P")):
                is_paused = not is_paused
                p_state = "paused" if is_paused else "resumed"
                tts.say(f"Tracking {p_state}.")
                print(f"[*] Tracking {p_state.upper()}")

            elif key in (ord("r"), ord("R")):
                tts.say(f"Total is {state.total}.")

            elif key in (ord("g"), ord("G")):
                new_gender = tts.toggle_gender()
                v_name = tts.voice_display
                tts.say(f"{new_gender.title()} voices. {v_name}.")
                print(f"[*] Voice gender filter: {new_gender.upper()} | Voice: {v_name}")

            elif key in (ord("v"), ord("V")):
                label = tts.engine_display
                tts.say(f"{label}.")
                print(f"[*] TTS engine: {label}")

            elif key in (ord("n"), ord("N")):
                new_voice = tts.next_voice()
                tts.say(f"{new_voice}.")
                print(f"[*] Sarvam voice: {new_voice}")

            elif key in (ord("b"), ord("B")):
                new_voice = tts.prev_voice()
                tts.say(f"{new_voice}.")
                print(f"[*] Sarvam voice: {new_voice}")

            elif key in (ord("+"), ord("=")):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"[*] Confidence threshold increased to: {conf_threshold:.2f}")

            elif key == ord("-"):
                conf_threshold = max(0.10, conf_threshold - 0.05)
                print(f"[*] Confidence threshold decreased to: {conf_threshold:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user...")

    finally:
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=0.5)
        cap.release()
        cv2.destroyAllWindows()

    # Session summary
    summary = state.summary_text()
    print("\n" + summary)
    tts.say(summary)
    time.sleep(4)
    tts.stop()


if __name__ == "__main__":
    main()
