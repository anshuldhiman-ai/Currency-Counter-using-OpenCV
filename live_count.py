"""
live_count.py

Main app: live webcam feed -> YOLOv8 detects notes -> user confirms each
note with SPACE (prevents double counting) -> running total tracked and
spoken aloud -> fake-note heuristic flags suspicious notes -> session
summary on quit.

Controls:
    SPACE  - confirm/count the currently highest-confidence detected note
    R      - speak the running total
    Q      - quit and hear/print the session summary

Usage:
    python live_count.py
    python live_count.py --model models/best.pt --cam 0
"""

import argparse
import time
from collections import defaultdict

import cv2
from ultralytics import YOLO

from utils.tts_engine import TTSEngine
from utils.fake_check import check_note

DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000"]


class SessionState:
    def __init__(self):
        self.counts = defaultdict(int)
        self.flagged_suspicious = []  # list of (denomination, confidence)

    @property
    def total(self):
        return sum(int(denom) * count for denom, count in self.counts.items())

    def add(self, denomination):
        self.counts[denomination] += 1

    def summary_text(self):
        if not self.counts:
            return "No notes were counted this session."
        parts = [f"{c} note{'s' if c != 1 else ''} of {d} rupees"
                 for d, c in sorted(self.counts.items(), key=lambda x: int(x[0]))]
        text = "Session summary. " + ", ".join(parts) + f". Grand total: {self.total} rupees."
        if self.flagged_suspicious:
            text += f" Warning: {len(self.flagged_suspicious)} note(s) were flagged as suspicious."
        return text


def best_detection(result, conf_threshold=0.6):
    """Return (denomination_str, confidence, xyxy_box) for the highest
    confidence detection in a single YOLO result, or None."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes
    best_idx = int(boxes.conf.argmax())
    conf = float(boxes.conf[best_idx])
    if conf < conf_threshold:
        return None
    cls_idx = int(boxes.cls[best_idx])
    denomination = result.names[cls_idx]
    xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    return denomination, conf, xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.6,
                         help="min detection confidence to allow counting")
    args = parser.parse_args()

    model = YOLO(args.model)
    tts = TTSEngine()
    state = SessionState()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    tts.say("Cash counter ready. Show a note and press space to count it.")
    print("SPACE = count note | R = hear total | Q = quit")

    last_counted_time = 0
    COOLDOWN_SEC = 1.0  # prevents accidental double-press double counting

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False)
        detection = best_detection(results[0], conf_threshold=args.conf)

        display = frame.copy()
        if detection:
            denom, conf, (x1, y1, x2, y2) = detection
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(display, f"Rs {denom} ({conf:.2f})", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

        cv2.putText(display, f"Total: Rs {state.total}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, "SPACE=count  R=total  Q=quit", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("INR Cash Counter", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and detection and (time.time() - last_counted_time) > COOLDOWN_SEC:
            denom, conf, (x1, y1, x2, y2) = detection
            crop = frame[max(y1, 0):y2, max(x1, 0):x2]
            is_genuine, fake_conf, details = check_note(crop, denom)

            state.add(denom)
            last_counted_time = time.time()

            if is_genuine:
                tts.say(f"{denom} rupee note counted. Total {state.total} rupees.")
            else:
                state.flagged_suspicious.append((denom, fake_conf))
                tts.say(f"Warning. {denom} rupee note looks suspicious. "
                        f"Counted anyway. Total {state.total} rupees.")
            print(f"Counted Rs{denom} (conf {conf:.2f}) | genuine={is_genuine} "
                  f"fake_score={fake_conf:.2f} {details} | total={state.total}")

        elif key == ord("r"):
            tts.say(f"Running total: {state.total} rupees.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    summary = state.summary_text()
    print("\n" + summary)
    tts.say(summary)
    time.sleep(4)  # give TTS time to finish before process exits
    tts.stop()


if __name__ == "__main__":
    main()
