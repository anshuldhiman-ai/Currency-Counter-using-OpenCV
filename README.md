# INR Cash Counter

A real-time Indian currency counter using YOLOv8, a webcam, and offline
text-to-speech. It detects notes, draws denomination-coloured detection boxes,
and maintains a live receipt and running total.

> The fake-note check is only a basic visual heuristic. It is not suitable for
> authentication, banking, or any decision where counterfeit detection matters.

## Supported denominations

`₹5`, `₹10`, `₹20`, `₹50`, `₹100`, `₹200`, and `₹500`.

Each detected note uses a matching visual overlay colour: green (₹5), brown
(₹10), yellow-green (₹20), turquoise (₹50), lavender (₹100), orange (₹200),
and stone grey (₹500).

## Features

- Live camera detection with YOLOv8 PyTorch or ONNX models.
- Three count modes: automatic, manual single-note, and manual multi-note.
- IoU tracking and time-based verification to prevent duplicate counts.
- Live per-denomination receipt and total, with offline speech feedback.
- ONNX Runtime support with QNN NPU, DirectML GPU, and CPU fallback.
- Optional fake-note heuristic that flags suspicious notes for review.

## Setup

Requires Python 3.10+ and a webcam.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For ONNX inference, also install a suitable ONNX Runtime package for your
hardware, for example `onnxruntime` (CPU) or `onnxruntime-directml` (GPU).

## Run the counter

Place your trained YOLO weights at `models/best.pt`, then run:

```bash
python live_count.py --model models/best.pt --mode auto
```

Useful options:

```bash
# Choose a camera and require a note to remain visible for 2 seconds.
python live_count.py --cam 0 --mode auto --verify_time 2

# Manually commit all verified visible notes with Space.
python live_count.py --mode multi

# Use a converted ONNX model.
python live_count.py --model models/best.onnx --mode auto
```

## Controls

| Key | Action |
| --- | --- |
| `M` | Cycle modes: Single → Multi → Auto |
| `Space` | Count the best note in Single mode; commit verified notes in Multi mode |
| `C` | Clear the session receipt and total |
| `P` | Pause or resume tracking |
| `R` | Speak the current total |
| `+` / `-` | Adjust detection confidence |
| `Q` / `Esc` | Close the counter and announce the session summary |

In Auto mode, a note is counted once after it has been consistently detected
for the configured verification time (3 seconds by default). Move a counted
note out of view before presenting the next note.

## Train a model

1. Download YOLOv8-format Indian-currency datasets into `data/raw/`.
2. Merge and normalize their classes:

   ```bash
   python merge_datasets.py
   ```

3. Train the detector:

   ```bash
   python train.py
   ```

## Export ONNX

```bash
python export_onnx.py --model models/best.pt --imgsz 640
```

## Repository hygiene

Trained weights, datasets, exported models, videos, local virtual environments,
and inspection output are excluded through `.gitignore`. Keep these assets out
of Git; use a release asset or model/data storage service when they need to be
shared.

## Development

Run a quick syntax check after changing the counter or tracker:

```bash
python -m py_compile live_count.py utils/tracker.py
```

## License

This project is released under the [MIT License](LICENSE).
