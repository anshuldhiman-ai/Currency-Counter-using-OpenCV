# INR Cash Counter — Live Denomination Counting + Fake Note Detection

Real-time Indian currency counting system using YOLOv8, with audio feedback
(designed to be usable by visually impaired users) and a fake-note sanity check.

## Features
- Live webcam detection of INR notes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)
- Manual-trigger counting (press SPACE or say "next") to avoid double-counting
- Running total announced via text-to-speech (offline, works without internet)
- Basic fake-note screening (color histogram + security-thread heuristic;
  upgradeable to a trained CNN once you have a real/fake dataset)
- End-of-session summary: breakdown by denomination + grand total

## Project structure
```
inr-cash-counter/
├── data/
│   ├── raw/              <- put your downloaded Roboflow datasets here
│   ├── merged/            <- output of merge_datasets.py
│   └── data.yaml           <- YOLO dataset config (auto-generated)
├── models/                 <- trained weights land here (best.pt)
├── utils/
│   ├── tts_engine.py        <- text-to-speech wrapper
│   └── fake_check.py       <- fake-note heuristic / classifier
├── merge_datasets.py       <- combine multiple Roboflow exports into one
├── train.py                 <- train YOLOv8 on the merged dataset
├── live_count.py             <- MAIN APP: live camera counting + TTS
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Get data
Download these from Roboflow Universe (export format: **YOLOv8**, zip):
- "Indian Currency Detection" by Rajshree (largest, ~2266 images)
- "Indian Currency Detector" by Jay Kumar (covers all 7 denominations)
- (optional) any real/fake dataset for the fake-note classifier upgrade

Unzip each into `data/raw/<dataset_name>/` — each folder should contain
`train/`, `valid/`, `test/` subfolders with `images/` and `labels/`, plus a
`data.yaml`.

## Step 2 — Merge datasets
```bash
python merge_datasets.py
```
This walks every folder in `data/raw/`, remaps class names to a single
canonical label set, and writes everything into `data/merged/` with one
combined `data.yaml`.

## Step 3 — Train
```bash
python train.py
```
Fine-tunes YOLOv8n (nano — fast, good for real-time on modest hardware) on
the merged dataset. Swap to `yolov8s.pt` in train.py for better accuracy if
you have a GPU.

## Step 4 — Run live counting
```bash
python live_count.py
```
- Point the camera at a note
- Press **SPACE** to confirm/count it (prevents double-counting)
- Press **R** to hear the running total
- Press **Q** to end session and hear the final summary

## Notes on the fake-note check
The included heuristic (`utils/fake_check.py`) is a placeholder — it checks
color distribution and looks for the security thread as a bright vertical
line. It is NOT forensic-grade. To make it robust, collect a real-vs-fake
dataset per denomination and train a small CNN classifier (see comments in
that file for how to swap it in).
