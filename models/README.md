# Models

Trained model weights are **not committed** to this repository — they are large
binary files that exceed GitHub's file-size limits (see `.gitignore`).

Place your trained weights here before running the app:

```
models/
├── best.pt      # PyTorch YOLOv8 weights (recommended)
└── best.onnx    # ONNX export for NPU / GPU / CPU acceleration
```

## How to get the weights

**Option A — Train your own (recommended):**

1. Add YOLOv8-format datasets under `data/raw/<dataset-name>/`.
2. Merge and normalise class labels:
   ```
   python merge_datasets.py
   ```
3. Train (Google Colab with a GPU is far faster — see `train_on_colab.ipynb`):
   ```
   python train.py
   ```
4. Export to ONNX:
   ```
   python export_onnx.py --model models/best.pt --imgsz 640
   ```

**Option B — Bring an existing YOLOv8 model** trained on the 7 INR classes
(`5, 10, 20, 50, 100, 200, 500`) and drop the `.pt` / `.onnx` file here.

## Run

```
python live_count.py --model models/best.onnx --conf 0.4 --verify_time 2 --detect_every 2
```
