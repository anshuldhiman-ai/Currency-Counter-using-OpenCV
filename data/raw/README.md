Place your downloaded Roboflow YOLOv8-format dataset exports here — one subfolder
per dataset. Each folder must contain a `data.yaml` with class names and the
standard YOLO `train/`, `valid/`, `test/` image/label split directories.

Example after downloading two datasets:

```
data/raw/
├── INDIAN CURRENCY DETECTION.v4i.yolov8/
│   ├── train/images/...  train/labels/...
│   ├── valid/images/...  valid/labels/...
│   ├── test/images/...   test/labels/...
│   └── data.yaml
└── Indian Currency Detector.v1i.yolov8/
    ├── train/images/...  train/labels/...
    ├── valid/images/...  valid/labels/...
    ├── test/images/...   test/labels/...
    └── data.yaml
```

Run `merge_datasets.py` from the project root to combine all datasets, remap
their label indices to the canonical class list, and split into an 85/15
train/validation split ready for `train.py`.

### Canonical class order

Index 0: ₹5, 1: ₹10, 2: ₹20, 3: ₹50, 4: ₹100, 5: ₹200, 6: ₹500

### Adding a new dataset

1. Export from Roboflow in **YOLOv8 format**.
2. Unzip into `data/raw/<your-dataset-name>/`.
3. If the dataset uses non-standard or differently-ordered class labels, add a
   `DATASET_CLASS_INDEX_OVERRIDES` entry in `merge_datasets.py` before running it.
