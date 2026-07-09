"""
crop_by_class.py

Crops EVERY labeled bounding box out of EVERY image in a dataset split, and
sorts the crops into folders by class index: crops_by_class/class_0/,
crops_by_class/class_1/, etc.

This gives you a folder full of clean, cropped note images per class —
much easier and more reliable to identify denominations from than
scanning full-frame sample images.

Usage:
    python crop_by_class.py "data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8"

    # limit crops per class if a dataset is huge (default: no limit)
    python crop_by_class.py "data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8" --max-per-class 40
"""

import argparse
import os

import cv2


def crop_dataset(dataset_dir, out_dir="crops_by_class", max_per_class=None):
    os.makedirs(out_dir, exist_ok=True)
    counts = {}

    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue

        for img_name in os.listdir(img_dir):
            stem, ext = os.path.splitext(img_name)
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl_path):
                continue

            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_path) as f:
                lines = [l.strip().split() for l in f if l.strip()]

            for i, parts in enumerate(lines):
                cls_idx = parts[0]
                if max_per_class is not None and counts.get(cls_idx, 0) >= max_per_class:
                    continue

                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(int((xc - bw / 2) * w), 0)
                y1 = max(int((yc - bh / 2) * h), 0)
                x2 = min(int((xc + bw / 2) * w), w)
                y2 = min(int((yc + bh / 2) * h), h)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                class_dir = os.path.join(out_dir, f"class_{cls_idx}")
                os.makedirs(class_dir, exist_ok=True)

                n = counts.get(cls_idx, 0)
                cv2.imwrite(os.path.join(class_dir, f"{n:04d}_{stem}.jpg"), crop)
                counts[cls_idx] = n + 1

    print("Crops saved. Counts per class:")
    for cls_idx in sorted(counts.keys(), key=lambda x: int(x)):
        print(f"  class_{cls_idx}: {counts[cls_idx]} crops -> {out_dir}/class_{cls_idx}/")
    print(f"\nOpen a few crops in each class_N folder under '{out_dir}/' to identify "
          f"the denomination, then report the mapping back.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--out", default="crops_by_class")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="cap crops saved per class (default: save all)")
    args = parser.parse_args()
    crop_dataset(args.dataset_dir, out_dir=args.out, max_per_class=args.max_per_class)
