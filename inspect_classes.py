"""
inspect_classes.py

Some Roboflow exports have class "names" like '0','1','2'... instead of
readable denomination labels, so there's no way to know which index means
which note without looking. This script draws bounding boxes + the raw
class index on a handful of sample images from one dataset so you can
visually match index -> denomination, then fill in CLASS_INDEX_MAP below
(or just tell Claude what you see and it'll update merge_datasets.py).

Usage:
    python inspect_classes.py "data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8"
    python inspect_classes.py "data/raw/INDIAN CURRENCY DETECTION.v4i.yolov8" --n 12
"""

import argparse
import os
import random

import cv2


def draw_and_save(dataset_dir, n=8, out_dir="inspect_out"):
    os.makedirs(out_dir, exist_ok=True)

    # find a split that has images
    split = None
    for candidate in ["train", "valid", "test"]:
        if os.path.isdir(os.path.join(dataset_dir, candidate, "images")):
            split = candidate
            break
    if split is None:
        print("No train/valid/test/images folder found in that path.")
        return

    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(img_files)

    saved = 0
    for img_name in img_files:
        if saved >= n:
            break
        stem, _ = os.path.splitext(img_name)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(lbl_path) as f:
            lines = [l.strip().split() for l in f if l.strip()]

        for parts in lines:
            cls_idx = parts[0]
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, f"class {cls_idx}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        out_path = os.path.join(out_dir, f"{saved:02d}_{img_name}")
        cv2.imwrite(out_path, img)
        saved += 1

    print(f"Saved {saved} annotated sample images to '{out_dir}/'. "
          f"Open them and note which class number appears on which denomination.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    draw_and_save(args.dataset_dir, n=args.n)
