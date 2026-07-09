"""
train.py

Fine-tunes a YOLOv8 model on the merged INR currency dataset.

Usage:
    python train.py
    python train.py --model yolov8s.pt --epochs 100 --imgsz 640
"""

import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/merged/data.yaml",
                         help="path to merged dataset yaml")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="base model to fine-tune (n=fastest, s/m=more accurate)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0,
                         help="data loader workers for reduced CPU usage")
    parser.add_argument("--threads", type=int, default=1,
                         help="torch intra-op thread count for reduced CPU usage")
    parser.add_argument("--project", default="models")
    parser.add_argument("--name", default="inr_counter")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers,
        # augmentation defaults in YOLOv8 already cover flip/hsv/mosaic;
        # bump these if your dataset is small
        degrees=10.0,       # random rotation, notes are rarely perfectly flat
        shear=2.0,
        perspective=0.0005, # slight perspective warp for camera-angle robustness
        mixup=0.1,
    )

    # Export a copy of best weights to a predictable path for live_count.py
    best_path = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nTraining complete. Best weights: {best_path}")
    print("Copy or symlink this to models/best.pt for live_count.py to find it, e.g.:")
    print(f"  cp {best_path} models/best.pt")


if __name__ == "__main__":
    main()
