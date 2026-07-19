"""
export_onnx.py

Converts a fine-tuned YOLOv8 PyTorch (.pt) model into the ONNX (.onnx) format
for hardware-accelerated inference on Snapdragon X NPU (QNN) or GPU (DirectML).

Usage:
    python export_onnx.py
    python export_onnx.py --model models/best.pt --imgsz 640
"""

import argparse
import sys
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch weights to ONNX format.")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to PyTorch model file (e.g. models/best.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference resolution. Must match the resolution used for live_count.")
    args = parser.parse_args()

    print(f"[*] Loading PyTorch YOLOv8 model from '{args.model}'...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        print("    Please ensure ultralytics is installed and the path to your weights is correct.")
        sys.exit(1)

    print(f"[*] Exporting model to ONNX format (imgsz={args.imgsz})...")
    print("    This may take a moment...")
    try:
        # We export with dynamic=False to allow fixed shape optimizations for hardware accelerators like QNN
        onnx_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=True  # Simplify ONNX graph for better performance
        )
        print(f"[+] Success! ONNX model saved to: {onnx_path}")
        print("    You can now run live counting with:")
        print(f"    python live_count.py --model {onnx_path} --mode auto")
    except Exception as e:
        print(f"[-] ONNX export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
