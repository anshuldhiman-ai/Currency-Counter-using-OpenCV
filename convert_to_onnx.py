"""
Convert YOLOv8 PyTorch model to ONNX format for NPU/GPU acceleration
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best3.pt", help="Path to PyTorch model")
    parser.add_argument("--output", default="models/best3.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    args = parser.parse_args()

    print(f"[*] Loading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"[*] Exporting to ONNX: {args.output}")
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=12
    )
    
    print(f"[✓] Successfully converted to ONNX: {args.output}")

if __name__ == "__main__":
    main()
