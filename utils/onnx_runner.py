"""
utils/onnx_runner.py

Inference wrapper for YOLOv8 ONNX models using ONNX Runtime.
Optimized for Snapdragon X Plus/Elite NPUs (via QNN) and Adreno GPUs (via DirectML) on Windows.
"""

import cv2
import numpy as np
import onnxruntime as ort

class ONNXRunner:
    def __init__(self, model_path, imgsz=640):
        """
        Initialize the ONNX Runtime session.
        
        Args:
            model_path (str): Path to the .onnx model file.
            imgsz (int): Input image size matching model's expected input dimension.
        """
        self.model_path = model_path
        self.imgsz = imgsz
        
        # Query available execution providers
        available_providers = ort.get_available_providers()
        print(f"\n[ONNXRunner] Available Providers: {available_providers}")
        
        # Build priority queue of provider configurations
        priority_providers = []
        
        # 1. QNN Execution Provider (Qualcomm Hexagon NPU)
        if "QNNExecutionProvider" in available_providers:
            priority_providers.append((
                "QNNExecutionProvider", {
                    "backend_path": "QnnHtp.dll",  # Path to the HTP NPU backend
                    "htp_performance_mode": "burst",
                    "htp_precision_mode": "fp16"   # Use FP16 on HTP NPU
                }
            ))
            
        # 2. DirectML Execution Provider (Qualcomm Adreno GPU via DirectX12)
        if "DmlExecutionProvider" in available_providers:
            priority_providers.append("DmlExecutionProvider")
            
        # 3. CPU Fallback
        priority_providers.append("CPUExecutionProvider")
        
        # Build CPU-optimized session options (applied regardless of provider)
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_options.intra_op_num_threads = 2
        self.session_options.inter_op_num_threads = 1
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session_options.enable_cpu_mem_arena = True
        # Enable all available CPU-specific optimizations
        self.session_options.enable_cpu_mem_arena = True
        self.session_options.enable_mem_pattern = True
        self.session_options.enable_profiling = False

        self.session = None
        # Attempt to load session, falling back sequentially if initialization fails
        for i in range(len(priority_providers)):
            candidate_providers = priority_providers[i:]
            current_provider = candidate_providers[0]
            prov_name = current_provider[0] if isinstance(current_provider, tuple) else current_provider

            print(f"[ONNXRunner] Attempting to load model with {prov_name}...")
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    sess_options=self.session_options,
                    providers=candidate_providers
                )
                print(f"[ONNXRunner] Success! ONNX Session loaded with active providers: {self.session.get_providers()}")
                break
            except Exception as e:
                print(f"[ONNXRunner] Warning: Failed to initialize {prov_name}. Fallback triggered. (Error: {e})")

        if self.session is None:
            print("[ONNXRunner] Critical: All prioritized providers failed. Forcing CPU fallback.")
            self.session = ort.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=["CPUExecutionProvider"]
            )
            
        # Read input metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Parse names metadata stored in ONNX model by YOLO
        self.names = {}
        meta = self.session.get_modelmeta()
        custom_metadata_map = meta.custom_metadata_map
        if "names" in custom_metadata_map:
            import ast
            try:
                self.names = ast.literal_eval(custom_metadata_map["names"])
                self.names = {int(k): v for k, v in self.names.items()}
            except Exception:
                pass
        
        print(f"[ONNXRunner] Model Input: {self.input_name} (Shape: {self.input_shape})")
        print(f"[ONNXRunner] Model Output: {self.output_name}")
        print(f"[ONNXRunner] Model Names: {self.names}\n")

    def preprocess(self, frame):
        """
        Preprocess image to match YOLOv8 requirements:
        - Maintain aspect ratio with black letterbox padding
        - Scale to 0.0 - 1.0
        - Transpose to CHW (Channels, Height, Width)
        - Add batch dimension: [1, 3, imgsz, imgsz]
        
        Args:
            frame (np.ndarray): Original BGR camera frame.
            
        Returns:
            tuple: (processed_blob, (orig_h, orig_w), (resized_h, resized_w))
        """
        orig_h, orig_w = frame.shape[:2]
        
        # Calculate letterbox scale
        r = self.imgsz / max(orig_h, orig_w)
        resized_h, resized_w = int(orig_h * r), int(orig_w * r)
        
        resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
        # Create a black canvas of [imgsz, imgsz, 3] and copy the resized frame to top-left
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w] = resized
        
        # Convert BGR to RGB (YOLO expects RGB)
        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Transpose: HWC -> CHW
        blob = rgb_canvas.transpose((2, 0, 1))
        
        # Scale pixel values [0-255] to [0.0-1.0], expand dims to [1, 3, H, W]
        blob = blob.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        
        return blob, (orig_h, orig_w), (resized_h, resized_w)

    def postprocess(self, model_outputs, orig_shape, resized_shape, conf_threshold=0.3, iou_threshold=0.55):
        """
        Parse YOLOv8 ONNX model outputs, scale bounding boxes back to original aspect ratio,
        and run Non-Maximum Suppression (NMS) to eliminate overlapping duplicates.
        
        Args:
            model_outputs (list): Raw list of outputs from ORT run.
            orig_shape (tuple): (orig_h, orig_w) of original frame.
            resized_shape (tuple): (resized_h, resized_w) of letterboxed frame before padding.
            conf_threshold (float): Minimum confidence to filter boxes.
            iou_threshold (float): NMS intersection over union threshold.
            
        Returns:
            list: List of detections as (class_idx, confidence, [x1, y1, x2, y2])
        """
        output = model_outputs[0]  # Shape: [1, 4 + num_classes, num_boxes]
        if len(output.shape) == 3:
            output = output[0]  # Shape: [11, 8400]
            
        # Boxes coordinates [cx, cy, w, h] are first 4 channels
        boxes = output[:4, :].T  # [8400, 4]
        # Class confidences are the remaining channels
        class_scores = output[4:, :].T  # [8400, 7]
        
        # Find best class per box
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)
        
        # Filter out low confidence boxes
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        # Convert [cx, cy, w, h] to [x1, y1, x2, y2] relative to the padded canvas
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - (w / 2.0)
        y1 = cy - (h / 2.0)
        x2 = cx + (w / 2.0)
        y2 = cy + (h / 2.0)

        orig_h, orig_w = orig_shape
        resized_h, resized_w = resized_shape

        # Compute scaling factors (excluding the black letterbox padding area)
        scale_x = orig_w / resized_w
        scale_y = orig_h / resized_h

        cv_boxes = []
        cv_confs = []
        cv_class_ids = []

        for i in range(len(boxes)):
            # Clip both corners to the actual resized image area (excluding the
            # black padding). Clipping only the top-left corner and reusing the
            # original width/height let a box that started off-frame overshoot
            # past its real extent once re-scaled, so clip x2/y2 the same way
            # and derive width/height from the clipped corners.
            bx1 = max(0, min(x1[i], resized_w))
            by1 = max(0, min(y1[i], resized_h))
            bx2 = max(0, min(x2[i], resized_w))
            by2 = max(0, min(y2[i], resized_h))
            bw = max(0, bx2 - bx1)
            bh = max(0, by2 - by1)

            # Scale coordinates back to the original image scale
            ox = int(bx1 * scale_x)
            oy = int(by1 * scale_y)
            ow = int(bw * scale_x)
            oh = int(bh * scale_y)
            
            cv_boxes.append([ox, oy, ow, oh])  # OpenCV expects [x, y, w, h] for NMS
            cv_confs.append(float(confidences[i]))
            cv_class_ids.append(int(class_ids[i]))
            
        # Run OpenCV's fast C++ NMS
        indices = cv2.dnn.NMSBoxes(cv_boxes, cv_confs, conf_threshold, iou_threshold)
        
        results = []
        if len(indices) > 0:
            # cv2.dnn.NMSBoxes output can be flattened depending on opencv version
            for idx in indices.flatten():
                x, y, w, h = cv_boxes[idx]
                results.append((cv_class_ids[idx], cv_confs[idx], [x, y, x + w, y + h]))
                
        return results

    def predict(self, frame, conf_threshold=0.3, iou_threshold=0.55):
        """
        Run the complete inference pipeline on a single frame.
        
        Args:
            frame (np.ndarray): Original camera frame in BGR.
            conf_threshold (float): Confidence threshold.
            iou_threshold (float): NMS threshold.
            
        Returns:
            list: List of detections as (denomination_str, confidence, [x1, y1, x2, y2])
        """
        blob, orig_shape, resized_shape = self.preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        raw_results = self.postprocess(outputs, orig_shape, resized_shape, conf_threshold, iou_threshold)
        
        # Map class index to class names dynamically
        results = []
        for class_idx, conf, bbox in raw_results:
            denom = self.names.get(class_idx, str(class_idx))
            results.append((denom, conf, bbox))
            
        return results
