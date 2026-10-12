import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from config import Config

class DetectionService:
    def __init__(self, model_path=None, conf_threshold=None, iou_threshold=None, device=None):
        self.conf_threshold = conf_threshold or Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or Config.IOU_THRESHOLD
        self.device = device or Config.DEVICE
        
        # Determine primary model path
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        elif os.path.exists(Config.CUSTOM_MODEL_PATH):
            self.model_path = Config.CUSTOM_MODEL_PATH
        elif os.path.exists(Config.DEFAULT_MODEL_PATH):
            self.model_path = Config.DEFAULT_MODEL_PATH
        else:
            self.model_path = 'yolo11n.pt'
            
        self.model = self._load_model(self.model_path)
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        self.is_custom_model = self._check_if_custom_model()

        # Check if primary model includes both riders and vehicles.
        # If not (e.g. models/best.pt only detects helmet, plate, triple-riding),
        # automatically load the complementary base model for person & motorcycle localization.
        self.base_model = None
        self.base_model_path = None
        names_lower = [str(n).lower() for n in self.class_names.values()]
        has_riders = any(r in names_lower for r in ['person', 'rider', 'driver', 'pillion'])
        has_bikes = any(b in names_lower for b in ['motorcycle', 'bike', 'two_wheeler', 'two wheeler'])

        if not (has_riders and has_bikes):
            base_cand = Config.DEFAULT_MODEL_PATH
            if os.path.exists(base_cand) and os.path.abspath(base_cand) != os.path.abspath(self.model_path):
                try:
                    print(f"[DetectionService] Primary model lacks vehicle/rider classes. Loading base detector: {base_cand}")
                    self.base_model = YOLO(base_cand)
                    self.base_model_path = base_cand
                except Exception as e:
                    print(f"[DetectionService] Warning: Could not load base vehicle model: {e}")

    def _load_model(self, path):
        try:
            print(f"[DetectionService] Loading primary YOLO model from: {path}")
            model = YOLO(path)
            if self.device == 'cuda' and not torch.cuda.is_available():
                print("[DetectionService] CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
            return model
        except Exception as e:
            fallback = Config.DEFAULT_MODEL_PATH
            if os.path.exists(fallback) and fallback != path:
                print(f"[DetectionService] Failed to load {path}: {e}. Falling back to {fallback}")
                self.model_path = fallback
                return YOLO(fallback)
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _check_if_custom_model(self):
        names = [str(n).lower() for n in self.class_names.values()]
        has_helmet = any('helmet' in n for n in names)
        has_plate = any('plate' in n for n in names)
        return has_helmet or has_plate

    def detect(self, image_or_frame, conf=None, iou=None):
        """
        Runs object detection on a single image.
        Uses composite detection (primary custom model + base vehicle model) when needed.
        Returns:
            list of structured detections:
            [
                {
                    'bbox': [x1, y1, x2, y2],
                    'class_id': int,
                    'class_name': str,
                    'confidence': float,
                    'model_source': str
                }, ...
            ]
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        detections = []

        # 1. Primary Model Inference
        # For fine-tuned helmet & plate classes, allow slightly lower threshold to prevent missing small objects
        custom_conf = min(conf, 0.20) if self.is_custom_model else conf
        results = self.model(
            image_or_frame,
            conf=custom_conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        if results and len(results) > 0 and results[0].boxes is not None:
            r = results[0]
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.class_names.get(cls_id, str(cls_id))

                detections.append({
                    'bbox': [round(float(v), 2) for v in xyxy],
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': round(confidence, 4),
                    'model_source': os.path.basename(self.model_path)
                })

        # 2. Base Model Inference (if vehicle/rider classes are needed)
        if self.base_model is not None:
            base_results = self.base_model(
                image_or_frame,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            if base_results and len(base_results) > 0 and base_results[0].boxes is not None:
                b_res = base_results[0]
                target_classes = {'person', 'motorcycle', 'bicycle', 'car', 'bus', 'truck'}
                for box in b_res.boxes:
                    cls_id = int(box.cls[0].item())
                    cname = self.base_model.names.get(cls_id, str(cls_id)).lower()
                    if cname in target_classes:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        confidence = float(box.conf[0].item())
                        detections.append({
                            'bbox': [round(float(v), 2) for v in xyxy],
                            'class_id': cls_id,
                            'class_name': cname,
                            'confidence': round(confidence, 4),
                            'model_source': os.path.basename(self.base_model_path)
                        })

        return detections

    def get_raw_diagnostic_data(self, image_or_frame, conf=None):
        """
        Gathers complete forensic inspection metadata for debugging and reporting.
        """
        conf = conf or self.conf_threshold
        detections = self.detect(image_or_frame, conf=conf)

        if isinstance(image_or_frame, str):
            img = cv2.imread(image_or_frame)
            img_shape = img.shape if img is not None else (0, 0, 0)
        elif hasattr(image_or_frame, 'shape'):
            img_shape = image_or_frame.shape
        else:
            img_shape = (0, 0, 0)

        class_counts = {}
        for d in detections:
            c = d['class_name']
            class_counts[c] = class_counts.get(c, 0) + 1

        return {
            'image_dimensions': {
                'height': img_shape[0],
                'width': img_shape[1],
                'channels': img_shape[2] if len(img_shape) > 2 else 1
            },
            'primary_model': {
                'path': self.model_path,
                'classes': self.class_names,
                'is_custom': self.is_custom_model
            },
            'base_model': {
                'path': self.base_model_path,
                'active': self.base_model is not None
            },
            'confidence_threshold': conf,
            'total_detections': len(detections),
            'class_counts': class_counts,
            'raw_detections': detections
        }

    def draw_detections(self, frame, detections, color_map=None):
        """
        Renders clear, professional bounding boxes and label badges on an image frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        default_colors = {
            'helmet': (0, 200, 0),        # Green
            'no-helmet': (0, 0, 220),     # Red
            'no helmet': (0, 0, 220),
            'motorcycle': (255, 140, 0),  # Blue/Cyan
            'bike': (255, 140, 0),
            'two_wheeler': (255, 140, 0),
            'person': (200, 200, 0),      # Yellow
            'rider': (200, 200, 0),
            'driver': (200, 200, 0),
            'pillion': (0, 215, 255),     # Gold
            'number-plate': (255, 0, 255),# Magenta
            'number plate': (255, 0, 255),
            'triple riding': (0, 0, 255), # Red
            'wrong-direction': (0, 0, 255),
            'right-direction': (0, 200, 0)
        }
        if color_map:
            default_colors.update(color_map)
            
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            cname = det['class_name'].lower()
            conf = det['confidence']
            color = default_colors.get(cname, (180, 180, 180))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label banner
            label = f"{det['class_name']} {conf:.2f}"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            bg_y1 = max(0, y1 - lh - 6)
            bg_y2 = y1
            cv2.rectangle(annotated, (x1, bg_y1), (x1 + lw + 6, bg_y2), color, -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        
        return annotated
