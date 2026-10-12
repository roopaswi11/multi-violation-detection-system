import os
import cv2
import numpy as np
from config import Config

class HelmetDetector:
    def __init__(self, vit_enabled=None, vit_model_path=None):
        self.vit_enabled = vit_enabled if vit_enabled is not None else Config.ENABLE_VIT_CLASSIFIER
        self.vit_classifier = None
        
        if self.vit_enabled:
            self._init_vit_classifier(vit_model_path)

    def _init_vit_classifier(self, model_path=None):
        """
        Initializes optional Vision Transformer (ViT) for head region helmet classification.
        Gracefully handles missing or untrained ViT weights.
        """
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            model_id = model_path or Config.VIT_MODEL_NAME
            print(f"[HelmetDetector] Loading ViT feature extractor / model from: {model_id}")
            self.image_processor = AutoImageProcessor.from_pretrained(model_id)
            self.vit_classifier = AutoModelForImageClassification.from_pretrained(model_id)
            self.vit_classifier.eval()
            print("[HelmetDetector] ViT classifier initialized successfully.")
        except Exception as e:
            print(f"[HelmetDetector] ViT classifier could not be loaded ({e}). Falling back to YOLO detection logic.")
            self.vit_classifier = None
            self.vit_enabled = False

    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0.0, xB - xA)
        interH = max(0.0, yB - yA)
        interArea = interW * interH

        boxAArea = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        denom = boxAArea + boxBArea - interArea
        if denom <= 0:
            return 0.0
        return interArea / denom

    @staticmethod
    def is_rider_on_bike(person_box, bike_box):
        """
        Checks whether a detected person is mounted on a motorcycle using spatial geometry.
        A rider's horizontal center should be within or near the bike's horizontal boundaries,
        and the rider's lower body/center should vertically align with or overlap the bike.
        """
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box
        
        person_cx = (px1 + px2) / 2.0
        bike_w = bx2 - bx1
        
        # Horizontal check: rider center within bike bounds with 20% margin
        h_margin = bike_w * 0.20
        if not (bx1 - h_margin <= person_cx <= bx2 + h_margin):
            return False
            
        # Vertical check: rider's feet/bottom (py2) should be inside or slightly below bike top half
        # Rider's top (py1) should generally be above the bike's bottom
        if py2 < by1 or py1 > by2:
            return False
            
        # Check intersection over rider area
        xA = max(px1, bx1)
        yA = max(py1, by1)
        xB = min(px2, bx2)
        yB = min(py2, by2)
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        rider_area = max(1.0, (px2 - px1) * (py2 - py1))
        overlap_ratio = inter / rider_area
        
        return overlap_ratio > 0.15

    def extract_head_crop(self, frame, person_box):
        """
        Extracts the head region (top 25-30%) of a rider bounding box for helmet analysis.
        """
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w - 1, px2), min(h - 1, py2)
        
        box_h = py2 - py1
        head_h = max(10, int(box_h * Config.VIT_HEAD_CROP_RATIO))
        head_y2 = min(py2, py1 + head_h)
        
        if head_y2 <= py1 or px2 <= px1:
            return None, [px1, py1, px2, py1 + 10]
            
        crop = frame[py1:head_y2, px1:px2]
        head_box = [px1, py1, px2, head_y2]
        return crop, head_box

    def classify_head_vit(self, head_crop):
        """
        Runs ViT classification on head crop if available.
        Returns: (status: 'helmet'|'no-helmet'|'uncertain', confidence: float)
        """
        if not self.vit_classifier or head_crop is None:
            return None, 0.0
            
        try:
            import torch
            from PIL import Image
            img = Image.fromarray(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB))
            inputs = self.image_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.vit_classifier(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                conf, pred = torch.max(probs, dim=-1)
                
            # Documented ViT class mapping: if trained binary (0: helmet, 1: no-helmet)
            cls_name = self.vit_classifier.config.id2label.get(pred.item(), str(pred.item())).lower()
            if 'no' in cls_name or 'without' in cls_name:
                return 'no-helmet', float(conf.item())
            elif 'helmet' in cls_name:
                return 'helmet', float(conf.item())
            return 'uncertain', float(conf.item())
        except Exception as e:
            print(f"[HelmetDetector] ViT inference error: {e}")
            return None, 0.0

    def analyze(self, frame, detections, track_id=None, frame_idx=0):
        """
        Analyzes detections in a frame to identify no-helmet violations.
        
        Decision Rule:
        1. Separate detections into:
           - motorcycles / bikes
           - persons / riders
           - direct helmet detections
           - direct no-helmet detections
        2. Associate riders with motorcycles.
        3. For each mounted rider, check head region:
           a. If a direct 'no-helmet' box overlaps the rider's head -> VIOLATION (high confidence).
           b. If a direct 'helmet' box overlaps the rider's head -> NO VIOLATION.
           c. If ViT classifier is available and gives high confidence -> follow ViT.
           d. If neither helmet nor no-helmet is detected:
              Evaluate head size and resolution:
              - If head is too small (< 20px) or blurred -> status 'uncertain' (do NOT fabricate violation).
              - If motorcycle is clearly detected with visible rider but helmet is absent,
                mark as candidate for review with cautious calibrated confidence (0.50 - 0.65).
        """
        violations = []
        
        bikes = []
        persons = []
        helmets = []
        no_helmets = []
        
        for det in detections:
            cname = det['class_name'].lower().replace('_', '-').strip()
            if cname in ['motorcycle', 'bike', 'two-wheeler', 'two wheeler']:
                bikes.append(det)
            elif cname in ['person', 'rider', 'driver', 'pillion']:
                persons.append(det)
            elif cname == 'helmet':
                helmets.append(det)
            elif cname in ['no-helmet', 'no helmet']:
                no_helmets.append(det)

        # Associate each rider with a motorcycle
        handled_no_helmets = set()
        for person in persons:
            p_box = person['bbox']
            associated_bike = None
            for bike in bikes:
                if self.is_rider_on_bike(p_box, bike['bbox']):
                    associated_bike = bike
                    break
                    
            # Only consider persons mounted on motorcycles (not pedestrians)
            if not associated_bike and len(bikes) > 0:
                continue

            head_crop, head_box = self.extract_head_crop(frame, p_box)
            
            # 1. Check direct 'no-helmet' overlap with rider/head
            direct_no_helmet = None
            for idx, nh in enumerate(no_helmets):
                iou = self.calculate_iou(head_box, nh['bbox'])
                if iou > 0.15 or self.calculate_iou(p_box, nh['bbox']) > 0.10:
                    direct_no_helmet = nh
                    handled_no_helmets.add(idx)
                    break
                    
            if direct_no_helmet:
                conf = max(direct_no_helmet['confidence'], person['confidence'])
                violations.append({
                    "violation_type": "no_helmet",
                    "detected": True,
                    "confidence": round(conf, 4),
                    "bounding_box": p_box,
                    "head_box": head_box,
                    "bike_box": associated_bike['bbox'] if associated_bike else None,
                    "track_id": track_id,
                    "frame_number": frame_idx,
                    "reason": "Direct no-helmet detection on mounted rider"
                })
                continue
                
            # 2. Check direct 'helmet' overlap with head
            direct_helmet = None
            for h in helmets:
                iou = self.calculate_iou(head_box, h['bbox'])
                if iou > 0.15 or self.calculate_iou(p_box, h['bbox']) > 0.10:
                    direct_helmet = h
                    break
                    
            if direct_helmet:
                # Helmet confirmed: no violation
                continue
                
            # 3. ViT classifier check (if enabled)
            if self.vit_enabled and head_crop is not None:
                vit_status, vit_conf = self.classify_head_vit(head_crop)
                if vit_status == 'no-helmet' and vit_conf > 0.70:
                    violations.append({
                        "violation_type": "no_helmet",
                        "detected": True,
                        "confidence": round(vit_conf, 4),
                        "bounding_box": p_box,
                        "head_box": head_box,
                        "bike_box": associated_bike['bbox'] if associated_bike else None,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "reason": f"ViT classifier confirmed no-helmet (confidence: {vit_conf:.2f})"
                    })
                    continue
                elif vit_status == 'helmet':
                    continue

            # 4. Neither helmet nor no-helmet directly detected on rider
            if associated_bike:
                head_w = head_box[2] - head_box[0]
                head_h = head_box[3] - head_box[1]
                if head_w >= 25 and head_h >= 25:
                    violations.append({
                        "violation_type": "no_helmet",
                        "detected": True,
                        "confidence": round(person['confidence'] * 0.60, 4),
                        "bounding_box": p_box,
                        "head_box": head_box,
                        "bike_box": associated_bike['bbox'],
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "reason": "Rider without detected helmet on motorcycle (marked for review)"
                    })

        # Also capture standalone direct no-helmet detections not linked to person box
        for idx, nh in enumerate(no_helmets):
            if idx not in handled_no_helmets:
                violations.append({
                    "violation_type": "no_helmet",
                    "detected": True,
                    "confidence": round(nh['confidence'], 4),
                    "bounding_box": nh['bbox'],
                    "head_box": nh['bbox'],
                    "bike_box": None,
                    "track_id": track_id,
                    "frame_number": frame_idx,
                    "reason": "Direct no-helmet detected in frame"
                })

        return violations
