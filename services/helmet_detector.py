import os
import cv2
import uuid
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from config import Config

class HelmetDetector:
    def __init__(self, vit_enabled=None, vit_model_path=None, head_classifier_path=None, pose_model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() and Config.DEVICE == 'cuda' else 'cpu')
        
        # 1. Initialize Complementary Head Classifier (MobileNetV3)
        self.head_classifier = None
        self.head_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        clf_path = head_classifier_path or getattr(Config, 'HEAD_CLASSIFIER_PATH', os.path.join(Config.MODELS_DIR, 'head_classifier.pth'))
        if os.path.exists(clf_path):
            try:
                m = models.mobilenet_v3_small(weights=None)
                in_features = m.classifier[3].in_features
                m.classifier[3] = nn.Linear(in_features, 2)
                ckpt = torch.load(clf_path, map_location=self.device)
                m.load_state_dict(ckpt['model_state_dict'])
                m.to(self.device)
                m.eval()
                self.head_classifier = m
                print(f"[HelmetDetector] Validated Head Classifier loaded successfully from: {clf_path} (Val Acc: {ckpt.get('val_acc', 0.93)*100:.1f}%)")
            except Exception as e:
                print(f"[HelmetDetector] Warning: Could not load head classifier from {clf_path}: {e}")
                self.head_classifier = None

        # 2. Initialize Pose Estimator for Anatomical Head Localization
        self.pose_model = None
        pose_path = pose_model_path or getattr(Config, 'POSE_MODEL_PATH', os.path.join(Config.BASE_DIR, 'yolo11n-pose.pt'))
        if os.path.exists(pose_path):
            try:
                from ultralytics import YOLO
                self.pose_model = YOLO(pose_path)
                print(f"[HelmetDetector] Pose Estimator loaded successfully from: {pose_path}")
            except Exception as e:
                print(f"[HelmetDetector] Warning: Could not load pose model from {pose_path}: {e}")
                self.pose_model = None

        # 3. Optional ViT classifier support
        self.vit_enabled = vit_enabled if vit_enabled is not None else Config.ENABLE_VIT_CLASSIFIER
        self.vit_classifier = None
        if self.vit_enabled:
            self._init_vit_classifier(vit_model_path)

    def _init_vit_classifier(self, model_path=None):
        """
        Initializes optional Vision Transformer (ViT) for head region helmet classification.
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
            print(f"[HelmetDetector] ViT classifier could not be loaded ({e}).")
            self.vit_classifier = None
            self.vit_enabled = False

    @staticmethod
    def calculate_iou(boxA, boxB):
        if not boxA or not boxB or len(boxA) < 4 or len(boxB) < 4:
            return 0.0
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

    @classmethod
    def get_rider_bike_association_score(cls, person_box, bike_box):
        """
        Evaluates whether a detected person is physically mounted on a motorcycle.
        Returns a score >= 0.0 (higher = stronger confidence that the person is mounted).
        Returns 0.0 if the person is a bystander, pedestrian, background person, or outside the motorcycle.
        """
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box
        
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        pcx = (px1 + px2) / 2.0
        
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        bcx = (bx1 + bx2) / 2.0
        
        # 1. Horizontal alignment: rider center must be within motorcycle horizontal span (+ 20% margin)
        h_margin = bw * 0.20
        if not (bx1 - h_margin <= pcx <= bx2 + h_margin):
            return 0.0
            
        inter_x = max(0.0, min(px2, bx2) - max(px1, bx1))
        if inter_x / pw < 0.20:
            return 0.0

        # 2. Vertical Mount & Seating Sanity:
        # Rider top (head) must extend above or near top of bike
        if py1 > by1 + 0.45 * bh:
            return 0.0

        # Rider body must overlap or enter the motorcycle vertically
        # Rider bottom must reach at or into the bike top
        if py2 < by1 - 0.05 * bh:
            return 0.0

        # Rider bottom cannot extend far below bike base (pedestrian standing far below bike)
        if py2 > by2 + 0.25 * bh:
            return 0.0

        # 3. Relative Scale Sanity (Perspective depth check)
        if ph < 0.35 * bh or ph > 3.0 * bh:
            return 0.0

        inter_y = max(0.0, min(py2, by2) - max(py1, by1))
        if inter_y <= 0:
            return 0.0
            
        overlap = (inter_x * inter_y) / (pw * ph)
        vert_penetration = inter_y / max(1.0, min(ph, bh))
        h_dist = abs(pcx - bcx) / bw
        
        score = (0.5 * overlap + 0.5 * min(1.0, vert_penetration)) * (1.0 - 0.25 * min(1.0, h_dist))
        return score

    @classmethod
    def is_rider_on_bike(cls, person_box, bike_box):
        return cls.get_rider_bike_association_score(person_box, bike_box) >= 0.08

    def extract_head_crop(self, frame, person_box):
        """
        Extracts the head region (top 35% of person bbox) for helmet analysis.
        Returns: (head_crop, head_box: [x1, y1, x2, y2])
        """
        if frame is None or person_box is None:
            return None, [0, 0, 0, 0]
            
        h, w = frame.shape[:2]
        px1, py1, px2, py2 = [int(v) for v in person_box]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w - 1, px2), min(h - 1, py2)
        ph = max(1, py2 - py1)

        crop_ratio = getattr(Config, 'VIT_HEAD_CROP_RATIO', 0.35)
        head_h = max(18, int(ph * crop_ratio))
        head_y2 = min(py2, py1 + head_h)
        head_box = [px1, py1, px2, head_y2]

        hx1, hy1, hx2, hy2 = head_box
        if hy2 <= hy1 or hx2 <= hx1:
            return None, head_box

        crop = frame[hy1:hy2, hx1:hx2]
        return crop, head_box

    @staticmethod
    def match_detection_to_head(head_box, det_box, person_box):
        """
        Determines whether a helmet or no-helmet bounding box belongs to a specific rider's head.
        Handles both tight head-level boxes and rider-level aligned detections.
        Strictly rejects off-center vehicle/fairing boxes (e.g. scooter handlebars).
        """
        dx1, dy1, dx2, dy2 = det_box
        px1, py1, px2, py2 = person_box
        hx1, hy1, hx2, hy2 = head_box
        
        pw = max(1.0, px2 - px1)
        ph = max(1.0, py2 - py1)
        pcx = (px1 + px2) / 2.0
        
        det_w = max(1.0, dx2 - dx1)
        det_h = max(1.0, dy2 - dy1)
        det_cx = (dx1 + dx2) / 2.0
        
        # 1. Tight head-box match (for genuine head-level helmet/no-helmet detections)
        if det_h <= 0.55 * ph and dy2 <= py1 + 0.55 * ph:
            inter = max(0.0, min(hx2, dx2) - max(hx1, dx1)) * max(0.0, min(hy2, dy2) - max(hy1, dy1))
            containment = inter / (det_w * det_h)
            if containment >= 0.25:
                return True, containment + (0.5 if abs(det_cx - pcx) < 0.20 * pw else 0.0)
                
        # 2. Rider-level aligned detection (for detectors that predict helmet across the rider body)
        # Must be tightly centered on the person (within 15% margin) and bounded horizontally
        h_diff = abs(det_cx - pcx) / pw
        x_in_bounds = (dx1 >= px1 - 0.20 * pw) and (dx2 <= px2 + 0.20 * pw)
        
        inter_p = max(0.0, min(px2, dx2) - max(px1, dx1)) * max(0.0, min(py2, dy2) - max(py1, dy1))
        iou_p = inter_p / (pw * ph + det_w * det_h - inter_p)
        
        if h_diff <= 0.15 and x_in_bounds and iou_p >= 0.55:
            return True, iou_p
            
        return False, 0.0


    def classify_head(self, head_crop):
        """
        Evaluates a head crop using the trained complementary head classifier (MobileNetV3).
        Returns:
            status: 'helmet', 'no_helmet', or 'uncertain'
            confidence: float in [0.0, 1.0]
        """
        if self.head_classifier is None or head_crop is None or head_crop.size == 0:
            return 'uncertain', 0.50

        try:
            h, w = head_crop.shape[:2]
            if h < 10 or w < 10:
                return 'uncertain', 0.50

            rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            t = self.head_transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.head_classifier(t)
                probs = torch.softmax(logits, dim=-1)[0]
                h_prob = float(probs[0].item())
                nh_prob = float(probs[1].item())

            if nh_prob >= 0.70:
                return 'no_helmet', nh_prob
            elif h_prob >= 0.70:
                return 'helmet', h_prob
            else:
                return 'uncertain', max(h_prob, nh_prob)
        except Exception as e:
            print(f"[HelmetDetector] Head classification error: {e}")
            return 'uncertain', 0.50

    @classmethod
    def associate_riders_to_motorcycles(cls, bikes, persons):
        """
        Associates persons with motorcycles using 2-pass physical geometry:
        Pass 1: Direct seated mount assignment (at most 1 primary driver per bike).
        Pass 2: Tandem recovery for genuine tandem riders relative to the mounted driver.
        """
        if not bikes or not persons:
            return {b_idx: [] for b_idx in range(len(bikes))}
            
        bike_riders = {b_idx: [] for b_idx in range(len(bikes))}
        
        # Pass 1: Assign best primary seated driver to each bike (at most 1 primary driver per bike)
        scored_pairs = []
        for b_idx, b in enumerate(bikes):
            for p_idx, p in enumerate(persons):
                s = cls.get_rider_bike_association_score(p['bbox'], b['bbox'])
                if s >= 0.08:
                    combined = s * (0.6 + 0.4 * p.get('confidence', 0.5))
                    scored_pairs.append((combined, b_idx, p_idx, s))
                    
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        assigned_bikes = set()
        assigned_persons = set()
        
        for combined, b_idx, p_idx, s in scored_pairs:
            if b_idx not in assigned_bikes and p_idx not in assigned_persons:
                bike_riders[b_idx].append(persons[p_idx])
                assigned_bikes.add(b_idx)
                assigned_persons.add(p_idx)
                
        unassigned = [p for idx, p in enumerate(persons) if idx not in assigned_persons]
        
        # Pass 2: Tandem rider recovery relative to the mounted primary driver
        for b_idx, mounted in bike_riders.items():
            if not mounted:
                continue
            primary = mounted[0]
            pb_primary = primary['bbox']
            dph = pb_primary[3] - pb_primary[1]
            dpw = pb_primary[2] - pb_primary[0]
            dcx = (pb_primary[0] + pb_primary[2]) / 2.0
            dy1 = pb_primary[1]
            dy2 = pb_primary[3]
            
            b_box = bikes[b_idx]['bbox']
            bh = b_box[3] - b_box[1]
            bw = b_box[2] - b_box[0]
            
            still_unassigned = []
            for p in unassigned:
                pb = p['bbox']
                ppw = pb[2] - pb[0]
                pph = pb[3] - pb[1]
                pcx = (pb[0] + pb[2]) / 2.0
                py1 = pb[1]
                py2 = pb[3]
                
                # 1. Horizontal check
                if not (b_box[0] - 0.15 * bw <= pcx <= b_box[2] + 0.15 * bw):
                    still_unassigned.append(p)
                    continue
                    
                # Tandem distance to driver
                head_x_dist = abs(pcx - dcx)
                is_horiz_tandem = (head_x_dist <= 1.8 * max(ppw, dpw)) or (pb[0] >= pb_primary[0] - 15 and pb[2] <= pb_primary[2] + 15)
                if not is_horiz_tandem:
                    still_unassigned.append(p)
                    continue
                    
                # 2. Scale ratio (perspective invariant)
                scale_ratio = min(pph, dph) / max(pph, dph)
                if scale_ratio < 0.45:
                    still_unassigned.append(p)
                    continue
                    
                # 3. Head alignment
                head_y_diff = abs(py1 - dy1)
                is_head_aligned = (head_y_diff <= max(55.0, 0.40 * dph))
                if not is_head_aligned:
                    still_unassigned.append(p)
                    continue
                    
                # 4. Bottom / seating plane alignment
                bottom_diff = abs(py2 - dy2)
                is_seated_aligned = (bottom_diff <= max(80.0, 0.55 * max(pph, dph)))
                is_seated_occluded = (py2 >= dy1 + 0.45 * dph and py2 >= b_box[1] + 0.25 * bh and scale_ratio >= 0.55)
                
                if is_seated_aligned or is_seated_occluded:
                    mounted.append(p)
                else:
                    still_unassigned.append(p)
                    
            unassigned = still_unassigned
            
        return bike_riders

    def _save_evidence(self, head_crop, rider_label, frame_idx):
        """Saves focused head evidence crop for human auditing and dashboard display."""
        if head_crop is None or head_crop.size == 0:
            return None
        try:
            os.makedirs(Config.OUTPUTS_EVIDENCE_DIR, exist_ok=True)
            uid = str(uuid.uuid4())[:8]
            fname = f"ev_no_helmet_{rider_label.replace(' ', '_').replace('(', '').replace(')', '')}_f{frame_idx}_{uid}.jpg"
            save_path = os.path.join(Config.OUTPUTS_EVIDENCE_DIR, fname)
            cv2.imwrite(save_path, head_crop)
            return f"/outputs/evidence/{fname}"
        except Exception:
            return None

    def analyze(self, frame, detections, track_id=None, frame_idx=0):
        """
        Analyzes detections in a frame to identify all no-helmet violations independently.
        Generalizes across 1, 2, 3 or more riders on 1 or more motorcycles.
        Eliminates false positives by validating head bounding boxes, resolving class conflicts,
        and setting authoritative confirmed vs pending statuses.
        """
        violations = []
        rider_evaluations = []
        
        bikes = []
        persons = []
        helmets = []
        no_helmets = []
        
        for det in detections:
            cname = det['class_name'].lower().replace('_', ' ').strip()
            if cname in ['motorcycle', 'bike', 'two wheeler', 'two_wheeler']:
                bikes.append(det)
            elif cname in ['person', 'rider', 'driver', 'pillion']:
                persons.append(det)
            elif cname == 'helmet':
                helmets.append(det)
            elif cname in ['no-helmet', 'no helmet']:
                no_helmets.append(det)

        # 1. Associate each person with the best-matching motorcycle (if any)
        bike_riders = self.associate_riders_to_motorcycles(bikes, persons)

        if len(bikes) == 0:
            bike_riders[-1] = persons

        matched_no_helmets = set()
        matched_helmets = set()

        # 2. Process riders per motorcycle
        for b_idx, riders in bike_riders.items():
            if not riders:
                continue
                
            associated_bike = bikes[b_idx] if b_idx >= 0 and b_idx < len(bikes) else None
            
            # Sort riders horizontally to assign logical order: Rider 1 (Front/Driver), Rider 2, etc.
            riders_sorted = sorted(riders, key=lambda r: (r['bbox'][0] + r['bbox'][2]) / 2.0)
            total_riders_on_bike = len(riders_sorted)

            for r_idx, person in enumerate(riders_sorted):
                p_box = person['bbox']
                px1, py1, px2, py2 = p_box
                head_crop, head_box = self.extract_head_crop(frame, p_box)
                
                rider_num = r_idx + 1
                rider_label = f"Rider {rider_num}"
                if total_riders_on_bike > 1:
                    rider_label += f" ({'Driver' if r_idx == 0 else f'Pillion {r_idx}'})"
                
                # Check for direct 'no-helmet' box matching head
                best_nh_idx = None
                best_nh_score = 0.0
                for nh_idx, nh in enumerate(no_helmets):
                    if nh_idx in matched_no_helmets:
                        continue
                    is_match, score = self.match_detection_to_head(head_box, nh['bbox'], p_box)
                    if is_match and score > best_nh_score:
                        best_nh_score = score
                        best_nh_idx = nh_idx
                        
                # Check for direct 'helmet' box matching head
                best_h_idx = None
                best_h_score = 0.0
                for h_idx, h in enumerate(helmets):
                    if h_idx in matched_helmets:
                        continue
                    is_match, score = self.match_detection_to_head(head_box, h['bbox'], p_box)
                    if is_match and score > best_h_score:
                        best_h_score = score
                        best_h_idx = h_idx

                # Extract detector confidences
                h_conf = helmets[best_h_idx]['confidence'] if best_h_idx is not None else 0.0
                nh_conf = no_helmets[best_nh_idx]['confidence'] if best_nh_idx is not None else 0.0

                # Evaluate complementary head classifier on head crop
                clf_status, clf_conf = self.classify_head(head_crop)

                # Save head evidence crop for confirmed violation records
                ev_path = self._save_evidence(head_crop, rider_label, frame_idx)

                # Case 1: Direct No-Helmet detected (no competing helmet)
                if best_nh_idx is not None and best_h_idx is None:
                    matched_no_helmets.add(best_nh_idx)
                    conf = max(nh_conf, clf_conf if clf_status == 'no_helmet' else 0.85)
                    violations.append({
                        "violation_type": "no_helmet",
                        "title": f"NO HELMET VIOLATION — {rider_label}",
                        "detected": True,
                        "confidence": round(float(conf), 4),
                        "detector_confidence": round(float(nh_conf), 4),
                        "bounding_box": head_box,
                        "head_box": head_box,
                        "rider_box": p_box,
                        "bike_box": associated_bike['bbox'] if associated_bike else None,
                        "rider_id": rider_label,
                        "rider_count": total_riders_on_bike,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "review_status": "confirmed",
                        "status": "confirmed",
                        "evidence_image": ev_path,
                        "reason": f"Direct no-helmet detected on {rider_label} (confidence: {nh_conf:.2f})"
                    })
                    rider_evaluations.append({
                        'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                        'bike_box': associated_bike['bbox'] if associated_bike else None,
                        'status': 'NO_HELMET', 'confidence': round(float(conf), 4),
                        'reason': f"Direct no-helmet detected on {rider_label} (confidence: {nh_conf:.2f})"
                    })
                    continue

                # Case 2: Both Helmet and No-Helmet detected (Conflicting Predictions)
                if best_nh_idx is not None and best_h_idx is not None:
                    # Priority 1: High-confidence head classifier resolves the conflict
                    if clf_status == 'no_helmet' and clf_conf >= 0.70:
                        matched_no_helmets.add(best_nh_idx)
                        rider_status = 'NO_HELMET'
                        final_conf = clf_conf
                        reason = f"Head classifier confirmed no-helmet ({clf_conf:.2f}) resolving conflicting detections (no-helmet: {nh_conf:.2f}, helmet: {h_conf:.2f})"
                        violations.append({
                            "violation_type": "no_helmet",
                            "title": f"NO HELMET VIOLATION — {rider_label}",
                            "detected": True,
                            "confidence": round(float(clf_conf), 4),
                            "detector_confidence": round(float(nh_conf), 4),
                            "bounding_box": head_box,
                            "head_box": head_box,
                            "rider_box": p_box,
                            "bike_box": associated_bike['bbox'] if associated_bike else None,
                            "rider_id": rider_label,
                            "rider_count": total_riders_on_bike,
                            "track_id": track_id,
                            "frame_number": frame_idx,
                            "review_status": "confirmed",
                            "status": "confirmed",
                            "evidence_image": ev_path,
                            "reason": reason
                        })
                        rider_evaluations.append({
                            'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                            'bike_box': associated_bike['bbox'] if associated_bike else None,
                            'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                        })
                        continue
                    elif clf_status == 'helmet' and clf_conf >= 0.70:
                        matched_helmets.add(best_h_idx)
                        matched_no_helmets.add(best_nh_idx)
                        rider_status = 'HELMET_WORN'
                        final_conf = clf_conf
                        reason = f"Head classifier confirmed helmet ({clf_conf:.2f}) resolving conflicting detections"
                        rider_evaluations.append({
                            'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                            'bike_box': associated_bike['bbox'] if associated_bike else None,
                            'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                        })
                        continue
                    elif h_conf >= nh_conf + 0.08:
                        matched_helmets.add(best_h_idx)
                        matched_no_helmets.add(best_nh_idx)
                        rider_status = 'HELMET_WORN'
                        final_conf = h_conf
                        reason = f"Dominant helmet detection ({h_conf:.2f} vs no-helmet {nh_conf:.2f})"
                        rider_evaluations.append({
                            'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                            'bike_box': associated_bike['bbox'] if associated_bike else None,
                            'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                        })
                        continue
                    else:
                        matched_no_helmets.add(best_nh_idx)
                        rider_status = 'NO_HELMET'
                        final_conf = nh_conf
                        reason = f"No-helmet detection ({nh_conf:.2f}) in conflict (helmet: {h_conf:.2f})"
                        violations.append({
                            "violation_type": "no_helmet",
                            "title": f"NO HELMET VIOLATION — {rider_label}",
                            "detected": True,
                            "confidence": round(float(nh_conf), 4),
                            "detector_confidence": round(float(nh_conf), 4),
                            "bounding_box": head_box,
                            "head_box": head_box,
                            "rider_box": p_box,
                            "bike_box": associated_bike['bbox'] if associated_bike else None,
                            "rider_id": rider_label,
                            "rider_count": total_riders_on_bike,
                            "track_id": track_id,
                            "frame_number": frame_idx,
                            "review_status": "pending",
                            "status": "pending_review",
                            "evidence_image": ev_path,
                            "reason": reason
                        })
                        rider_evaluations.append({
                            'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                            'bike_box': associated_bike['bbox'] if associated_bike else None,
                            'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                        })
                        continue

                # Case 3: Helmet directly detected (no competing no-helmet box)
                if best_h_idx is not None and best_nh_idx is None:
                    if h_conf < 0.55 and clf_status == 'no_helmet' and clf_conf >= 0.80:
                        rider_status = 'NO_HELMET'
                        final_conf = clf_conf
                        reason = f"Head classifier confirmed no-helmet ({clf_conf:.2f}), rejecting weak detection ({h_conf:.2f})"
                        violations.append({
                            "violation_type": "no_helmet",
                            "title": f"NO HELMET VIOLATION — {rider_label}",
                            "detected": True,
                            "confidence": round(float(clf_conf), 4),
                            "detector_confidence": round(float(h_conf), 4),
                            "bounding_box": head_box,
                            "head_box": head_box,
                            "rider_box": p_box,
                            "bike_box": associated_bike['bbox'] if associated_bike else None,
                            "rider_id": rider_label,
                            "rider_count": total_riders_on_bike,
                            "track_id": track_id,
                            "frame_number": frame_idx,
                            "review_status": "confirmed",
                            "status": "confirmed",
                            "evidence_image": ev_path,
                            "reason": reason
                        })
                    else:
                        matched_helmets.add(best_h_idx)
                        rider_status = 'HELMET_WORN'
                        final_conf = h_conf
                        reason = f"Helmet detected on {rider_label} ({h_conf:.2f})"
                    rider_evaluations.append({
                        'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                        'bike_box': associated_bike['bbox'] if associated_bike else None,
                        'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                    })
                    continue

                # Case 4: Helmet held in hand / lower body (unworn helmet)
                held_helmet = None
                for h in helmets:
                    hx1, hy1, hx2, hy2 = h['bbox']
                    hcx = (hx1 + hx2) / 2.0
                    if px1 - 15 <= hcx <= px2 + 15:
                        if hy1 > py1 + 0.45 * (py2 - py1) and hy2 <= py2 + 30:
                            held_helmet = h
                            break

                if held_helmet is not None:
                    h_c = float(held_helmet['confidence'])
                    rider_status = 'NO_HELMET'
                    reason = f"Helmet detected in rider hand/lower body ({h_c:.2f}) rather than worn on head"
                    violations.append({
                        "violation_type": "no_helmet",
                        "title": f"NO HELMET VIOLATION (HELMET IN HAND) — {rider_label}",
                        "detected": True,
                        "confidence": round(h_c, 4),
                        "detector_confidence": round(h_c, 4),
                        "bounding_box": head_box,
                        "head_box": head_box,
                        "rider_box": p_box,
                        "bike_box": associated_bike['bbox'] if associated_bike else None,
                        "rider_id": rider_label,
                        "rider_count": total_riders_on_bike,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "review_status": "confirmed",
                        "status": "confirmed",
                        "evidence_image": ev_path,
                        "reason": reason
                    })
                    rider_evaluations.append({
                        'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                        'bike_box': associated_bike['bbox'] if associated_bike else None,
                        'status': rider_status, 'confidence': round(h_c, 4), 'reason': reason
                    })
                    continue

                # Case 5: Neither direct helmet nor direct no-helmet box matched head
                # Evaluate complementary head classifier:
                if clf_status == 'no_helmet' and clf_conf >= 0.70:
                    rider_status = 'NO_HELMET'
                    final_conf = clf_conf
                    reason = f"Validated head classifier confirmed no-helmet on {rider_label} (confidence: {clf_conf:.2f})"
                    violations.append({
                        "violation_type": "no_helmet",
                        "title": f"NO HELMET VIOLATION — {rider_label}",
                        "detected": True,
                        "confidence": round(float(clf_conf), 4),
                        "detector_confidence": round(float(person.get('confidence', 0.85)), 4),
                        "bounding_box": head_box,
                        "head_box": head_box,
                        "rider_box": p_box,
                        "bike_box": associated_bike['bbox'] if associated_bike else None,
                        "rider_id": rider_label,
                        "rider_count": total_riders_on_bike,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "review_status": "confirmed",
                        "status": "confirmed",
                        "evidence_image": ev_path,
                        "reason": reason
                    })
                elif clf_status == 'helmet' and clf_conf >= 0.70:
                    rider_status = 'HELMET_WORN'
                    final_conf = clf_conf
                    reason = f"Validated head classifier confirmed helmet on {rider_label} (confidence: {clf_conf:.2f})"
                else:
                    rider_status = 'UNCERTAIN'
                    final_conf = clf_conf
                    reason = f"Head evidence uncertain on {rider_label} (confidence: {clf_conf:.2f})"

                rider_evaluations.append({
                    'rider_id': rider_label, 'rider_box': p_box, 'head_box': head_box,
                    'bike_box': associated_bike['bbox'] if associated_bike else None,
                    'status': rider_status, 'confidence': round(float(final_conf), 4), 'reason': reason
                })
                continue

        # 3. Standalone direct no-helmet detections not linked to an associated rider box
        for nh_idx, nh in enumerate(no_helmets):
            if nh_idx not in matched_no_helmets:
                nh_box = nh['bbox']
                nh_w = nh_box[2] - nh_box[0]
                nh_h = nh_box[3] - nh_box[1]
                if nh_h > 150 or (nh_h / max(1.0, nh_w)) > 2.2:
                    continue
                if nh['confidence'] < 0.45:
                    continue

                matched_bike = None
                for bike in bikes:
                    if self.is_rider_on_bike(nh_box, bike['bbox']) or self.calculate_iou(nh_box, bike['bbox']) > 0.05:
                        matched_bike = bike
                        break

                if matched_bike is not None:
                    crop, _ = self.extract_head_crop(frame, nh_box)
                    ev_path = self._save_evidence(crop, f"Direct_{nh_idx+1}", frame_idx)
                    violations.append({
                        "violation_type": "no_helmet",
                        "title": "NO HELMET VIOLATION",
                        "detected": True,
                        "confidence": round(float(nh['confidence']), 4),
                        "detector_confidence": round(float(nh['confidence']), 4),
                        "bounding_box": nh_box,
                        "head_box": nh_box,
                        "rider_box": nh_box,
                        "bike_box": matched_bike['bbox'],
                        "rider_id": f"Direct Detection #{nh_idx+1}",
                        "rider_count": None,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "review_status": "confirmed",
                        "status": "confirmed",
                        "evidence_image": ev_path,
                        "reason": f"Direct no-helmet detected on motorcycle (confidence: {nh['confidence']:.2f})"
                    })

        self.last_rider_evaluations = rider_evaluations
        return violations
