import cv2
import numpy as np
from config import Config

def box_iou(b1, b2):
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

class TripleRidingDetector:
    def __init__(self, max_permitted_riders=None, min_consecutive_frames=None):
        self.max_permitted_riders = max_permitted_riders or Config.MAX_PERMITTED_RIDERS
        self.min_consecutive_frames = min_consecutive_frames or Config.MIN_CONSECUTIVE_FRAMES_TRIPLE
        # Track history for video debounce: {track_id or bike_key: [count1, count2, ...]}
        self.bike_history = {}

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
        
        # 1. Horizontal alignment: rider center must be within motorcycle horizontal span (+ 15% margin)
        h_margin = bw * 0.15
        if not (bx1 - h_margin <= pcx <= bx2 + h_margin):
            return 0.0
            
        inter_x = max(0.0, min(px2, bx2) - max(px1, bx1))
        if inter_x / pw < 0.25:
            return 0.0

        # 2. Foot/Base Proximity:
        # A mounted rider's feet rest on footpegs/footboards near the lower frame.
        foot_offset_from_base = (by2 - py2) / bh
        if foot_offset_from_base > 0.55:
            return 0.0

        # Rider top must extend above or near top of bike
        if py1 > by1 + 0.45 * bh:
            return 0.0

        # Rider bottom cannot extend far below bike base
        if py2 > by2 + 0.20 * bh:
            return 0.0

        # 3. Relative Scale Sanity
        if ph < 0.35 * bh or ph > 2.5 * bh:
            return 0.0

        inter_y = max(0.0, min(py2, by2) - max(py1, by1))
        overlap = (inter_x * inter_y) / (pw * ph)
        
        h_dist = abs(pcx - bcx) / bw
        foot_dist = max(0.0, foot_offset_from_base)
        
        score = overlap * (1.0 - 0.35 * min(1.0, h_dist)) * (1.0 - 0.40 * min(1.0, foot_dist))
        return max(0.01, score)

    @classmethod
    def is_rider_on_bike(cls, person_box, bike_box):
        return cls.get_rider_bike_association_score(person_box, bike_box) > 0.0

    @classmethod
    def associate_riders_to_motorcycles(cls, bikes, persons):
        """
        Associates persons with motorcycles using 2-pass physical geometry:
        Pass 1: Direct seated mount assignment (exclusive).
        Pass 2: Tandem recovery for partially occluded tandem riders.
        """
        if not bikes or not persons:
            return {b_idx: [] for b_idx in range(len(bikes))}
            
        bike_riders = {b_idx: [] for b_idx in range(len(bikes))}
        unassigned = []
        
        # Pass 1: Direct mount scoring
        for p in persons:
            best_b = None
            best_s = 0.0
            for b_idx, b in enumerate(bikes):
                s = cls.get_rider_bike_association_score(p['bbox'], b['bbox'])
                if s > best_s:
                    best_s = s
                    best_b = b_idx
            if best_b is not None and best_s > 0.05:
                bike_riders[best_b].append(p)
            else:
                unassigned.append(p)
                
        # Pass 2: Tandem rider recovery for partially occluded riders on motorcycles
        for b_idx, mounted in bike_riders.items():
            if not mounted:
                continue
            b_box = bikes[b_idx]['bbox']
            bh = b_box[3] - b_box[1]
            bw = b_box[2] - b_box[0]
            
            still_unassigned = []
            for p in unassigned:
                pb = p['bbox']
                pw = pb[2] - pb[0]
                ph = pb[3] - pb[1]
                pcx = (pb[0] + pb[2]) / 2.0
                
                if not (b_box[0] - 0.10 * bw <= pcx <= b_box[2] + 0.10 * bw):
                    still_unassigned.append(p)
                    continue
                    
                is_tandem = False
                for m in mounted:
                    mb = m['bbox']
                    mph = mb[3] - mb[1]
                    mpw = mb[2] - mb[0]
                    
                    head_y_diff = abs(pb[1] - mb[1])
                    head_x_dist = abs(pcx - (mb[0] + mb[2]) / 2.0)
                    scale_ratio = min(ph, mph) / max(ph, mph)
                    bottom_diff = abs(pb[3] - mb[3]) / max(ph, mph)
                    
                    if (head_y_diff <= max(40.0, 0.25 * max(ph, mph)) and 
                        head_x_dist <= 1.8 * max(pw, mpw) and 
                        scale_ratio >= 0.55 and 
                        bottom_diff <= 0.40):
                        if pb[3] > b_box[1] + 0.20 * bh:
                            is_tandem = True
                            break
                            
                if is_tandem:
                    mounted.append(p)
                else:
                    still_unassigned.append(p)
            unassigned = still_unassigned
            
        return bike_riders

    @staticmethod
    def deduplicate_overlapping_riders(riders, iou_thresh=0.65):
        """
        Suppresses duplicate detector bounding boxes for the SAME person,
        while strictly preserving DISTINCT physical riders seated tandem on a motorcycle.
        """
        if not riders:
            return []
            
        thresh = iou_thresh if iou_thresh is not None else 0.65
        sorted_riders = sorted(riders, key=lambda r: r['confidence'], reverse=True)
        distinct = []
        
        for r in sorted_riders:
            r_box = r['bbox']
            rcx = (r_box[0] + r_box[2]) / 2.0
            rw = max(1.0, r_box[2] - r_box[0])
            rh = max(1.0, r_box[3] - r_box[1])
            
            is_dup = False
            for d in distinct:
                d_box = d['bbox']
                dcx = (d_box[0] + d_box[2]) / 2.0
                iou = box_iou(r_box, d_box)
                
                # Check center offset and head offset
                cx_offset = abs(rcx - dcx) / rw
                top_offset = abs(r_box[1] - d_box[1]) / rh
                
                # If high IoU and minimal center offset -> duplicate box of same rider
                if iou > thresh and (cx_offset < 0.15 and top_offset < 0.15 or iou > 0.85):
                    is_dup = True
                    break
                    
            if not is_dup:
                distinct.append(r)
                
        return distinct

    def analyze(self, frame, detections, track_id=None, frame_idx=0, is_video=False):
        """
        Analyzes a single image or video frame to identify all triple riding violations.
        Produces exactly ONE triple-riding event per qualifying motorcycle.
        """
        violations = []
        bikes = []
        persons = []
        direct_triple_boxes = []
        
        # Categorize input detections
        for det in detections:
            cname = det['class_name'].lower().replace('_', ' ').strip()
            if cname in ['motorcycle', 'bike', 'two wheeler', 'two_wheeler']:
                bikes.append(det)
            elif cname in ['person', 'rider', 'driver', 'pillion']:
                persons.append(det)
            elif 'triple' in cname or 'triple riding' in cname:
                direct_triple_boxes.append(det)

        # 1. Associate persons with motorcycles using validated 2-pass association
        bike_riders = self.associate_riders_to_motorcycles(bikes, persons)

        # 2. Process each motorcycle independently
        for b_idx, bike in enumerate(bikes):
            b_box = bike['bbox']
            associated_riders = bike_riders.get(b_idx, [])
            distinct_riders = self.deduplicate_overlapping_riders(associated_riders)
            rider_count = len(distinct_riders)
            
            # Check if a direct triple-riding detection overlaps this motorcycle
            matching_direct_triple = None
            for dt in direct_triple_boxes:
                if box_iou(b_box, dt['bbox']) > 0.20 or self.is_rider_on_bike(dt['bbox'], b_box):
                    matching_direct_triple = dt
                    break

            # If direct triple riding model fired on this bike, rider count is at least 3
            effective_rider_count = max(rider_count, 3 if matching_direct_triple else rider_count)
            
            # Key for tracking history
            bike_key = f"{track_id or b_idx}_{round(b_box[0]/50)}_{round(b_box[1]/50)}"
            if bike_key not in self.bike_history:
                self.bike_history[bike_key] = []
            self.bike_history[bike_key].append(effective_rider_count)
            if len(self.bike_history[bike_key]) > 15:
                self.bike_history[bike_key].pop(0)

            # Check violation condition: rider count > max_permitted_riders (typically 2)
            if effective_rider_count > self.max_permitted_riders:
                rider_confs = [r['confidence'] for r in distinct_riders]
                bike_conf = bike['confidence']
                avg_rider_conf = sum(rider_confs) / len(rider_confs) if rider_confs else 0.55
                
                if matching_direct_triple:
                    combined_conf = max(matching_direct_triple['confidence'],
                                        round(0.40 * matching_direct_triple['confidence'] + 0.35 * bike_conf + 0.25 * avg_rider_conf, 4))
                else:
                    combined_conf = round(0.55 * bike_conf + 0.45 * avg_rider_conf, 4)

                is_confirmed = True
                if is_video and len(self.bike_history[bike_key]) >= self.min_consecutive_frames:
                    recent_counts = self.bike_history[bike_key][-self.min_consecutive_frames:]
                    violating_frames = sum(1 for c in recent_counts if c > self.max_permitted_riders)
                    if violating_frames < 2:
                        is_confirmed = False

                reason_text = f"Detected {effective_rider_count} distinct riders on single motorcycle (permitted: {self.max_permitted_riders})"
                if matching_direct_triple:
                    reason_text += f" [Model verified: {matching_direct_triple['confidence']:.2f}]"

                violations.append({
                    "violation_type": "triple_riding",
                    "title": f"TRIPLE RIDING VIOLATION ({effective_rider_count} RIDERS)",
                    "detected": True,
                    "confidence": round(float(combined_conf), 4),
                    "bounding_box": b_box,
                    "bike_box": b_box,
                    "rider_boxes": [r['bbox'] for r in distinct_riders],
                    "rider_count": effective_rider_count,
                    "track_id": track_id,
                    "frame_number": frame_idx,
                    "review_status": "confirmed" if is_confirmed else "pending",
                    "reason": reason_text
                })

        # 3. If direct triple-riding detections exist that were not associated with any bike (e.g. bike wasn't detected by COCO):
        for dt in direct_triple_boxes:
            already_covered = any(box_iou(dt['bbox'], v['bounding_box']) > 0.40 for v in violations)
            if not already_covered:
                violations.append({
                    "violation_type": "triple_riding",
                    "title": "TRIPLE RIDING VIOLATION (3+ RIDERS)",
                    "detected": True,
                    "confidence": round(float(dt['confidence']), 4),
                    "bounding_box": dt['bbox'],
                    "bike_box": dt['bbox'],
                    "rider_boxes": [],
                    "rider_count": 3,
                    "track_id": track_id,
                    "frame_number": frame_idx,
                    "review_status": "confirmed",
                    "reason": f"Direct triple-riding detection from fine-tuned model ({dt['confidence']:.2f})"
                })

        return violations
