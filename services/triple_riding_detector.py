import cv2
import numpy as np
from config import Config

def box_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

class TripleRidingDetector:
    def __init__(self, max_permitted_riders=None, min_consecutive_frames=None):
        self.max_permitted_riders = max_permitted_riders or Config.MAX_PERMITTED_RIDERS
        self.min_consecutive_frames = min_consecutive_frames or Config.MIN_CONSECUTIVE_FRAMES_TRIPLE
        # Track history for video debounce: {track_id or bike_key: [count1, count2, ...]}
        self.bike_history = {}

    @staticmethod
    def is_rider_on_bike(person_box, bike_box):
        """
        Determines whether a detected person is physically mounted on the motorcycle.
        Criteria:
        1. Horizontal center of the person must fall within the horizontal span of the motorcycle (with 25% tolerance).
        2. Vertical extent: person's lower body must intersect or sit inside the motorcycle frame.
        3. Area overlap: at least 15% of the person's bounding box area overlaps the motorcycle.
        """
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bike_box
        
        person_cx = (px1 + px2) / 2.0
        bike_w = bx2 - bx1
        
        # Horizontal alignment check
        h_margin = bike_w * 0.25
        if not (bx1 - h_margin <= person_cx <= bx2 + h_margin):
            return False
            
        # Vertical sanity check: person cannot be entirely above or entirely below the bike
        if py2 < by1 or py1 > by2:
            return False
            
        # Overlap area check
        xA = max(px1, bx1)
        yA = max(py1, by1)
        xB = min(px2, bx2)
        yB = min(py2, by2)
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        rider_area = max(1.0, (px2 - px1) * (py2 - py1))
        
        return (inter / rider_area) > 0.15

    @staticmethod
    def deduplicate_overlapping_riders(riders, iou_thresh=0.40):
        """
        Suppresses redundant/duplicate person bounding boxes on the same rider
        caused by detector jitter or partial occlusion, returning distinct riders.
        """
        if not riders:
            return []
            
        # Sort by confidence descending
        sorted_riders = sorted(riders, key=lambda r: r['confidence'], reverse=True)
        distinct = []
        
        for r in sorted_riders:
            r_box = r['bbox']
            # If box significantly overlaps an already accepted distinct rider, skip
            if any(box_iou(r_box, d['bbox']) > iou_thresh for d in distinct):
                continue
            distinct.append(r)
            
        return distinct

    def analyze(self, frame, detections, track_id=None, frame_idx=0, is_video=False):
        """
        Analyzes a single image or video frame to identify triple riding violations.
        Returns:
            list of structured violation dicts.
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

        # 1. Direct 'triple riding' class detections from fine-tuned model
        for dt in direct_triple_boxes:
            violations.append({
                "violation_type": "triple_riding",
                "detected": True,
                "confidence": round(dt['confidence'], 4),
                "bounding_box": dt['bbox'],
                "rider_boxes": [],
                "rider_count": 3,
                "track_id": track_id,
                "frame_number": frame_idx,
                "review_status": "confirmed",
                "reason": f"Direct triple-riding detection from fine-tuned model ({dt['confidence']:.2f})"
            })

        # 2. Geometric Rider Counting per Motorcycle
        for b_idx, bike in enumerate(bikes):
            b_box = bike['bbox']
            associated_riders = []
            
            for person in persons:
                p_box = person['bbox']
                if self.is_rider_on_bike(p_box, b_box):
                    associated_riders.append(person)

            # Deduplicate overlapping person detections to isolate distinct physical riders
            distinct_riders = self.deduplicate_overlapping_riders(associated_riders, iou_thresh=0.40)
            rider_count = len(distinct_riders)
            
            # Key for tracking history
            bike_key = f"{track_id or b_idx}_{round(b_box[0]/50)}_{round(b_box[1]/50)}"
            if bike_key not in self.bike_history:
                self.bike_history[bike_key] = []
            self.bike_history[bike_key].append(rider_count)
            if len(self.bike_history[bike_key]) > 15:
                self.bike_history[bike_key].pop(0)

            # Check violation condition: distinct riders > max_permitted_riders (typically 2)
            if rider_count > self.max_permitted_riders:
                rider_confs = [r['confidence'] for r in distinct_riders]
                bike_conf = bike['confidence']
                avg_rider_conf = sum(rider_confs) / len(rider_confs) if rider_confs else 0.5
                combined_conf = round(0.55 * bike_conf + 0.45 * avg_rider_conf, 4)

                is_confirmed = True
                if is_video and len(self.bike_history[bike_key]) >= self.min_consecutive_frames:
                    recent_counts = self.bike_history[bike_key][-self.min_consecutive_frames:]
                    violating_frames = sum(1 for c in recent_counts if c > self.max_permitted_riders)
                    if violating_frames < 2:
                        is_confirmed = False

                # Prevent adding duplicate violation if direct triple riding already covered this bike
                already_covered = any(
                    box_iou(b_box, v['bounding_box']) > 0.50
                    for v in violations if v['violation_type'] == 'triple_riding'
                )

                if not already_covered:
                    violations.append({
                        "violation_type": "triple_riding",
                        "detected": True,
                        "confidence": combined_conf,
                        "bounding_box": b_box,
                        "rider_boxes": [r['bbox'] for r in distinct_riders],
                        "rider_count": rider_count,
                        "track_id": track_id,
                        "frame_number": frame_idx,
                        "review_status": "confirmed" if is_confirmed else "pending",
                        "reason": f"Detected {rider_count} distinct riders on single motorcycle (permitted: {self.max_permitted_riders})"
                    })

        return violations
