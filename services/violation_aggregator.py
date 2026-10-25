import os
import cv2
import uuid
from config import Config

class ViolationAggregator:
    def __init__(self, evidence_dir=None):
        self.evidence_dir = evidence_dir or Config.OUTPUTS_EVIDENCE_DIR
        os.makedirs(self.evidence_dir, exist_ok=True)
        # Deduplication cache for video: key -> { 'highest_conf': float, 'violation': dict }
        # key: (violation_type, track_id)
        self.active_tracked_violations = {}

    def save_evidence_crop(self, frame, bbox, violation_type, track_id=None, frame_idx=0, head_box=None):
        """
        Extracts a clean, high-resolution crop around the violation with context margin,
        draws a focused indicator, and saves to the evidence directory.
        For no-helmet violations: focuses on the head/upper torso for clear human auditing.
        Returns the relative web path.
        """
        if frame is None or bbox is None or len(bbox) < 4:
            return None
            
        h, w = frame.shape[:2]
        
        # For no-helmet, if a distinct head_box is provided, crop around head with 30% margin
        crop_box = head_box if (violation_type == 'no_helmet' and head_box and len(head_box) == 4) else bbox
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        
        # Add context margin (25% for small head crops, 15% for vehicle crops)
        margin_pct = 0.30 if violation_type == 'no_helmet' else 0.15
        bw = max(10, x2 - x1)
        bh = max(10, y2 - y1)
        mx = int(bw * margin_pct)
        my = int(bh * margin_pct)
        
        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)
        
        crop = frame[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            return None
            
        # Draw focused bounding box inside the cropped evidence
        box_in_crop_x1 = max(0, x1 - cx1)
        box_in_crop_y1 = max(0, y1 - cy1)
        box_in_crop_x2 = min(crop.shape[1] - 1, x2 - cx1)
        box_in_crop_y2 = min(crop.shape[0] - 1, y2 - cy1)
        
        color = (0, 0, 220) if violation_type == 'no_helmet' else (
            (0, 140, 255) if violation_type == 'triple_riding' else (0, 0, 220)
        )
        cv2.rectangle(crop, (box_in_crop_x1, box_in_crop_y1), (box_in_crop_x2, box_in_crop_y2), color, 2)
        
        # Label badge in crop
        badge_text = violation_type.upper().replace('_', ' ')
        cv2.putText(crop, badge_text, (box_in_crop_x1 + 4, max(16, box_in_crop_y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Unique evidence filename
        uid = str(uuid.uuid4())[:8]
        filename = f"ev_{violation_type}_f{frame_idx}_t{track_id or 'img'}_{uid}.jpg"
        abs_path = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(abs_path, crop)
        
        # Return web-accessible path
        return f"/outputs/evidence/{filename}"

    @staticmethod
    def _compute_iou(box1, box2):
        if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter_area <= 0.0:
            return 0.0
        b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def aggregate_frame(self, frame, helmet_violations, triple_violations, wrong_side_violations,
                        plate_ocr_results, frame_idx=0, timestamp_sec=0.0, is_video=False):
        """
        Combines and deduplicates violations from all detectors for a given frame.
        Guarantees that:
        - Distinct riders on the same motorcycle produce separate helmet violation events.
        - Exactly one triple-riding event is produced per qualifying motorcycle.
        - Number-plate information is associated with matching motorcycles.
        """
        frame_violations = []
        raw_list = []
        
        if helmet_violations:
            raw_list.extend(helmet_violations)
        if triple_violations:
            raw_list.extend(triple_violations)
        if wrong_side_violations:
            raw_list.extend(wrong_side_violations)

        # Collect distinct vehicle target boxes across all violations in this frame
        all_vehicle_boxes = []
        for rv in raw_list:
            v_box = rv.get('bike_box') or rv.get('bounding_box')
            if v_box and len(v_box) >= 4:
                if not any(self._compute_iou(v_box, existing) > 0.60 for existing in all_vehicle_boxes):
                    all_vehicle_boxes.append(v_box)
        is_multi_vehicle = len(all_vehicle_boxes) > 1

        for v in raw_list:
            v_type = v['violation_type']
            track_id = v.get('track_id')
            bbox = v.get('bounding_box', [])
            head_box = v.get('head_box', bbox)
            bike_box = v.get('bike_box')
            conf = v.get('confidence')
            rider_id = v.get('rider_id')
            is_candidate = v.get('is_candidate', False) or (v.get('review_status') == 'insufficient_evidence')
            detected = v.get('detected', not is_candidate)

            # Associate plate OCR info: match to vehicle box (bike_box or bbox)
            plate_info = {'text': 'UNREADABLE', 'confidence': None, 'status': 'none_detected', 'raw_text': None, 'crop_path': None}
            if plate_ocr_results:
                best_plate = None
                best_score = -1.0
                target_box = bike_box or bbox

                for p_res in plate_ocr_results:
                    # A. Match by track_id if both are tracked
                    p_tid = p_res.get('track_id')
                    if track_id is not None and p_tid is not None and track_id == p_tid:
                        best_plate = p_res
                        break

                    # B. Match by tagged vehicle_bbox
                    vbox = p_res.get('vehicle_bbox')
                    if vbox and target_box and len(target_box) >= 4:
                        iou = self._compute_iou(vbox, target_box)
                        if iou > 0.40:
                            match_score = 10.0 + iou * 2.0 + float(p_res.get('confidence') or 0.0)
                            if p_res.get('status') == 'readable':
                                match_score += 5.0
                            if match_score > best_score:
                                best_score = match_score
                                best_plate = p_res
                                continue

                    # C. Spatial containment / proximity match
                    pbox = p_res.get('bbox')
                    if pbox and target_box and len(target_box) >= 4:
                        pcx = (pbox[0] + pbox[2]) / 2.0
                        pcy = (pbox[1] + pbox[3]) / 2.0
                        in_x = (target_box[0] - 25 <= pcx <= target_box[2] + 25)
                        in_y = (target_box[1] - 20 <= pcy <= target_box[3] + 35)
                        if in_x and in_y:
                            # Verify target_box is the closest vehicle to this plate center
                            dist_to_target = (pcx - (target_box[0] + target_box[2]) / 2.0) ** 2 + (pcy - (target_box[1] + target_box[3]) / 2.0) ** 2
                            is_closest = True
                            for other_vbox in all_vehicle_boxes:
                                if other_vbox is not target_box and self._compute_iou(other_vbox, target_box) < 0.40:
                                    dist_to_other = (pcx - (other_vbox[0] + other_vbox[2]) / 2.0) ** 2 + (pcy - (other_vbox[1] + other_vbox[3]) / 2.0) ** 2
                                    if dist_to_other < dist_to_target:
                                        is_closest = False
                                        break
                            if is_closest:
                                match_score = 2.0 + float(p_res.get('confidence') or 0.0)
                                if p_res.get('status') == 'readable':
                                    match_score += 2.0
                                elif p_res.get('status') == 'partial':
                                    match_score += 0.5
                                if match_score > best_score:
                                    best_score = match_score
                                    best_plate = p_res

                # D. Fallback only if there is strictly ONE vehicle in the scene
                # In multi-vehicle scenes, NEVER assign a plate from one vehicle to another!
                if not best_plate and not is_multi_vehicle:
                    for p_res in plate_ocr_results:
                        if p_res.get('status') in ['readable', 'partial'] and p_res.get('normalized_text') not in ['', 'UNREADABLE']:
                            best_plate = p_res
                            break

                if best_plate:
                    norm_text = best_plate.get('normalized_text')
                    p_status = best_plate.get('status', 'readable' if norm_text and norm_text != 'UNREADABLE' else 'unreadable')
                    plate_info = {
                        'text': norm_text if p_status != 'unreadable' else 'UNREADABLE',
                        'confidence': best_plate.get('confidence'),
                        'status': p_status,
                        'raw_text': best_plate.get('raw_text'),
                        'crop_path': best_plate.get('crop_path')
                    }

            # Standardized title & review status
            if is_candidate:
                title = v.get('title') or f"{v_type.upper().replace('_', ' ')} — NEEDS REVIEW"
                rev_status = v.get('review_status', 'insufficient_evidence')
                status_code = v.get('status') or "pending_review"
            else:
                title = v.get('title') or f"{v_type.upper().replace('_', ' ')}"
                rev_status = v.get('review_status', 'pending' if (conf is not None and conf < 0.65) else 'confirmed')
                status_code = v.get('status', 'confirmed' if rev_status == 'confirmed' else 'pending_review')

            assessment_type = v.get('assessment_type', 'video' if is_video else 'image')
            event = {
                "violation_type": v_type,
                "title": title,
                "detected": detected,
                "is_candidate": is_candidate,
                "assessment_type": assessment_type,
                "confidence": round(float(conf), 4) if conf is not None else None,
                "detector_confidence": round(float(v['detector_confidence']), 4) if v.get('detector_confidence') is not None else None,
                "bounding_box": [round(float(c), 2) for c in bbox] if bbox else [],
                "head_box": [round(float(c), 2) for c in head_box] if head_box else [],
                "bike_box": [round(float(c), 2) for c in bike_box] if bike_box else None,
                "rider_id": rider_id,
                "rider_count": v.get('rider_count'),
                "track_id": track_id,
                "frame_number": frame_idx if is_video else None,
                "evidence_frame": frame_idx if is_video else None,
                "timestamp_sec": round(float(timestamp_sec), 2) if is_video else 0.0,
                "plate_text": plate_info['text'],
                "plate_confidence": plate_info['confidence'],
                "plate_status": plate_info['status'],
                "raw_plate_text": plate_info.get('raw_text'),
                "plate_crop_path": plate_info.get('crop_path'),
                "review_status": rev_status,
                "status": status_code,
                "reason": v.get('reason', ''),
                "estimated_orientation": v.get('estimated_orientation'),
                "estimated_movement": v.get('estimated_movement'),
                "permitted_direction": v.get('permitted_direction'),
                "permitted_direction_label": v.get('permitted_direction_label'),
                "evidence_path": None,
                "evidence_image": None
            }

            if is_video and track_id is not None:
                # Video deduplication across frames by track_id
                dedup_key = (v_type, track_id, rider_id or 0)
                if dedup_key not in self.active_tracked_violations:
                    evidence_path = self.save_evidence_crop(frame, bbox, v_type, track_id, frame_idx, head_box)
                    event['evidence_path'] = evidence_path
                    event['evidence_image'] = evidence_path
                    self.active_tracked_violations[dedup_key] = {
                        'highest_conf': conf,
                        'event': event
                    }
                    frame_violations.append(event)
                else:
                    existing = self.active_tracked_violations[dedup_key]
                    existing_event = existing['event']
                    if existing_event.get('is_candidate') and not is_candidate:
                        new_evidence = self.save_evidence_crop(frame, bbox, v_type, track_id, frame_idx, head_box)
                        existing['highest_conf'] = conf
                        existing_event['detected'] = True
                        existing_event['is_candidate'] = False
                        existing_event['title'] = title
                        existing_event['confidence'] = round(float(conf), 4) if conf is not None else None
                        existing_event['review_status'] = rev_status
                        existing_event['status'] = status_code
                        existing_event['reason'] = v.get('reason', '')
                        existing_event['frame_number'] = frame_idx
                        existing_event['evidence_frame'] = frame_idx
                        existing_event['timestamp_sec'] = round(float(timestamp_sec), 2)
                        existing_event['bounding_box'] = [round(float(c), 2) for c in bbox]
                        existing_event['head_box'] = [round(float(c), 2) for c in head_box]
                        existing_event['evidence_path'] = new_evidence
                        existing_event['evidence_image'] = new_evidence
                    elif conf is not None and (existing['highest_conf'] is None or conf > existing['highest_conf']):
                        new_evidence = self.save_evidence_crop(frame, bbox, v_type, track_id, frame_idx, head_box)
                        existing['highest_conf'] = conf
                        existing_event['confidence'] = round(float(conf), 4)
                        existing_event['frame_number'] = frame_idx
                        existing_event['evidence_frame'] = frame_idx
                        existing_event['timestamp_sec'] = round(float(timestamp_sec), 2)
                        existing_event['bounding_box'] = [round(float(c), 2) for c in bbox]
                        existing_event['head_box'] = [round(float(c), 2) for c in head_box]
                        existing_event['evidence_path'] = new_evidence
                        existing_event['evidence_image'] = new_evidence
            else:
                # Spatial deduplication for static images:
                # CRITICAL: For no_helmet, compare head_box (not whole body, not motorcycle!)
                # Two violations are duplicates ONLY if their head boxes overlap with high IoU (> 0.55).
                is_duplicate = False
                for existing_event in frame_violations:
                    if existing_event['violation_type'] == v_type:
                        if v_type == 'no_helmet':
                            target_box1 = existing_event.get('head_box') or existing_event.get('bounding_box')
                            target_box2 = head_box or bbox
                            iou = self._compute_iou(target_box1, target_box2)
                            direct_overlap = self._compute_iou(existing_event.get('bounding_box'), bbox)
                            
                            if iou > 0.45 or direct_overlap > 0.40:
                                is_duplicate = True
                                # If one was pending and one was confirmed, upgrade to confirmed
                                if existing_event.get('review_status') != 'confirmed' and rev_status == 'confirmed':
                                    existing_event['review_status'] = 'confirmed'
                                    existing_event['title'] = title
                                    existing_event['status'] = 'confirmed'
                                    existing_event['reason'] = v.get('reason', '')
                                if conf is not None and (existing_event.get('confidence') is None or conf > existing_event['confidence']):
                                    existing_event['confidence'] = round(float(conf), 4)
                                    existing_event['bounding_box'] = [round(float(c), 2) for c in bbox]
                                    existing_event['head_box'] = [round(float(c), 2) for c in head_box]
                                if rider_id and (not existing_event.get('rider_id') or 'Direct' in str(existing_event.get('rider_id'))):
                                    existing_event['rider_id'] = rider_id
                                break
                        else:
                            iou = self._compute_iou(existing_event.get('bounding_box'), bbox)
                            if iou > 0.50:
                                is_duplicate = True
                                if conf is not None and (existing_event.get('confidence') is None or conf > existing_event['confidence']):
                                    existing_event['confidence'] = round(float(conf), 4)
                                    existing_event['bounding_box'] = [round(float(c), 2) for c in bbox]
                                break
                            
                if not is_duplicate:
                    evidence_path = self.save_evidence_crop(frame, bbox, v_type, track_id, frame_idx, head_box)
                    event['evidence_path'] = evidence_path
                    event['evidence_image'] = evidence_path
                    frame_violations.append(event)

        return frame_violations

    def get_final_video_violations(self):
        """
        Returns the deduplicated list of unique violations recorded during video processing.
        """
        return [item['event'] for item in self.active_tracked_violations.values()]
