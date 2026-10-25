import os
import cv2
import numpy as np
import time
import uuid
from config import Config
from services.detection_service import DetectionService
from services.tracking_service import TrackingService
from services.helmet_detector import HelmetDetector
from services.triple_riding_detector import TripleRidingDetector
from services.wrong_side_detector import WrongSideDetector
from services.plate_service import PlateOCR
from services.violation_aggregator import ViolationAggregator

class VideoProcessor:
    def __init__(self, detector=None, permitted_direction=None, road_roi=None,
                 enable_vit=False, conf_threshold=None):
        self.detector = detector or DetectionService(conf_threshold=conf_threshold)
        self.permitted_direction = permitted_direction or Config.DEFAULT_PERMITTED_DIRECTION
        self.tracker = TrackingService()
        self.helmet_detector = HelmetDetector(vit_enabled=enable_vit)
        self.triple_detector = TripleRidingDetector()
        self.wrong_side_detector = WrongSideDetector(
            permitted_direction=self.permitted_direction,
            road_roi=road_roi
        )
        self.plate_ocr = PlateOCR()
        self.aggregator = ViolationAggregator()
        self.wrong_direction_detector = None
        if os.path.exists(Config.WRONG_DIRECTION_MODEL_PATH):
            try:
                self.wrong_direction_detector = DetectionService(
                    model_path=Config.WRONG_DIRECTION_MODEL_PATH,
                    conf_threshold=0.45
                )
            except Exception as e:
                print(f"[VideoProcessor] Note: Could not load wrong_direction model: {e}")

    def process_image(self, image_path, output_path=None):
        """
        Processes a single static image safely with EXIF orientation correction.
        Resets per-image state completely so detections from previous uploads cannot contaminate.
        Returns: (annotated_image_path, list_of_violations, detections)
        """
        # Reset per-image aggregator instance completely
        self.aggregator = ViolationAggregator()

        # Safe image loading with EXIF orientation normalization
        frame = None
        try:
            from PIL import Image, ImageOps
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            frame = cv2.imread(image_path)

        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            raise ValueError(f"Unable to read or decode image at: {image_path}")

        h, w = frame.shape[:2]
        if h < 10 or w < 10:
            raise ValueError(f"Image dimensions too small ({w}x{h}) for analysis.")

        # 1. Run YOLO detection
        detections = self.detector.detect(frame)

        # Structured diagnostics and logging
        print(f"\n[VideoProcessor] === PROCESSING IMAGE: {os.path.basename(image_path)} ===")
        print(f"[VideoProcessor] Image shape: {w}x{h}")
        print(f"[VideoProcessor] Models: Primary={getattr(self.detector, 'model_path', 'default')}, Base={getattr(self.detector, 'base_model_path', 'none')}")
        print(f"[VideoProcessor] Raw detections found: {len(detections)}")
        for idx, d in enumerate(detections):
            print(f"  [{idx+1}] {d['class_name']} ({d['confidence']:.2f}) -> bbox: {d['bbox']}")

        # 2. Extract number plates across all detected vehicles (two-wheelers, four-wheelers, etc.)
        plate_results = []
        bikes = [d for d in detections if d['class_name'].lower().replace('_', ' ').strip() in ['motorcycle', 'bike', 'two wheeler', 'two_wheeler', 'scooter']]
        four_wheelers = [d for d in detections if d['class_name'].lower().replace('_', ' ').strip() in ['car', 'truck', 'bus', 'autorickshaw', 'auto']]
        riders = [d for d in detections if d['class_name'].lower().replace('_', ' ').strip() in ['person', 'rider', 'driver', 'pillion']]
        direct_plates = [d for d in detections if 'plate' in d['class_name'].lower()]
        
        # Extract from motorcycles / scooters
        for bike in bikes:
            res = self.plate_ocr.extract_plate_from_vehicle(frame, bike['bbox'], direct_plates=direct_plates, vehicle_type='motorcycle')
            if res:
                res['vehicle_bbox'] = bike['bbox']
                if res.get('normalized_text') != 'UNREADABLE':
                    plate_results.append(res)
                elif res.get('bbox'):
                    plate_results.append(res)

        # Extract from four-wheelers
        for veh in four_wheelers:
            res = self.plate_ocr.extract_plate_from_vehicle(frame, veh['bbox'], direct_plates=direct_plates, vehicle_type='car')
            if res:
                res['vehicle_bbox'] = veh['bbox']
                if res.get('normalized_text') != 'UNREADABLE':
                    plate_results.append(res)
                elif res.get('bbox'):
                    plate_results.append(res)

        # Extract from standalone or unassociated direct plates
        if direct_plates:
            has_readable = any(pr.get('status') in ['readable', 'partial'] and pr.get('normalized_text') not in ['', 'UNREADABLE'] for pr in plate_results)
            if not has_readable or not plate_results:
                for p in direct_plates:
                    px1, py1, px2, py2 = [int(v) for v in p['bbox']]
                    pad_x = max(6, int((px2 - px1) * 0.14))
                    pad_y = max(5, int((py2 - py1) * 0.14))
                    c_px1 = max(0, px1 - pad_x)
                    c_py1 = max(0, py1 - pad_y)
                    c_px2 = min(w, px2 + pad_x)
                    c_py2 = min(h, py2 + pad_y)
                    crop = frame[c_py1:c_py2, c_px1:c_px2]
                    if crop.size > 0:
                        res = self.plate_ocr.extract_text(crop)
                        res['bbox'] = [c_px1, c_py1, c_px2, c_py2]
                        if res.get('status') in ['readable', 'partial'] and res.get('normalized_text') not in ['', 'UNREADABLE']:
                            uid = uuid.uuid4().hex[:8]
                            fname = f"plate_{uid}.jpg"
                            cv2.imwrite(os.path.join(Config.OUTPUTS_EVIDENCE_DIR, fname), crop)
                            res['crop_path'] = f"/outputs/evidence/{fname}"
                            plate_results.append(res)

        # 3. Analyze helmet violations
        helmet_violations = self.helmet_detector.analyze(frame, detections, track_id=None, frame_idx=0)

        # 4. Analyze triple riding
        triple_violations = self.triple_detector.analyze(frame, detections, track_id=None, frame_idx=0, is_video=False)

        # 5. Dedicated Image-Based Wrong-Side Assessment:
        # Evaluates motorcycle orientation, visual cues, road geometry context,
        # and selected permitted direction without requiring ByteTrack or temporal frames.
        wrong_side_violations, wrong_side_info = self.wrong_side_detector.evaluate_image_wrong_side(
            frame, bikes, detections
        )

        # 6. Aggregate violations
        violations = self.aggregator.aggregate_frame(
            frame, helmet_violations, triple_violations, wrong_side_violations,
            plate_results, frame_idx=0, timestamp_sec=0.0, is_video=False
        )

        print(f"[VideoProcessor] Scene Analysis: {len(bikes)} bikes, {len(riders)} riders")
        print(f"[VideoProcessor] Evaluated Violations: Helmet={len(helmet_violations)}, Triple={len(triple_violations)}, WrongSide={len(wrong_side_violations)}")
        print(f"[VideoProcessor] Total Aggregated Violations: {len(violations)}")

        # Build diagnostic dict
        class_summary = {}
        for d in detections:
            c = d['class_name']
            class_summary[c] = class_summary.get(c, 0) + 1

        if len(violations) > 0:
            status_desc = "violations_found"
            diag_msg = f"Detected {len(violations)} traffic violation(s) across {len(detections)} observed objects."
        elif len(detections) > 0:
            status_desc = "compliant"
            diag_msg = f"Observed {len(detections)} traffic objects ({', '.join(f'{k}: {v}' for k, v in class_summary.items())}). No traffic violations detected."
        else:
            status_desc = "no_objects_detected"
            diag_msg = "No traffic objects (motorcycles, riders, helmets) detected at current confidence threshold."

        self.last_diagnostics = {
            'file_name': os.path.basename(image_path),
            'image_dimensions': {'width': w, 'height': h},
            'total_detections': len(detections),
            'class_summary': class_summary,
            'violations_count': len(violations),
            'rider_evaluations': getattr(self.helmet_detector, 'last_rider_evaluations', []),
            'status': status_desc,
            'diagnostic_message': diag_msg,
            'permitted_direction': self.permitted_direction,
            'permitted_direction_label': wrong_side_info.get('selected_direction_label', self.permitted_direction),
            'wrong_side_status': wrong_side_info.get('message', ''),
            'wrong_side_details': wrong_side_info,
            'detections': detections
        }

        # 7. Draw annotations:
        # Suppress raw uncalibrated/conflicting helmet boxes from base detections to prevent
        # drawing confusing false-positive boxes that conflict with validated head analysis.
        violating_bboxes = [v['bounding_box'] for v in violations if v.get('bounding_box')]
        base_detections = []
        for d in detections:
            cname = d['class_name'].lower().replace('_', ' ').strip()
            if 'helmet' in cname:
                continue
            is_violating_bike = False
            if 'bike' in cname or 'motorcycle' in cname:
                for vb in violating_bboxes:
                    if self.aggregator._compute_iou(d['bbox'], vb) > 0.60:
                        is_violating_bike = True
                        break
            if not is_violating_bike:
                base_detections.append(d)

        annotated = self.detector.draw_detections(frame, base_detections)
        rider_evals = getattr(self.helmet_detector, 'last_rider_evaluations', [])
        annotated = self._draw_violation_overlays(annotated, violations, plate_results, rider_evaluations=rider_evals)

        # Save annotated image
        if not output_path:
            bname = os.path.basename(image_path)
            output_path = os.path.join(Config.OUTPUTS_IMAGES_DIR, f"annotated_{bname}")
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, annotated)

        return output_path, violations, detections

    def process_video(self, video_path, output_path=None, progress_callback=None, max_frames=None):
        """
        Processes a video file frame-by-frame with ByteTrack tracking,
        trajectory calculation, wrong-side detection, and deduplication.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or width <= 0 or height <= 0:
            cap.release()
            raise ValueError("Corrupted or empty video stream.")

        if not output_path:
            bname = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(Config.OUTPUTS_VIDEOS_DIR, f"annotated_{bname}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Initialize video writer with universal web-compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        all_violations = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                timestamp_sec = frame_idx / fps

                # 1. Detect objects on current frame
                raw_detections = self.detector.detect(frame)

                # 2. Update multi-object tracker
                tracked_detections = self.tracker.update(raw_detections)

                # 3. Update trajectory histories for tracked motorcycles
                for det in tracked_detections:
                    cname = det['class_name'].lower()
                    if cname in ['motorcycle', 'bike'] and 'track_id' in det:
                        self.wrong_side_detector.update_track(
                            det['track_id'], det['bbox'], frame_idx, timestamp_sec
                        )

                # 4. Analyze violations on tracked objects
                frame_helmet_violations = []
                frame_triple_violations = []
                frame_wrong_side_violations = []

                # Group by motorcycle tracks
                bike_dets = [d for d in tracked_detections if d['class_name'].lower() in ['motorcycle', 'bike']]
                other_veh_dets = [d for d in tracked_detections if d['class_name'].lower().replace('_', ' ').strip() in ['car', 'truck', 'bus', 'autorickshaw', 'auto']]
                for bike in bike_dets:
                    tid = bike.get('track_id')
                    
                    # Helmet check
                    h_viols = self.helmet_detector.analyze(frame, tracked_detections, track_id=tid, frame_idx=frame_idx)
                    frame_helmet_violations.extend(h_viols)

                    # Triple riding check
                    t_viols = self.triple_detector.analyze(frame, tracked_detections, track_id=tid, frame_idx=frame_idx, is_video=True)
                    frame_triple_violations.extend(t_viols)

                    # Wrong-side check
                    if tid is not None:
                        ws_viol = self.wrong_side_detector.evaluate_wrong_side(
                            tid, bike['bbox'], frame.shape, frame_idx=frame_idx
                        )
                        if ws_viol:
                            frame_wrong_side_violations.append(ws_viol)

                # 5. Extract plates occasionally (every 5 frames to conserve CPU)
                plate_results = []
                if frame_idx % 5 == 1:
                    direct_plates = [d for d in tracked_detections if 'plate' in d['class_name'].lower()]
                    for bike in bike_dets:
                        p_res = self.plate_ocr.extract_plate_from_vehicle(frame, bike['bbox'], direct_plates=direct_plates, vehicle_type='motorcycle')
                        if p_res:
                            p_res['vehicle_bbox'] = bike['bbox']
                            p_res['track_id'] = bike.get('track_id')
                            if p_res.get('normalized_text') != 'UNREADABLE':
                                plate_results.append(p_res)
                            elif p_res.get('bbox'):
                                plate_results.append(p_res)
                    for veh in other_veh_dets:
                        p_res = self.plate_ocr.extract_plate_from_vehicle(frame, veh['bbox'], direct_plates=direct_plates, vehicle_type='car')
                        if p_res:
                            p_res['vehicle_bbox'] = veh['bbox']
                            p_res['track_id'] = veh.get('track_id')
                            if p_res.get('normalized_text') != 'UNREADABLE':
                                plate_results.append(p_res)
                            elif p_res.get('bbox'):
                                plate_results.append(p_res)

                # 6. Aggregate & deduplicate frame violations
                frame_v = self.aggregator.aggregate_frame(
                    frame, frame_helmet_violations, frame_triple_violations, frame_wrong_side_violations,
                    plate_results, frame_idx=frame_idx, timestamp_sec=timestamp_sec, is_video=True
                )
                if frame_v:
                    all_violations.extend(frame_v)

                # 7. Draw visual annotations
                annotated = self.detector.draw_detections(frame, tracked_detections)
                annotated = self.wrong_side_detector.draw_trajectories_and_overlay(
                    annotated, frame_wrong_side_violations
                )
                annotated = self._draw_violation_overlays(
                    annotated, frame_helmet_violations + frame_triple_violations + frame_wrong_side_violations, plate_results,
                    rider_evaluations=getattr(self.helmet_detector, 'last_rider_evaluations', [])
                )
                
                # Write frame
                out.write(annotated)

                # Progress reporting
                if progress_callback and frame_idx % 10 == 0:
                    pct = min(100.0, (frame_idx / max(1, total_frames)) * 100.0)
                    progress_callback(frame_idx, total_frames, pct)

        finally:
            cap.release()
            out.release()

        # Retrieve final deduplicated violations across entire video
        final_violations = self.aggregator.get_final_video_violations()
        ws_info = self.wrong_side_detector.get_status_summary(is_video=True)
        self.last_diagnostics = {
            'file_name': os.path.basename(video_path),
            'total_frames': total_frames,
            'violations_count': len(final_violations),
            'status': 'violations_found' if len(final_violations) > 0 else 'compliant',
            'diagnostic_message': f"Processed {total_frames} frames. Found {len(final_violations)} distinct traffic violations." if len(final_violations) > 0 else f"Processed {total_frames} frames. No traffic violations observed.",
            'permitted_direction': self.permitted_direction,
            'permitted_direction_label': ws_info.get('selected_direction_label', self.permitted_direction),
            'wrong_side_status': ws_info.get('message', ''),
            'wrong_side_details': ws_info
        }
        return output_path, final_violations, total_frames

    def _draw_violation_overlays(self, frame, violations, plate_results=None, rider_evaluations=None):
        """
        Draws high-visibility warning pill badges and plates on detected violators,
        and renders explicit status badges (HELMET WORN / UNCERTAIN / NO HELMET) for all riders.
        Deduplicates overlapping bounding boxes to avoid drawing duplicate boxes.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        drawn_boxes = []

        # 1. Draw non-violating rider status badges (HELMET WORN and UNCERTAIN)
        if rider_evaluations:
            for reval in rider_evaluations:
                status = reval.get('status')
                if status == 'NO_HELMET':
                    # Explicitly handled with prominent violation styling below
                    continue
                hbox = reval.get('head_box') or reval.get('rider_box')
                if not hbox or len(hbox) < 4:
                    continue
                hx1, hy1, hx2, hy2 = [int(v) for v in hbox]
                rid = reval.get('rider_id', 'Rider')
                conf = reval.get('confidence', 0.0)

                if status == 'HELMET_WORN':
                    color = (0, 180, 0)  # Green
                    text = f"HELMET WORN: {rid} ({conf:.0%})" if conf > 0 else f"HELMET WORN: {rid}"
                else:  # UNCERTAIN
                    color = (130, 130, 130)  # Neutral Gray / Amber
                    text = f"UNCERTAIN: {rid}"

                # Draw subtle head outline
                cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                by1 = max(0, hy1 - th - 6)
                by2 = hy1
                cv2.rectangle(annotated, (hx1, by1), (hx1 + tw + 6, by2), color, -1)
                cv2.putText(annotated, text, (hx1 + 3, hy1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw confirmed violations (NO HELMET, WRONG SIDE, TRIPLE RIDING)
        for v in violations:
            bbox = v.get('bounding_box')
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(val) for val in bbox]
            vtype = v.get('violation_type', '')
            is_cand = v.get('review_status') == 'insufficient_evidence' or v.get('is_candidate') or v.get('assessment_type') == 'image'
            conf = v.get('confidence', 1.0)
            
            # Badge text and color
            if vtype == 'no_helmet':
                rider_label = v.get('rider_id') or 'Rider'
                text = f"NO HELMET: {rider_label} ({conf:.0%})" if conf is not None else f"NO HELMET: {rider_label}"
                color = (0, 0, 220)  # Red
            elif vtype == 'triple_riding':
                rc = v.get('rider_count', 3)
                text = f"VIOLATION: TRIPLE RIDING ({rc} RIDERS)"
                color = (0, 140, 255)  # Orange
            elif vtype == 'wrong_side':
                if is_cand:
                    text = "POTENTIAL WRONG-SIDE — REVIEW"
                    color = (0, 140, 255)  # Amber / Orange highlight
                else:
                    text = "VIOLATION: WRONG SIDE"
                    color = (0, 0, 255)  # Deep Red
            else:
                text = f"VIOLATION: {vtype.upper()}"
                color = (0, 0, 220)

            # Check if this box or its parent vehicle overlaps an already drawn violation box
            offset_y = 0
            for prev_box, prev_h in drawn_boxes:
                if self.aggregator._compute_iou(prev_box, bbox) > 0.45:
                    offset_y += prev_h + 4

            # Draw prominent violation outline only once per target box
            if offset_y == 0:
                cv2.rectangle(annotated, (max(0, x1 - 2), max(0, y1 - 2)),
                              (min(w - 1, x2 + 2), min(h - 1, y2 + 2)), color, 3)

            # Draw banner tag with intelligent boundary handling
            f_scale = 0.42 if w < 600 else 0.50
            f_thick = 1
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick)
            
            # If box is too close to top edge, place banner just below top edge or inside top
            if y1 < (th + 10 + offset_y):
                banner_y1 = y1 + offset_y
                banner_y2 = y1 + th + 6 + offset_y
                text_y = banner_y2 - 3
            else:
                banner_y1 = max(0, y1 - th - 6 - offset_y)
                banner_y2 = max(th + 6, y1 - offset_y)
                text_y = banner_y2 - 3
                
            banner_x2 = min(w - 1, x1 + tw + 8)
            cv2.rectangle(annotated, (x1, banner_y1), (banner_x2, banner_y2), color, -1)
            cv2.putText(annotated, text, (x1 + 4, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, f_scale, (255, 255, 255), f_thick, cv2.LINE_AA)

            drawn_boxes.append((bbox, th + 8))

        # 3. Draw plate tags
        if plate_results:
            for pr in plate_results:
                if pr.get('status') in ['readable', 'partial'] and pr.get('bbox') and pr.get('normalized_text') not in ['', 'UNREADABLE']:
                    px1, py1, px2, py2 = [int(val) for val in pr['bbox']]
                    ptext = f"PLATE: {pr['normalized_text']}"
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 0, 255), 2)
                    cv2.putText(annotated, ptext, (px1, max(15, py1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        return annotated

    def get_diagnostics(self):
        """Returns the structured diagnostics from the most recent processing run."""
        return getattr(self, 'last_diagnostics', {})
