import os
import math
import cv2
import numpy as np
from config import Config

DIRECTION_CONFIG = {
    'auto': {
        'label': 'Automatic / Unknown',
        'target_angle': None,
        'description': 'Direction inferred from vehicle movement consensus',
        'coord_meaning': 'Dynamic consensus vector'
    },
    'unknown': {
        'label': 'Automatic / Unknown',
        'target_angle': None,
        'description': 'Direction inferred from vehicle movement consensus',
        'coord_meaning': 'Dynamic consensus vector'
    },
    'downward': {
        'label': 'Downward Flow (Top to Bottom)',
        'target_angle': 90.0,
        'description': 'Traffic traveling Southward (Top to Bottom)',
        'coord_meaning': 'Increasing Y coordinate (+Y)'
    },
    'upward': {
        'label': 'Upward Flow (Bottom to Top)',
        'target_angle': 270.0,
        'description': 'Traffic traveling Northward (Bottom to Top)',
        'coord_meaning': 'Decreasing Y coordinate (-Y)'
    },
    'leftward': {
        'label': 'Leftward Flow (Right to Left)',
        'target_angle': 180.0,
        'description': 'Traffic traveling Westward (Right to Left)',
        'coord_meaning': 'Decreasing X coordinate (-X)'
    },
    'right_to_left': {
        'label': 'Leftward Flow (Right to Left)',
        'target_angle': 180.0,
        'description': 'Traffic traveling Westward (Right to Left)',
        'coord_meaning': 'Decreasing X coordinate (-X)'
    },
    'rightward': {
        'label': 'Rightward Flow (Left to Right)',
        'target_angle': 0.0,
        'description': 'Traffic traveling Eastward (Left to Right)',
        'coord_meaning': 'Increasing X coordinate (+X)'
    },
    'left_to_right': {
        'label': 'Rightward Flow (Left to Right)',
        'target_angle': 0.0,
        'description': 'Traffic traveling Eastward (Left to Right)',
        'coord_meaning': 'Increasing X coordinate (+X)'
    }
}

class WrongSideDetector:
    def __init__(self, permitted_direction=None, road_roi=None, min_displacement=None, min_frames=None, orientation_model_path=None):
        """
        permitted_direction:
          - 'auto': Automatically infer permitted traffic flow direction from vehicle trajectories consensus.
          - 'downward': vehicle y coordinates should increase (traveling top to bottom, angle ~90°)
          - 'upward': vehicle y coordinates should decrease (traveling bottom to top, angle ~270°)
          - 'leftward' / 'right_to_left': vehicle x coordinates should decrease (traveling right to left, angle ~180°)
          - 'rightward' / 'left_to_right': vehicle x coordinates should increase (traveling left to right, angle ~0°)
        """
        raw_direction = permitted_direction or Config.DEFAULT_PERMITTED_DIRECTION
        if isinstance(raw_direction, str):
            raw_direction = raw_direction.strip().lower()
        if raw_direction not in DIRECTION_CONFIG:
            raw_direction = 'auto'
        self.permitted_direction = raw_direction

        self.road_roi = road_roi  # np.array polygon if provided
        self.min_displacement = min_displacement or Config.MIN_TRACK_DISPLACEMENT
        self.min_frames = min_frames or Config.MIN_TRACK_FRAMES
        
        # Track trajectories: {track_id: [{'center': (cx, cy), 'bbox': bbox, 'frame': int, 'time': float}, ...]}
        self.trajectories = {}
        # Track motion vectors: {track_id: {'dx': float, 'dy': float, 'displacement': float, 'angle': float}}
        self.track_vectors = {}
        # Flow consensus state
        self.flow_status = 'pending' # 'flow_established' | 'insufficient_motion_evidence' | 'direction_uncertain'
        self.flow_angle = None
        self.flow_coherence = 0.0
        self.flow_message = "Wrong-side status could not be verified because reliable road-direction evidence was unavailable."

        # Orientation model configuration
        self.orientation_model_path = orientation_model_path or getattr(Config, 'WRONG_DIRECTION_MODEL_PATH', None)
        self._orientation_model = None

    def get_orientation_model(self):
        """Lazily loads the trained orientation / wrong-direction model if available."""
        if self._orientation_model is None:
            if self.orientation_model_path and os.path.exists(self.orientation_model_path):
                try:
                    from ultralytics import YOLO
                    self._orientation_model = YOLO(self.orientation_model_path)
                except Exception as e:
                    print(f"[WrongSideDetector] Warning: Could not load orientation model: {e}")
                    self._orientation_model = False
            else:
                self._orientation_model = False
        return self._orientation_model if self._orientation_model not in (None, False) else None

    @staticmethod
    def _box_iou(box1, box2):
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

    def analyze_road_context(self, frame):
        """
        Analyzes road geometry and visible lane markings using edge detection
        and Hough line transform on the road region.
        Returns:
            dict with 'road_axis' ('longitudinal', 'lateral', 'unknown', or 'ambiguous'),
                      'dominant_angle', 'confidence', 'lines_count'.
        """
        if frame is None or getattr(frame, 'size', 0) == 0:
            return {
                'road_axis': 'unknown',
                'dominant_angle': None,
                'confidence': 0.0,
                'lines_count': 0,
                'status': 'insufficient_road_context'
            }

        h, w = frame.shape[:2]
        if h < 30 or w < 30:
            return {
                'road_axis': 'unknown',
                'dominant_angle': None,
                'confidence': 0.0,
                'lines_count': 0,
                'status': 'insufficient_road_context'
            }

        # Focus on lower 60% of frame where road typically lies
        roi_y1 = int(h * 0.40)
        road_crop = frame[roi_y1:h, :]
        
        try:
            gray = cv2.cvtColor(road_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Detect line segments
            min_len = max(20, int(w * 0.08))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=25)
        except Exception:
            return {
                'road_axis': 'unknown',
                'dominant_angle': None,
                'confidence': 0.0,
                'lines_count': 0,
                'status': 'insufficient_road_context'
            }

        if lines is None or len(lines) == 0:
            return {
                'road_axis': 'unknown',
                'dominant_angle': None,
                'confidence': 0.0,
                'lines_count': 0,
                'status': 'insufficient_road_context'
            }

        longitudinal_angles = []
        lateral_angles = []

        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = float(l[0]), float(l[1]), float(l[2]), float(l[3])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 25:
                continue
            ang = (math.degrees(math.atan2(dy, dx)) + 360) % 180  # 0 to 180
            # Longitudinal (roughly vertical lines, 50° to 130°)
            if 50.0 <= ang <= 130.0:
                longitudinal_angles.append(ang)
            # Lateral (roughly horizontal lines, < 35° or > 145°)
            elif ang <= 35.0 or ang >= 145.0:
                lateral_angles.append(ang)

        long_cnt = len(longitudinal_angles)
        lat_cnt = len(lateral_angles)
        total_valid = long_cnt + lat_cnt

        if total_valid < 2:
            return {
                'road_axis': 'unknown',
                'dominant_angle': None,
                'confidence': 0.2,
                'lines_count': total_valid,
                'status': 'insufficient_road_context'
            }

        if long_cnt > lat_cnt * 1.5:
            sin_sum = sum(math.sin(math.radians(2 * a)) for a in longitudinal_angles)
            cos_sum = sum(math.cos(math.radians(2 * a)) for a in longitudinal_angles)
            avg_2a = math.degrees(math.atan2(sin_sum, cos_sum))
            avg_ang = (avg_2a / 2.0) % 180
            conf = min(0.90, 0.40 + (long_cnt / (total_valid + 2)) * 0.50)
            return {
                'road_axis': 'longitudinal',
                'dominant_angle': round(avg_ang, 1),
                'confidence': round(conf, 2),
                'lines_count': long_cnt,
                'status': 'road_axis_detected'
            }
        elif lat_cnt > long_cnt * 1.5:
            sin_sum = sum(math.sin(math.radians(2 * a)) for a in lateral_angles)
            cos_sum = sum(math.cos(math.radians(2 * a)) for a in lateral_angles)
            avg_2a = math.degrees(math.atan2(sin_sum, cos_sum))
            avg_ang = (avg_2a / 2.0) % 180
            conf = min(0.90, 0.40 + (lat_cnt / (total_valid + 2)) * 0.50)
            return {
                'road_axis': 'lateral',
                'dominant_angle': round(avg_ang, 1),
                'confidence': round(conf, 2),
                'lines_count': lat_cnt,
                'status': 'road_axis_detected'
            }
        else:
            return {
                'road_axis': 'ambiguous',
                'dominant_angle': None,
                'confidence': 0.35,
                'lines_count': total_valid,
                'status': 'road_geometry_ambiguous'
            }

    def estimate_motorcycle_orientation(self, frame, bike_bbox, all_detections=None, road_context=None, ml_detections=None):
        """
        Estimates a motorcycle's orientation in the image frame using:
        1. Trained orientation / wrong-direction model if available.
        2. Visual features: aspect ratio, rider posture/head offset, license plate position, headlight/taillight distribution.
        Returns:
            dict containing:
                'orientation': 'downward' | 'upward' | 'leftward' | 'rightward' | 'ambiguous' | None
                'orientation_label': readable description
                'angle': estimated angle (e.g. 90.0, 270.0, 180.0, 0.0)
                'confidence': float 0.0 - 1.0
                'aspect_ratio': float
                'ml_class': str or None
                'ml_confidence': float or None
                'status': 'orientation_estimated' | 'ambiguous' | 'insufficient_features'
        """
        if frame is None or bike_bbox is None or len(bike_bbox) < 4:
            return {
                'orientation': None,
                'orientation_label': 'Unknown / Insufficient Data',
                'angle': None,
                'confidence': 0.0,
                'aspect_ratio': 1.0,
                'ml_class': None,
                'ml_confidence': None,
                'status': 'insufficient_features'
            }

        h_f, w_f = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in bike_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f, x2), min(h_f, y2)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        ar = bw / float(bh)

        # 1. Match against ML orientation model detections if available
        matched_ml_class = None
        matched_ml_conf = None
        if ml_detections:
            best_iou = 0.0
            for md in ml_detections:
                iou = self._box_iou(bike_bbox, md['bbox'])
                if iou > best_iou and iou >= 0.25:
                    best_iou = iou
                    matched_ml_class = md['class_name']
                    matched_ml_conf = md.get('confidence', 0.5)

        # 2. Check rider head/body offset relative to bike center
        rider_h_offset = 0.0
        rider_found = False
        has_plate_lower = False

        if all_detections:
            for det in all_detections:
                cname = det.get('class_name', '').lower()
                bbox = det.get('bbox', [])
                if not bbox or len(bbox) < 4:
                    continue
                # Rider check
                if any(r in cname for r in ['person', 'rider', 'driver', 'pillion']):
                    rx1, ry1, rx2, ry2 = bbox
                    rcx = (rx1 + rx2) / 2.0
                    bcx = (x1 + x2) / 2.0
                    if (x1 - 0.25 * bw <= rcx <= x2 + 0.25 * bw) and (y1 - 0.3 * bh <= ry2 <= y2 + 0.1 * bh):
                        rider_h_offset = (rcx - bcx) / float(bw)
                        rider_found = True
                # License plate check
                elif 'plate' in cname:
                    px1, py1, px2, py2 = bbox
                    pcy = (py1 + py2) / 2.0
                    if (x1 - 5 <= px1 and px2 <= x2 + 10) and (pcy >= y1 + 0.35 * bh):
                        has_plate_lower = True

        # 3. Analyze pixel features in crop
        crop = frame[y1:y2, x1:x2]
        has_red_taillight = False
        has_bright_headlight_lower = False
        if crop.size > 0 and bh > 10 and bw > 10:
            try:
                # Lower half analysis
                bottom_half = crop[int(bh * 0.5):, :]
                if bottom_half.size > 0:
                    hsv_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
                    mask1 = cv2.inRange(hsv_bottom, np.array([0, 70, 60]), np.array([10, 255, 255]))
                    mask2 = cv2.inRange(hsv_bottom, np.array([170, 70, 60]), np.array([180, 255, 255]))
                    red_pixels = cv2.countNonZero(mask1 | mask2)
                    total_pixels = bottom_half.shape[0] * bottom_half.shape[1]
                    if total_pixels > 0 and (red_pixels / float(total_pixels)) > 0.02:
                        has_red_taillight = True

                    gray_bottom = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
                    bright_pixels = cv2.countNonZero(cv2.inRange(gray_bottom, 210, 255))
                    if total_pixels > 0 and (bright_pixels / float(total_pixels)) > 0.04:
                        has_bright_headlight_lower = True
            except Exception:
                pass

        # 4. Integrate signals to estimate orientation
        orientation = None
        angle = None
        conf = 0.0
        reason_cues = []

        # Case A: Strong lateral profile (aspect ratio >= 1.15)
        if ar >= 1.15:
            reason_cues.append(f"lateral aspect ratio ({ar:.2f})")
            if rider_found and rider_h_offset < -0.06:
                orientation = 'leftward'
                angle = 180.0
                conf = 0.78
                reason_cues.append("rider posture facing leftward")
            elif rider_found and rider_h_offset > 0.06:
                orientation = 'rightward'
                angle = 0.0
                conf = 0.78
                reason_cues.append("rider posture facing rightward")
            else:
                if crop.size > 0:
                    left_w = crop[:, :bw // 2]
                    right_w = crop[:, bw // 2:]
                    if left_w.size > 0 and right_w.size > 0:
                        l_bright = np.mean(left_w)
                        r_bright = np.mean(right_w)
                        if l_bright > r_bright + 18:
                            orientation = 'leftward'
                            angle = 180.0
                            conf = 0.65
                            reason_cues.append("front illumination on left")
                        elif r_bright > l_bright + 18:
                            orientation = 'rightward'
                            angle = 0.0
                            conf = 0.65
                            reason_cues.append("front illumination on right")
                        else:
                            orientation = 'ambiguous'
                            conf = 0.35
                else:
                    orientation = 'ambiguous'
                    conf = 0.30

        # Case B: Longitudinal profile (aspect ratio < 1.15, typical front/rear)
        else:
            reason_cues.append(f"longitudinal aspect ratio ({ar:.2f})")
            # Signal 1: ML model
            if matched_ml_class == 'Wrong-Direction' and (matched_ml_conf or 0) >= 0.35:
                if self.permitted_direction == 'downward':
                    orientation = 'upward'
                    angle = 270.0
                elif self.permitted_direction == 'upward':
                    orientation = 'downward'
                    angle = 90.0
                else:
                    orientation = 'upward'
                    angle = 270.0
                conf = max(conf, float(matched_ml_conf))
                reason_cues.append(f"ML wrong-direction classifier ({matched_ml_conf:.2f})")
            elif matched_ml_class == 'Right-Direction' and (matched_ml_conf or 0) >= 0.35:
                if self.permitted_direction == 'upward':
                    orientation = 'upward'
                    angle = 270.0
                else:
                    orientation = 'downward'
                    angle = 90.0
                conf = max(conf, float(matched_ml_conf))
                reason_cues.append(f"ML right-direction classifier ({matched_ml_conf:.2f})")

            # Signal 2: License plate & taillight (rear facing = upward travel away from camera)
            if has_plate_lower or has_red_taillight:
                if orientation is None or conf < 0.70:
                    orientation = 'upward'
                    angle = 270.0
                    conf = max(conf, 0.72)
                    cues = []
                    if has_plate_lower: cues.append("rear license plate")
                    if has_red_taillight: cues.append("taillight illumination")
                    reason_cues.append(" + ".join(cues))
            # Signal 3: Headlamp / frontal illumination (facing camera = downward travel)
            elif has_bright_headlight_lower and not has_plate_lower:
                if orientation is None or conf < 0.70:
                    orientation = 'downward'
                    angle = 90.0
                    conf = max(conf, 0.68)
                    reason_cues.append("frontal headlamp illumination")

            # Fallback if no specific feature distinguished front from rear
            if orientation is None:
                orientation = 'ambiguous'
                conf = 0.30
                reason_cues.append("symmetrical longitudinal profile")

        # Human readable label
        orient_labels = {
            'downward': 'Downward (Top to Bottom)',
            'upward': 'Upward (Bottom to Top)',
            'leftward': 'Leftward (Right to Left)',
            'rightward': 'Rightward (Left to Right)',
            'ambiguous': 'Ambiguous / Indeterminate'
        }
        label = orient_labels.get(orientation, 'Unknown')
        status = 'orientation_estimated' if orientation and orientation != 'ambiguous' and conf >= 0.50 else 'ambiguous'

        return {
            'orientation': orientation,
            'orientation_label': label,
            'angle': angle,
            'confidence': round(conf, 2),
            'aspect_ratio': round(ar, 2),
            'ml_class': matched_ml_class,
            'ml_confidence': round(matched_ml_conf, 2) if matched_ml_conf is not None else None,
            'status': status,
            'cues': reason_cues
        }

    def evaluate_image_wrong_side(self, frame, bike_detections, all_detections=None):
        """
        Executes dedicated image-based wrong-side assessment for uploaded images.
        Evaluates motorcycle orientation, road geometry/context, and permitted direction.
        Returns:
            (violations_list, summary_dict)
        """
        all_detections = all_detections or []
        bike_detections = bike_detections or []

        # 1. Analyze road geometry & lane markings
        road_context = self.analyze_road_context(frame)

        # 2. Run orientation model on frame once if available
        ml_detections = []
        ml_model = self.get_orientation_model()
        if ml_model is not None and frame is not None and getattr(frame, 'size', 0) > 0:
            try:
                res = ml_model(frame, conf=0.20, verbose=False)
                if res and len(res) > 0 and res[0].boxes is not None:
                    names = res[0].names
                    for box in res[0].boxes:
                        cls_id = int(box.cls[0].item())
                        cname = names.get(cls_id, str(cls_id))
                        ml_detections.append({
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'class_name': cname,
                            'confidence': float(box.conf[0].item())
                        })
            except Exception as e:
                print(f"[WrongSideDetector] ML orientation inference notice: {e}")

        # 3. Estimate orientation for each motorcycle
        bike_assessments = []
        for bike in bike_detections:
            orient_info = self.estimate_motorcycle_orientation(
                frame, bike['bbox'], all_detections=all_detections,
                road_context=road_context, ml_detections=ml_detections
            )
            bike_assessments.append({
                'bike': bike,
                'orientation_info': orient_info
            })

        # 4. Multi-motorcycle scene consensus (if 2+ motorcycles have clear orientation)
        valid_orientations = [ba['orientation_info']['orientation'] for ba in bike_assessments 
                              if ba['orientation_info']['status'] == 'orientation_estimated']
        consensus_direction = None
        if len(valid_orientations) >= 2:
            counts = {}
            for o in valid_orientations:
                counts[o] = counts.get(o, 0) + 1
            most_common = max(counts.items(), key=lambda kv: kv[1])
            if most_common[1] >= 2 and most_common[1] > len(valid_orientations) * 0.5:
                consensus_direction = most_common[0]

        # 5. Evaluate each motorcycle against permitted direction & scene context
        violations = []
        permitted = self.permitted_direction
        target_angle = DIRECTION_CONFIG.get(permitted, {}).get('target_angle')

        for ba in bike_assessments:
            bike = ba['bike']
            oinfo = ba['orientation_info']
            orient = oinfo['orientation']
            orient_angle = oinfo['angle']
            orient_conf = oinfo['confidence']
            status = oinfo['status']

            is_potential_violation = False
            reason_msg = ""

            # Case: Auto / Unknown permitted direction
            if permitted in ['auto', 'unknown', None]:
                if consensus_direction and orient and orient != consensus_direction and status == 'orientation_estimated':
                    # Bike opposes the established consensus of other vehicles in the image
                    is_potential_violation = True
                    reason_msg = (
                        f"Motorcycle orientation ({oinfo['orientation_label']}) opposes scene traffic consensus "
                        f"({DIRECTION_CONFIG.get(consensus_direction, {}).get('label', consensus_direction)}); "
                        "movement velocity not verified in static image."
                    )
                elif oinfo.get('ml_class') == 'Wrong-Direction' and (oinfo.get('ml_confidence') or 0) >= 0.40:
                    is_potential_violation = True
                    reason_msg = (
                        f"Orientation classifier indicated contraflow orientation ({oinfo.get('ml_class')}, "
                        f"conf: {oinfo.get('ml_confidence'):.2f}); movement velocity not verified in static image."
                    )

            # Case: Specified permitted direction
            else:
                if status == 'orientation_estimated' and orient_angle is not None and target_angle is not None:
                    # Calculate angle difference
                    ang_diff = abs(orient_angle - target_angle)
                    if ang_diff > 180:
                        ang_diff = 360 - ang_diff

                    if ang_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE:
                        # Angle strongly opposes permitted direction
                        # Road context check
                        road_axis = road_context.get('road_axis', 'unknown')
                        is_longitudinal_match = (road_axis == 'longitudinal' and permitted in ['downward', 'upward'])
                        is_lateral_match = (road_axis == 'lateral' and permitted in ['leftward', 'rightward', 'right_to_left', 'left_to_right'])

                        if road_axis in ['unknown', 'ambiguous'] or is_longitudinal_match or is_lateral_match:
                            is_potential_violation = True
                            reason_msg = (
                                f"Estimated vehicle orientation ({oinfo['orientation_label']}) conflicts with permitted "
                                f"{self.get_direction_label()} traffic flow (angular deviation: {ang_diff:.0f}°); "
                                "movement velocity not verified in static frame."
                            )
                        elif orient_conf >= 0.60:
                            # Vehicle orientation features strongly oppose permitted flow, despite mixed background edges
                            is_potential_violation = True
                            reason_msg = (
                                f"Estimated vehicle orientation ({oinfo['orientation_label']}) conflicts with permitted "
                                f"{self.get_direction_label()} traffic flow (angular deviation: {ang_diff:.0f}°); "
                                "road line context mixed; movement velocity not verified in static frame."
                            )

                elif oinfo.get('ml_class') == 'Wrong-Direction' and (oinfo.get('ml_confidence') or 0) >= 0.40:
                    is_potential_violation = True
                    reason_msg = (
                        f"Orientation model detected contraflow ({oinfo.get('ml_class')}, "
                        f"conf: {oinfo.get('ml_confidence'):.2f}) against permitted {self.get_direction_label()}; "
                        "movement velocity not verified in static frame."
                    )

            if is_potential_violation:
                candidate = {
                    "violation_type": "wrong_side",
                    "title": "WRONG-SIDE RIDING — NEEDS REVIEW",
                    "detected": False,
                    "is_candidate": True,
                    "status": "potential_wrong_side",
                    "assessment_type": "image",
                    "confidence": None,
                    "detector_confidence": bike.get('confidence'),
                    "track_id": None,
                    "frame_number": None,
                    "evidence_frame": None,
                    "bounding_box": bike['bbox'],
                    "evidence_path": None,
                    "evidence_image": None,
                    "estimated_orientation": oinfo['orientation_label'],
                    "estimated_movement": f"Orientation: {oinfo['orientation_label']} (Velocity unverified)",
                    "permitted_direction": permitted,
                    "permitted_direction_label": self.get_direction_label(),
                    "reason": reason_msg,
                    "review_status": "insufficient_evidence",
                    "road_context": road_context
                }
                violations.append(candidate)

        # 6. Build summary
        summary = self.get_status_summary(
            is_video=False, image_detections=all_detections, wrong_side_violations=violations
        )
        return violations, summary

    def get_direction_label(self, direction=None):
        """Returns clean human-readable label for the permitted direction."""
        d = direction or self.permitted_direction
        return DIRECTION_CONFIG.get(d, DIRECTION_CONFIG['auto'])['label']

    def describe_angle(self, angle):
        """
        Translates an angle in image coordinate space (0=Right, 90=Down, 180=Left, 270=Up)
        into a human-readable direction label.
        """
        if angle is None:
            return "Unknown"
        a = angle % 360
        if 45 <= a < 135:
            return "Downward (Top to Bottom)"
        elif 135 <= a < 225:
            return "Leftward (Right to Left)"
        elif 225 <= a < 315:
            return "Upward (Bottom to Top)"
        else:
            return "Rightward (Left to Right)"

    def update_track(self, track_id, bbox, frame_idx, timestamp=0.0):
        """
        Updates the trajectory history and motion vector for a tracked vehicle.
        bbox: [x1, y1, x2, y2]
        """
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        if track_id not in self.trajectories:
            self.trajectories[track_id] = []
            
        self.trajectories[track_id].append({
            'center': (cx, cy),
            'bbox': bbox,
            'frame': frame_idx,
            'time': timestamp
        })
        
        # Keep recent 45 points (~1.5 to 2 seconds of history)
        if len(self.trajectories[track_id]) > 45:
            self.trajectories[track_id].pop(0)

        # Update motion vector if sufficient frames exist
        if len(self.trajectories[track_id]) >= self.min_frames:
            start_pt = self.trajectories[track_id][0]['center']
            end_pt = self.trajectories[track_id][-1]['center']
            dx = end_pt[0] - start_pt[0]
            dy = end_pt[1] - start_pt[1]
            disp = math.hypot(dx, dy)
            if disp >= self.min_displacement:
                angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                self.track_vectors[track_id] = {
                    'dx': dx,
                    'dy': dy,
                    'displacement': disp,
                    'angle': angle
                }

    def is_inside_roi(self, point, frame_shape):
        """
        Checks if a centroid (x, y) is inside the configured road Region of Interest (ROI).
        """
        if self.road_roi is None:
            return True
        pts = np.array(self.road_roi, dtype=np.int32)
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def calculate_motion_vector(self, track_id):
        """
        Computes cumulative displacement and motion vector over the track history.
        Returns: (dx, dy, total_displacement, angle_deg)
        """
        history = self.trajectories.get(track_id, [])
        if len(history) < self.min_frames:
            return None, None, 0.0, None

        start_pt = history[0]['center']
        end_pt = history[-1]['center']
        
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        displacement = math.hypot(dx, dy)
        
        angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        return dx, dy, displacement, angle

    def infer_dominant_flow(self):
        """
        Statistically infers the dominant road flow direction from consensus of all moving tracks.
        Requires at least 3 distinct moving vehicle tracks with significant displacement.
        """
        if len(self.track_vectors) < 3:
            self.flow_status = 'insufficient_motion_evidence'
            self.flow_message = "Wrong-side status could not be verified because reliable road-direction evidence was unavailable."
            return None

        # Compute average unit motion vector
        sum_ux = 0.0
        sum_uy = 0.0
        for vec in self.track_vectors.values():
            d = max(1e-4, vec['displacement'])
            sum_ux += vec['dx'] / d
            sum_uy += vec['dy'] / d

        n = len(self.track_vectors)
        avg_ux = sum_ux / n
        avg_uy = sum_uy / n
        coherence = math.hypot(avg_ux, avg_uy)
        self.flow_coherence = coherence

        # Coherence >= 0.60 indicates a clear dominant road flow direction
        if coherence >= 0.60:
            dom_angle = (math.degrees(math.atan2(avg_uy, avg_ux)) + 360) % 360
            self.flow_status = 'flow_established'
            self.flow_angle = dom_angle
            self.flow_message = f"Automatic road direction inferred: {self.describe_angle(dom_angle)} ({dom_angle:.0f}°, flow coherence: {coherence:.2f})"
            return dom_angle
        else:
            self.flow_status = 'direction_uncertain'
            self.flow_message = "Wrong-side status could not be verified because reliable road-direction evidence was unavailable."
            return None

    def evaluate_wrong_side(self, track_id, current_bbox, frame_shape, frame_idx=0):
        """
        Evaluates whether a tracked vehicle is moving in the wrong direction.
        In 'auto' mode, compares against dominant traffic flow consensus.
        In manual mode, compares against specified permitted direction.
        Returns:
            - Confirmed violation dict (detected=True, review_status='pending', confidence float)
            - Candidate dict with insufficient evidence (detected=False, review_status='insufficient_evidence', confidence=None)
            - None if compliant, stationary, or no contraflow candidate evidence exists.
        """
        history = self.trajectories.get(track_id, [])
        if len(history) < min(3, self.min_frames):
            return None
            
        cx = (current_bbox[0] + current_bbox[2]) / 2.0
        cy = (current_bbox[1] + current_bbox[3]) / 2.0
        if not self.is_inside_roi((cx, cy), frame_shape):
            return None
            
        dx, dy, displacement, angle = self.calculate_motion_vector(track_id)
        if displacement < 10.0:
            # Stationary or negligible jitter: do not flag
            return None

        is_wrong = False
        is_candidate = False
        motion_desc = ""
        candidate_reason = ""

        # --- Automatic Mode ---
        if self.permitted_direction in ['auto', 'unknown', None]:
            dom_flow = self.infer_dominant_flow()
            if dom_flow is not None:
                angle_diff = abs(angle - dom_flow)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff

                if angle_diff > 130.0 and displacement >= self.min_displacement:
                    is_wrong = True
                    motion_desc = f"Moving against dominant traffic flow (traffic: {dom_flow:.0f}°, vehicle: {angle:.0f}°, diff: {angle_diff:.1f}°)"
                elif (100.0 < angle_diff <= 130.0) or (angle_diff > 130.0 and displacement < self.min_displacement):
                    # Potential contraflow but borderline angle or displacement
                    is_candidate = True
                    candidate_reason = f"Candidate contraflow heading observed ({self.describe_angle(angle)}), but evidence is insufficient for automated confirmation (flow diff: {angle_diff:.1f}°, displacement: {displacement:.1f}px)."
            else:
                # Flow consensus not established: check if this vehicle opposes other tracked vehicles
                for other_id, other_vec in self.track_vectors.items():
                    if other_id != track_id and other_vec.get('displacement', 0) >= 15.0:
                        diff = abs(angle - other_vec['angle'])
                        if diff > 180: diff = 360 - diff
                        if diff > 135.0:
                            is_candidate = True
                            candidate_reason = f"Opposing vehicle trajectory observed against track #{other_id} ({diff:.0f}° divergence), but dominant road direction consensus is insufficient to confirm without officer review."
                            break

        # --- Manual Direction Modes ---
        elif self.permitted_direction == 'downward':
            # Downward: expected angle ~90° (+Y)
            angle_diff = abs(angle - 90.0)
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if dy < -10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement >= self.min_displacement:
                is_wrong = True
                motion_desc = f"Moving upward (North) against permitted Downward (South) traffic flow (angle diff: {angle_diff:.1f}°)"
            elif (dy < -6.0 and angle_diff > 75.0) or (dy < -10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement < self.min_displacement):
                is_candidate = True
                candidate_reason = f"Suspected contraflow movement upward (angle diff: {angle_diff:.1f}°), but displacement/trajectory evidence ({displacement:.1f}px) is insufficient for automated confirmation."

        elif self.permitted_direction == 'upward':
            # Upward: expected angle ~270° (-Y)
            angle_diff = abs(angle - 270.0)
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if dy > 10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement >= self.min_displacement:
                is_wrong = True
                motion_desc = f"Moving downward (South) against permitted Upward (North) traffic flow (angle diff: {angle_diff:.1f}°)"
            elif (dy > 6.0 and angle_diff > 75.0) or (dy > 10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement < self.min_displacement):
                is_candidate = True
                candidate_reason = f"Suspected contraflow movement downward (angle diff: {angle_diff:.1f}°), but displacement/trajectory evidence ({displacement:.1f}px) is insufficient for automated confirmation."

        elif self.permitted_direction in ['leftward', 'right_to_left']:
            # Leftward: expected angle ~180° (-X, Right to Left)
            angle_diff = abs(angle - 180.0)
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if dx > 10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement >= self.min_displacement:
                is_wrong = True
                motion_desc = f"Moving rightward (East) against permitted Leftward (West) traffic flow (angle diff: {angle_diff:.1f}°)"
            elif (dx > 6.0 and angle_diff > 75.0) or (dx > 10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement < self.min_displacement):
                is_candidate = True
                candidate_reason = f"Suspected contraflow movement rightward (angle diff: {angle_diff:.1f}°), but displacement/trajectory evidence ({displacement:.1f}px) is insufficient for automated confirmation."

        elif self.permitted_direction in ['rightward', 'left_to_right']:
            # Rightward: expected angle ~0° (+X, Left to Right)
            angle_diff = abs(angle - 0.0)
            if angle_diff > 180: angle_diff = 360 - angle_diff
            if dx < -10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement >= self.min_displacement:
                is_wrong = True
                motion_desc = f"Moving leftward (West) against permitted Rightward (East) traffic flow (angle diff: {angle_diff:.1f}°)"
            elif (dx < -6.0 and angle_diff > 75.0) or (dx < -10.0 and angle_diff > Config.WRONG_SIDE_ANGLE_TOLERANCE and displacement < self.min_displacement):
                is_candidate = True
                candidate_reason = f"Suspected contraflow movement leftward (angle diff: {angle_diff:.1f}°), but displacement/trajectory evidence ({displacement:.1f}px) is insufficient for automated confirmation."

        if is_wrong:
            conf = min(0.95, round(0.70 + min(0.25, (displacement / 200.0) * 0.25), 4))
            return {
                "violation_type": "wrong_side",
                "title": "WRONG-SIDE RIDING",
                "detected": True,
                "is_candidate": False,
                "confidence": conf,
                "bounding_box": current_bbox,
                "track_id": track_id,
                "frame_number": frame_idx,
                "evidence_frame": frame_idx,
                "assessment_type": "video",
                "displacement": round(displacement, 2),
                "angle": round(angle, 1),
                "estimated_movement": f"{self.describe_angle(angle)} ({angle:.0f}°)",
                "permitted_direction": self.permitted_direction,
                "permitted_direction_label": self.get_direction_label(),
                "motion_desc": motion_desc,
                "reason": motion_desc,
                "review_status": "pending",
                "status": "pending_review"
            }
        elif is_candidate:
            return {
                "violation_type": "wrong_side",
                "title": "WRONG-SIDE RIDING — NEEDS REVIEW",
                "detected": False,
                "is_candidate": True,
                "confidence": None,
                "bounding_box": current_bbox,
                "track_id": track_id,
                "frame_number": frame_idx,
                "evidence_frame": frame_idx,
                "assessment_type": "video",
                "displacement": round(displacement, 2),
                "angle": round(angle, 1),
                "estimated_movement": f"{self.describe_angle(angle)} ({angle:.0f}°)",
                "permitted_direction": self.permitted_direction,
                "permitted_direction_label": self.get_direction_label(),
                "motion_desc": candidate_reason,
                "reason": candidate_reason,
                "review_status": "insufficient_evidence",
                "status": "insufficient_evidence"
            }
            
        return None

    def get_status_summary(self, is_video=True, image_detections=None, wrong_side_violations=None):
        """
        Returns comprehensive road-direction assessment summary, including
        selected direction, estimated movement, explanation, and confidence info.
        For images, uses statuses:
          - 'potential_wrong_side'
          - 'direction_consistent'
          - 'direction_uncertain'
          - 'insufficient_evidence'
        """
        label = self.get_direction_label()
        
        if not is_video:
            # 1. If candidates were identified
            if wrong_side_violations and len(wrong_side_violations) > 0:
                target_angle = DIRECTION_CONFIG.get(self.permitted_direction, {}).get('target_angle')
                return {
                    "status": "potential_wrong_side",
                    "selected_direction": self.permitted_direction,
                    "selected_direction_label": label,
                    "estimated_movement": f"Potential direction conflict identified for {len(wrong_side_violations)} vehicle(s)",
                    "message": f"Potential wrong-side riding detected ({len(wrong_side_violations)} candidate(s)); velocity and trajectory require officer review.",
                    "explanation": "Orientation analysis indicates suspected contraflow travel; actual movement velocity cannot be verified from a single static frame.",
                    "confidence_info": "Visual orientation evidence (velocity unverified)",
                    "dominant_angle": target_angle,
                    "tracked_vehicles_count": 0,
                    "is_video": False
                }
            
            # 2. If image detections are provided
            if image_detections is not None:
                bike_count = len([d for d in image_detections if 'bike' in d.get('class_name', '').lower() or 'motorcycle' in d.get('class_name', '').lower()])
                if bike_count > 0:
                    if self.permitted_direction in ['auto', 'unknown', None]:
                        return {
                            "status": "direction_uncertain",
                            "selected_direction": "auto",
                            "selected_direction_label": label,
                            "estimated_movement": f"Observed {bike_count} vehicle(s) - road consensus unverified",
                            "message": "Wrong-side status could not be verified because reliable road-direction evidence was unavailable.",
                            "explanation": "Automatic road direction inference on a single static frame requires multi-frame temporal motion tracking or calibrated road geometry.",
                            "confidence_info": "Uncertain (0 temporal motion vectors)",
                            "dominant_angle": None,
                            "tracked_vehicles_count": 0,
                            "is_video": False
                        }
                    else:
                        target_angle = DIRECTION_CONFIG[self.permitted_direction]['target_angle']
                        return {
                            "status": "direction_consistent",
                            "selected_direction": self.permitted_direction,
                            "selected_direction_label": label,
                            "estimated_movement": f"Observed {bike_count} vehicle(s) - orientation consistent with {label}",
                            "message": f"Permitted direction specified as {label}. Observed vehicle(s) appear oriented consistently with traffic flow.",
                            "explanation": f"Motorcycles observed in image show no contraflow conflict against permitted {label}.",
                            "confidence_info": "Visual orientation consistent; velocity unverified",
                            "dominant_angle": target_angle,
                            "tracked_vehicles_count": 0,
                            "is_video": False
                        }
                elif len(image_detections) == 0:
                    return {
                        "status": "insufficient_evidence",
                        "selected_direction": self.permitted_direction,
                        "selected_direction_label": label,
                        "estimated_movement": "No vehicles observed",
                        "message": "Insufficient vehicle or road-direction evidence to assess traffic flow.",
                        "explanation": "No motorcycles detected to assess direction or orientation.",
                        "confidence_info": "No vehicle objects",
                        "dominant_angle": None,
                        "tracked_vehicles_count": 0,
                        "is_video": False
                    }

            # 3. Default when image movement/direction is unverified
            target_angle = DIRECTION_CONFIG.get(self.permitted_direction, {}).get('target_angle')
            return {
                "status": "direction_uncertain",
                "selected_direction": self.permitted_direction,
                "selected_direction_label": label,
                "estimated_movement": "Velocity/motion unverified in static frame",
                "message": "Wrong-side status could not be verified because reliable road-direction evidence was unavailable.",
                "explanation": "Automatic road direction inference on a single static frame requires multi-frame temporal motion tracking or calibrated road geometry.",
                "confidence_info": "Uncertain (0 temporal motion vectors)",
                "dominant_angle": target_angle,
                "tracked_vehicles_count": 0,
                "is_video": False
            }

        # For video:
        if self.permitted_direction in ['auto', 'unknown', None]:
            dom_flow = self.infer_dominant_flow()
            if dom_flow is not None:
                return {
                    "status": "flow_established",
                    "selected_direction": "auto",
                    "selected_direction_label": label,
                    "estimated_movement": f"Dominant consensus: {self.describe_angle(dom_flow)} ({dom_flow:.0f}°)",
                    "message": f"Automatic road direction inferred: {self.describe_angle(dom_flow)} ({dom_flow:.0f}°, flow coherence: {self.flow_coherence:.2f}).",
                    "explanation": f"Statistical consensus established across {len(self.track_vectors)} moving vehicles with {self.flow_coherence:.2f} vector coherence.",
                    "confidence_info": f"High (Coherence: {self.flow_coherence:.2f}, Tracks: {len(self.track_vectors)})",
                    "dominant_angle": round(dom_flow, 1),
                    "tracked_vehicles_count": len(self.track_vectors),
                    "is_video": True
                }
            else:
                return {
                    "status": "direction_uncertain",
                    "selected_direction": "auto",
                    "selected_direction_label": label,
                    "estimated_movement": f"Insufficient motion evidence ({len(self.track_vectors)} moving vehicle track(s))",
                    "message": "Wrong-side status could not be verified because reliable road-direction evidence was unavailable.",
                    "explanation": f"Observed {len(self.track_vectors)} moving vehicle track(s); requires at least 3 coherent tracks to infer road direction automatically.",
                    "confidence_info": f"Uncertain (Tracks: {len(self.track_vectors)}, Coherence: {self.flow_coherence:.2f})",
                    "dominant_angle": None,
                    "tracked_vehicles_count": len(self.track_vectors),
                    "is_video": True
                }
        else:
            target_angle = DIRECTION_CONFIG[self.permitted_direction]['target_angle']
            tracked_count = len(self.track_vectors)
            
            if tracked_count > 0:
                angles = [v['angle'] for v in self.track_vectors.values()]
                avg_angle = sum(angles) / len(angles)
                movement_desc = f"Observed {tracked_count} vehicle(s), average heading: {self.describe_angle(avg_angle)} ({avg_angle:.0f}°)"
            else:
                movement_desc = "No vehicle motion tracks detected yet"

            return {
                "status": "manual_direction_active",
                "selected_direction": self.permitted_direction,
                "selected_direction_label": label,
                "estimated_movement": movement_desc,
                "message": f"Permitted road direction active: {label}. Vehicle tracks are actively evaluated against heading {target_angle:.0f}°.",
                "explanation": f"Vehicle trajectories in image coordinate system are checked against expected heading {target_angle:.0f}° (coordinates: {DIRECTION_CONFIG[self.permitted_direction]['coord_meaning']}).",
                "confidence_info": f"User specified direction ({tracked_count} vehicles tracked)",
                "dominant_angle": target_angle,
                "tracked_vehicles_count": tracked_count,
                "is_video": True
            }

    def draw_trajectories_and_overlay(self, frame, violations):
        """
        Draws motion trails, track IDs, road ROI, and directional indicators on frame.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if self.road_roi:
            pts = np.array(self.road_roi, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 200, 0), thickness=2)

        violating_track_ids = {v['track_id'] for v in violations if v.get('track_id') is not None}
        
        for tid, points in self.trajectories.items():
            if len(points) < 2:
                continue
                
            is_violator = tid in violating_track_ids
            line_color = (0, 0, 255) if is_violator else (0, 220, 100)
            thickness = 3 if is_violator else 2
            
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1]['center'][0]), int(points[i - 1]['center'][1]))
                pt2 = (int(points[i]['center'][0]), int(points[i]['center'][1]))
                cv2.line(annotated, pt1, pt2, line_color, thickness, cv2.LINE_AA)
                
            latest = points[-1]['center']
            cv2.circle(annotated, (int(latest[0]), int(latest[1])), 4, line_color, -1)
            cv2.putText(annotated, f"ID:{tid}", (int(latest[0]) + 6, int(latest[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)

        return annotated
