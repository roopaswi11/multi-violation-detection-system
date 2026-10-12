"""
Violation Logic Module.
Provides standalone geometric, spatial, and rule-based decision logic
for no-helmet detection, triple-riding rider counting, and trajectory directional analysis.
"""
import math
import numpy as np

def check_no_helmet_rule(rider_box, head_box, direct_no_helmet_boxes, direct_helmet_boxes, vit_prediction=None):
    """
    Evaluates helmet compliance for a detected rider.
    Returns: (is_violation: bool, confidence: float, rule_applied: str)
    """
    # 1. Direct no-helmet detection
    for nh_box in direct_no_helmet_boxes:
        if compute_box_iou(head_box, nh_box['bbox']) > 0.15:
            return True, nh_box['confidence'], "Direct no-helmet detected on rider head"

    # 2. Direct helmet detection
    for h_box in direct_helmet_boxes:
        if compute_box_iou(head_box, h_box['bbox']) > 0.15:
            return False, h_box['confidence'], "Direct helmet confirmed"

    # 3. ViT classifier prediction
    if vit_prediction:
        status, conf = vit_prediction
        if status == 'no-helmet' and conf > 0.70:
            return True, conf, f"ViT classifier confirmed no-helmet ({conf:.2f})"
        elif status == 'helmet':
            return False, conf, "ViT classifier confirmed helmet"

    # 4. Absence heuristic (marked for review, not definitive proof)
    head_w = head_box[2] - head_box[0]
    head_h = head_box[3] - head_box[1]
    if head_w >= 25 and head_h >= 25:
        return True, 0.55, "No helmet detected on visible rider (pending review)"

    return False, 0.0, "Indeterminate head region / low resolution"

def check_triple_riding_rule(rider_count, max_permitted=2):
    """
    Evaluates triple riding condition:
    Rider count <= 2: No violation
    Rider count > 2: Violation
    """
    if rider_count > max_permitted:
        return True, f"Excess riders detected: {rider_count} riders (limit: {max_permitted})"
    return False, f"Normal rider count: {rider_count}"

def check_wrong_side_rule(trajectory_points, permitted_direction='downward', min_disp=35.0):
    """
    Evaluates trajectory displacement and travel angle against permitted direction.
    """
    if len(trajectory_points) < 5:
        return False, 0.0, "Insufficient trajectory points"

    start_x, start_y = trajectory_points[0]
    end_x, end_y = trajectory_points[-1]
    dx = end_x - start_x
    dy = end_y - start_y
    displacement = math.hypot(dx, dy)

    if displacement < min_disp:
        return False, 0.0, "Vehicle stationary or negligible movement"

    angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    
    is_wrong = False
    if permitted_direction == 'downward' and dy < -10.0:
        is_wrong = True
    elif permitted_direction == 'upward' and dy > 10.0:
        is_wrong = True
    elif permitted_direction == 'left_to_right' and dx < -10.0:
        is_wrong = True
    elif permitted_direction == 'right_to_left' and dx > 10.0:
        is_wrong = True

    if is_wrong:
        conf = min(0.95, 0.70 + (displacement / 200.0) * 0.25)
        return True, conf, f"Vehicle trajectory moving against {permitted_direction} direction (angle {angle:.1f}°)"

    return False, 0.0, "Vehicle conforms to permitted travel direction"

def compute_box_iou(boxA, boxB):
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
    return interArea / denom if denom > 0 else 0.0
