import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from services.detection_service import DetectionService

def get_association_score(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    pcx = (px1 + px2) / 2.0
    
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    bcx = (bx1 + bx2) / 2.0
    
    # 1. Horizontal span: rider center must be within bike bounds (+15% margin)
    h_margin = bw * 0.15
    if not (bx1 - h_margin <= pcx <= bx2 + h_margin):
        return 0.0
        
    inter_x = max(0.0, min(px2, bx2) - max(px1, bx1))
    if inter_x / pw < 0.25:
        return 0.0

    # 2. Foot/Base Proximity:
    # A rider's lower body (py2) must reach down into the lower half of the bike
    # specifically near the footpegs/seat: by2 - py2 must not be excessively far above the bike base
    foot_offset_from_base = (by2 - py2) / bh
    
    # If the person's feet are more than 48% of the bike height ABOVE the bike bottom,
    # they are not mounted unless they are an occluded tandem rider matching an already mounted rider.
    if foot_offset_from_base > 0.48:
        return 0.0
        
    # Rider top must extend above or near top of bike
    if py1 > by1 + 0.45 * bh:
        return 0.0
        
    # Rider bottom cannot extend far below bike
    if py2 > by2 + 0.20 * bh:
        return 0.0
        
    # 3. Relative scale sanity
    if ph < 0.35 * bh or ph > 2.5 * bh:
        return 0.0
        
    inter_y = max(0.0, min(py2, by2) - max(py1, by1))
    overlap = (inter_x * inter_y) / (pw * ph)
    
    # Penalize large horizontal or foot distance
    h_dist = abs(pcx - bcx) / bw
    foot_dist = max(0.0, foot_offset_from_base)
    
    score = overlap * (1.0 - 0.35 * min(1.0, h_dist)) * (1.0 - 0.4 * min(1.0, foot_dist))
    return max(0.01, score)

def associate_riders_to_motorcycles(bikes, persons):
    """
    Associates each person with the most suitable motorcycle.
    Handles:
    - Normal riders
    - Tandem riders (including occluded front/middle riders)
    - Distinguishes bystanders / pedestrians / background riders on separate bikes
    """
    if not bikes or not persons:
        return {b_idx: [] for b_idx in range(len(bikes))}
        
    bike_riders = {b_idx: [] for b_idx in range(len(bikes))}
    unassigned = []
    
    # Pass 1: Direct association for clearly mounted riders
    for p in persons:
        best_b = None
        best_s = 0.0
        for b_idx, b in enumerate(bikes):
            s = get_association_score(p['bbox'], b['bbox'])
            if s > best_s:
                best_s = s
                best_b = b_idx
        if best_b is not None and best_s > 0.05:
            bike_riders[best_b].append(p)
        else:
            unassigned.append(p)
            
    # Pass 2: Tandem rider recovery for partially occluded riders on motorcycles
    # If a motorcycle already has at least one confirmed mounted rider,
    # check unassigned persons whose heads are adjacent and horizontally aligned along the seat
    for b_idx, mounted in bike_riders.items():
        if not mounted:
            continue
        b_box = bikes[b_idx]['bbox']
        bh = b_box[3] - b_box[1]
        
        still_unassigned = []
        for p in unassigned:
            pb = p['bbox']
            pw = pb[2] - pb[0]
            ph = pb[3] - pb[1]
            pcx = (pb[0] + pb[2]) / 2.0
            
            # Check if this person is horizontally inside the bike
            if not (b_box[0] - 0.10 * (b_box[2]-b_box[0]) <= pcx <= b_box[2] + 0.10 * (b_box[2]-b_box[0])):
                still_unassigned.append(p)
                continue
                
            # Check if person's head aligns with any already mounted rider on this bike
            is_tandem = False
            for m in mounted:
                mb = m['bbox']
                mph = mb[3] - mb[1]
                mpw = mb[2] - mb[0]
                
                # Tandem check:
                # 1. Heads aligned vertically within 35px or 25% of height
                head_y_diff = abs(pb[1] - mb[1])
                # 2. Horizontally spaced along seat (between 0.35 and 1.8 person widths)
                head_x_dist = abs(pcx - (mb[0] + mb[2]) / 2.0)
                # 3. Relative scale ratio: a tandem rider cannot be less than 55% the size of the mounted rider
                scale_ratio = min(ph, mph) / max(ph, mph)
                # 4. Vertical depth alignment: even if lower legs are occluded, bottom cannot be > 40% different
                bottom_diff = abs(pb[3] - mb[3]) / max(ph, mph)
                
                if (head_y_diff <= max(40.0, 0.25 * max(ph, mph)) and 
                    head_x_dist <= 1.8 * max(pw, mpw) and 
                    scale_ratio >= 0.55 and 
                    bottom_diff <= 0.40):
                    # Must also enter the bike top
                    if pb[3] > b_box[1] + 0.20 * bh:
                        is_tandem = True
                        break
                        
            if is_tandem:
                mounted.append(p)
                print(f"  [Pass 2 Recovery] Recovered occluded tandem rider for Bike {b_idx}: conf={p['confidence']:.2f}, bbox={pb}")
            else:
                still_unassigned.append(p)
        unassigned = still_unassigned
        
    return bike_riders

det_service = DetectionService()

for test_name, img_path in [
    ("NON-VIOLATING SCOOTER (65ab4a80)", r'uploads/65ab4a80_Screenshot_2026-10-22_204709.png'),
    ("GENUINE TRIPLE RIDING (24361fe8)", r'uploads/24361fe8_Screenshot_2026-10-12_183619.png')
]:
    print(f"\n=================== {test_name} ===================")
    frame = cv2.imread(img_path)
    dets = det_service.detect(frame)
    bikes = [d for d in dets if d['class_name'] in ['motorcycle', 'bike']]
    persons = [d for d in dets if d['class_name'] in ['person', 'rider']]
    
    print(f"Bikes: {len(bikes)}, Persons: {len(persons)}")
    assigned = associate_riders_to_motorcycles(bikes, persons)
    
    for b_idx, rlist in assigned.items():
        print(f"  Bike {b_idx} bbox={[round(x,1) for x in bikes[b_idx]['bbox']]}: {len(rlist)} rider(s)")
        for r in rlist:
            print(f"    -> Rider conf={r['confidence']:.2f}, bbox={[round(x,1) for x in r['bbox']]}")
