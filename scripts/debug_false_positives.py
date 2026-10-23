import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import json
from services.detection_service import DetectionService
from services.helmet_detector import HelmetDetector
from services.triple_riding_detector import TripleRidingDetector
from services.violation_aggregator import ViolationAggregator
from services.video_processor import VideoProcessor

img_path = r'uploads/65ab4a80_Screenshot_2026-10-22_204709.png'
frame = cv2.imread(img_path)
print('Image shape:', frame.shape)

det_service = DetectionService()
raw_dets = det_service.detect(frame)
print(f'Total raw detections: {len(raw_dets)}')
for idx, d in enumerate(raw_dets):
    cname = d.get('class_name')
    conf = d.get('confidence', 0.0)
    bbox = [round(x, 1) for x in d.get('bbox', [])]
    mtype = d.get('model_type', '')
    print(f'  [{idx+1}] {cname} ({conf:.3f}) bbox={bbox} model={mtype}')

vp = VideoProcessor(permitted_direction="two_way")
output_path, viols, dets = vp.process_image(img_path)
print(f'\n--- VideoProcessor Aggregated Output ---')
print(f'Total aggregated violations: {len(viols)}')
for idx, v in enumerate(viols):
    print(f'  [{idx+1}] Type={v.get("violation_type")}, Rider={v.get("rider_id")}, Conf={v.get("confidence")}, Status={v.get("review_status")}, bbox={v.get("bounding_box")}')
