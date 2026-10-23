import os, sys, cv2, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.detection_service import DetectionService
from services.helmet_detector import HelmetDetector
from services.triple_riding_detector import TripleRidingDetector
from services.violation_aggregator import ViolationAggregator

det_service = DetectionService()
h_detector = HelmetDetector()
t_detector = TripleRidingDetector()
aggregator = ViolationAggregator()

test_files = [
    'uploads/1673a036_Screenshot_2026-10-11_224343.png',
    'uploads/0398de43_Screenshot_2026-10-12_182720.png',
    'uploads/24361fe8_Screenshot_2026-10-12_183619.png',
    'uploads/7f523c5a_Screenshot_2026-10-11_224013.png',
    'data/samples/1010_jpg.rf.7cb05db3a94c3a2c5bf6874ff560e79f.jpg',
    'data/samples/1031_jpg.rf.e6a9f0a0d8e91d9ea2b938fe7f3baff9.jpg',
    'data/samples/1033_jpg.rf.f8760a72873985e9faf59e01a854a850.jpg'
]

for tf in test_files:
    if not os.path.exists(tf):
        continue
    frame = cv2.imread(tf)
    if frame is None:
        continue
    print(f"=== Testing {os.path.basename(tf)} (shape: {frame.shape}) ===")
    dets = det_service.detect(frame)
    print(f"Detections ({len(dets)}):")
    for d in dets:
        print(f"  {d['class_name']}: conf={d['confidence']}, bbox={d['bbox']}")
    
    h_viols = h_detector.analyze(frame, dets)
    t_viols = t_detector.analyze(frame, dets)
    agg_viols = aggregator.aggregate_frame(frame, h_viols, t_viols, [], [])
    print(f"Violations: Helmet={len(h_viols)}, Triple={len(t_viols)}, TotalAgg={len(agg_viols)}")
    for v in agg_viols:
        print(f"  -> {v['violation_type']}: conf={v.get('confidence')}, status={v.get('review_status')}, reason={v.get('reason')}")
    print()
