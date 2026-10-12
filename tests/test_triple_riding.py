import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.triple_riding_detector import TripleRidingDetector

class TestTripleRidingDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TripleRidingDetector(max_permitted_riders=2)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_single_and_double_riders_no_violation(self):
        # 1 Motorcycle with 2 riders (allowed)
        detections = [
            {'bbox': [100, 200, 350, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            {'bbox': [120, 100, 200, 320], 'class_name': 'person', 'confidence': 0.85},
            {'bbox': [210, 110, 290, 320], 'class_name': 'person', 'confidence': 0.87}
        ]
        violations = self.detector.analyze(self.frame, detections, is_video=False)
        self.assertEqual(len(violations), 0)

    def test_three_riders_violation(self):
        # 1 Motorcycle with 3 mounted riders (> 2, violation)
        detections = [
            {'bbox': [100, 200, 400, 450], 'class_name': 'motorcycle', 'confidence': 0.92},
            {'bbox': [120, 100, 190, 320], 'class_name': 'person', 'confidence': 0.88},
            {'bbox': [195, 105, 265, 320], 'class_name': 'person', 'confidence': 0.86},
            {'bbox': [270, 110, 340, 320], 'class_name': 'person', 'confidence': 0.89}
        ]
        violations = self.detector.analyze(self.frame, detections, is_video=False)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['violation_type'], 'triple_riding')
        self.assertEqual(violations[0]['rider_count'], 3)
        self.assertGreater(violations[0]['confidence'], 0.70)

    def test_pedestrian_not_counted(self):
        # 1 Motorcycle with 2 riders + 1 pedestrian far away
        detections = [
            {'bbox': [100, 200, 350, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            {'bbox': [120, 100, 200, 320], 'class_name': 'person', 'confidence': 0.85},
            {'bbox': [210, 110, 290, 320], 'class_name': 'person', 'confidence': 0.87},
            {'bbox': [500, 200, 560, 420], 'class_name': 'person', 'confidence': 0.90} # Pedestrian on pavement
        ]
        violations = self.detector.analyze(self.frame, detections, is_video=False)
        self.assertEqual(len(violations), 0)

if __name__ == '__main__':
    unittest.main()
