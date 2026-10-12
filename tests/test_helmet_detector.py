import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.helmet_detector import HelmetDetector

class TestHelmetDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HelmetDetector(vit_enabled=False)
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_rider_on_bike_spatial_association(self):
        # Motorcycle box: [x1, y1, x2, y2]
        bike_box = [100, 200, 300, 450]
        # Mounted rider box: overlaps bike top/body
        mounted_rider = [150, 100, 250, 350]
        # Distant pedestrian: clearly off to the side
        pedestrian = [450, 100, 520, 350]

        self.assertTrue(self.detector.is_rider_on_bike(mounted_rider, bike_box))
        self.assertFalse(self.detector.is_rider_on_bike(pedestrian, bike_box))

    def test_direct_no_helmet_detection(self):
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90, 'class_id': 3},
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88, 'class_id': 0},
            {'bbox': [170, 100, 230, 160], 'class_name': 'no-helmet', 'confidence': 0.92, 'class_id': 2}
        ]
        violations = self.detector.analyze(self.dummy_frame, detections)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['violation_type'], 'no_helmet')
        self.assertGreaterEqual(violations[0]['confidence'], 0.85)

    def test_direct_helmet_no_violation(self):
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90, 'class_id': 3},
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88, 'class_id': 0},
            {'bbox': [170, 100, 230, 160], 'class_name': 'helmet', 'confidence': 0.94, 'class_id': 1}
        ]
        violations = self.detector.analyze(self.dummy_frame, detections)
        self.assertEqual(len(violations), 0)

if __name__ == '__main__':
    unittest.main()
