import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from services.detection_service import DetectionService

class TestDetectionService(unittest.TestCase):
    def setUp(self):
        self.detector = DetectionService(conf_threshold=0.25)

    def test_model_loading(self):
        self.assertIsNotNone(self.detector.model)
        self.assertTrue(len(self.detector.class_names) > 0)

    def test_detection_on_dummy_frame(self):
        # Create a synthetic 640x640 frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        # Add a simple white rectangle
        frame[200:400, 200:400] = 255
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list)

    def test_detection_on_sample_image(self):
        sample_dir = Config.SAMPLES_DIR
        samples = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png'))]
        if samples:
            test_img = samples[0]
            detections = self.detector.detect(test_img)
            self.assertIsInstance(detections, list)
            for d in detections:
                self.assertIn('bbox', d)
                self.assertEqual(len(d['bbox']), 4)
                self.assertIn('confidence', d)
                self.assertIn('class_name', d)

if __name__ == '__main__':
    unittest.main()
