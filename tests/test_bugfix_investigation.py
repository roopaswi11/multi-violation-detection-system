import unittest
import os
import sys
import io
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from config import Config
from services.detection_service import DetectionService
from services.triple_riding_detector import TripleRidingDetector
from services.helmet_detector import HelmetDetector
from services.violation_aggregator import ViolationAggregator
from services.database_service import DatabaseService

class TestBugfixInvestigation(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.db = DatabaseService()
        self.detector = DetectionService(conf_threshold=0.25)
        self.triple_detector = TripleRidingDetector()

    # 1. Image upload test
    def test_01_image_upload_endpoint(self):
        # Create a small 100x100 dummy JPEG in memory
        import cv2
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', dummy)
        data = {
            'file': (io.BytesIO(buf.tobytes()), 'test_upload.jpg'),
            'permitted_direction': 'downward',
            'conf_threshold': '0.25'
        }
        res = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Detection Results', res.data)

    # 2. YOLO11 inference test
    def test_02_yolo11_inference(self):
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        dets = self.detector.detect(dummy)
        self.assertIsInstance(dets, list)

    # 3. Class-name mapping verification
    def test_03_class_name_mapping(self):
        self.assertIsNotNone(self.detector.class_names)
        self.assertTrue(len(self.detector.class_names) > 0)
        # Verify base model vehicle/rider classes are loaded
        if self.detector.base_model is not None:
            names = [str(n).lower() for n in self.detector.base_model.names.values()]
            self.assertIn('person', names)
            self.assertIn('motorcycle', names)

    # 4. Multiple-person detection
    def test_04_multiple_person_detection_deduplication(self):
        # Simulated riders on a motorcycle: 2 distinct riders + 1 duplicate jitter box
        bike_box = [100.0, 150.0, 400.0, 350.0]
        raw_riders = [
            {'bbox': [110.0, 80.0, 190.0, 280.0], 'confidence': 0.85, 'class_name': 'person'},
            {'bbox': [112.0, 82.0, 188.0, 278.0], 'confidence': 0.40, 'class_name': 'person'}, # Duplicate box on rider 1
            {'bbox': [220.0, 90.0, 300.0, 290.0], 'confidence': 0.75, 'class_name': 'person'}, # Distinct rider 2
            {'bbox': [310.0, 95.0, 380.0, 295.0], 'confidence': 0.70, 'class_name': 'person'}, # Distinct rider 3
        ]
        distinct = TripleRidingDetector.deduplicate_overlapping_riders(raw_riders, iou_thresh=0.40)
        self.assertEqual(len(distinct), 3)

    # 5. Motorcycle detection & classification
    def test_05_motorcycle_detection_geometry(self):
        bike_box = [50.0, 100.0, 300.0, 300.0]
        person_mounted = [80.0, 50.0, 160.0, 250.0]
        person_pedestrian = [500.0, 50.0, 580.0, 250.0]
        
        self.assertTrue(TripleRidingDetector.is_rider_on_bike(person_mounted, bike_box))
        self.assertFalse(TripleRidingDetector.is_rider_on_bike(person_pedestrian, bike_box))

    # 6. Rider-to-motorcycle association
    def test_06_rider_to_motorcycle_association(self):
        dets = [
            {'bbox': [50.0, 100.0, 350.0, 300.0], 'class_name': 'motorcycle', 'confidence': 0.80},
            {'bbox': [60.0, 40.0, 140.0, 240.0], 'class_name': 'person', 'confidence': 0.75},
            {'bbox': [150.0, 40.0, 230.0, 240.0], 'class_name': 'person', 'confidence': 0.72},
            {'bbox': [240.0, 40.0, 320.0, 240.0], 'class_name': 'person', 'confidence': 0.70},
            {'bbox': [600.0, 40.0, 680.0, 240.0], 'class_name': 'person', 'confidence': 0.85}, # Pedestrian far away
        ]
        violations = self.triple_detector.analyze(None, dets, is_video=False)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['rider_count'], 3)

    # 7. Triple-riding violation logic (legal vs illegal)
    def test_07_triple_riding_legal_vs_illegal(self):
        # 2 riders = legal
        legal_dets = [
            {'bbox': [50.0, 100.0, 300.0, 300.0], 'class_name': 'motorcycle', 'confidence': 0.85},
            {'bbox': [70.0, 40.0, 150.0, 240.0], 'class_name': 'person', 'confidence': 0.80},
            {'bbox': [180.0, 40.0, 260.0, 240.0], 'class_name': 'person', 'confidence': 0.78},
        ]
        v_legal = self.triple_detector.analyze(None, legal_dets, is_video=False)
        self.assertEqual(len(v_legal), 0)

        # 3 riders = illegal
        illegal_dets = legal_dets + [
            {'bbox': [120.0, 35.0, 200.0, 235.0], 'class_name': 'person', 'confidence': 0.70}
        ]
        v_illegal = self.triple_detector.analyze(None, illegal_dets, is_video=False)
        self.assertEqual(len(v_illegal), 1)
        self.assertTrue(v_illegal[0]['detected'])

    # 8. Empty detection results
    def test_08_empty_detection_results(self):
        empty_dets = []
        v_triple = self.triple_detector.analyze(None, empty_dets, is_video=False)
        h_detector = HelmetDetector()
        v_helmet = h_detector.analyze(None, empty_dets)
        self.assertEqual(len(v_triple), 0)
        self.assertEqual(len(v_helmet), 0)

    # 9. Model loading fallback
    def test_09_model_loading_fallback(self):
        # Instantiating DetectionService with nonexistent path should gracefully fallback to default
        ds = DetectionService(model_path='nonexistent_path_12345.pt')
        self.assertIsNotNone(ds.model)

    # 10. Dashboard rendering: distinguishes compliant vs empty vs violations
    def test_10_dashboard_rendering(self):
        import uuid
        job_id = f"test_{uuid.uuid4().hex[:6]}"
        
        # Test empty objects job
        self.db.create_job(job_id, 'image', 'empty.jpg', f'{job_id}.jpg')
        self.db.complete_job(job_id, None, 0, status='completed', metadata={
            'total_detections': 0,
            'diagnostic_message': 'No traffic objects detected at current threshold.',
            'conf_threshold': 0.25
        })
        res = self.client.get(f'/results/{job_id}')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'No Traffic Objects Detected', res.data)

        # Test compliant traffic job
        job_id2 = f"test_{uuid.uuid4().hex[:6]}"
        self.db.create_job(job_id2, 'image', 'compliant.jpg', f'{job_id2}.jpg')
        self.db.complete_job(job_id2, None, 0, status='completed', metadata={
            'total_detections': 3,
            'diagnostic_message': 'Observed 3 traffic objects. All verified compliant.',
            'conf_threshold': 0.25
        })
        res2 = self.client.get(f'/results/{job_id2}')
        self.assertEqual(res2.status_code, 200)
        self.assertIn(b'No Traffic Violations Detected', res2.data)

if __name__ == '__main__':
    unittest.main()
