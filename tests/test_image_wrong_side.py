import unittest
import os
import sys
import uuid
import numpy as np
import cv2
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from services.wrong_side_detector import WrongSideDetector
from services.violation_aggregator import ViolationAggregator
from services.database_service import DatabaseService
from services.video_processor import VideoProcessor
from config import Config

class TestImageAndVideoWrongSide(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.db = DatabaseService()
        self.frame_shape = (720, 1280, 3)
        
        # Create a synthetic road frame with longitudinal lane lines
        self.road_frame = np.ones(self.frame_shape, dtype=np.uint8) * 80  # Dark asphalt
        # Draw two vertical white lane lines in lower 60%
        cv2.line(self.road_frame, (400, 300), (400, 700), (255, 255, 255), 4)
        cv2.line(self.road_frame, (880, 300), (880, 700), (255, 255, 255), 4)

    # 1. Image containing a motorcycle aligned with the permitted direction
    def test_1_image_motorcycle_aligned_with_permitted_direction(self):
        det = WrongSideDetector(permitted_direction='downward')
        # Frontal motorcycle facing downward (headlight bright at bottom, no rear plate)
        frame = self.road_frame.copy()
        # Bike bbox [x1, y1, x2, y2]
        bike_bbox = [500, 350, 580, 520] # ar ~ 0.47 (longitudinal)
        # Draw bright frontal headlight in lower half of motorcycle
        cv2.circle(frame, (540, 480), 12, (255, 255, 255), -1)

        bikes = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.88}]
        all_detections = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.88}]

        violations, summary = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        # Should be compliant / aligned: no violation flagged
        self.assertEqual(len(violations), 0)
        self.assertEqual(summary['status'], 'direction_consistent')
        self.assertIn('consistent', summary['message'].lower())

    # 2. Image containing a motorcycle whose estimated orientation conflicts with permitted direction
    def test_2_image_motorcycle_conflicts_with_permitted_direction(self):
        det = WrongSideDetector(permitted_direction='downward')
        frame = self.road_frame.copy()
        bike_bbox = [500, 350, 580, 520] # ar ~ 0.47 (longitudinal)
        # Rear plate and red taillight detected (facing upward away from camera)
        plate_bbox = [525, 480, 555, 505]
        all_detections = [
            {'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85},
            {'bbox': plate_bbox, 'class_name': 'number_plate', 'confidence': 0.90}
        ]
        # Draw red taillight in lower half
        cv2.circle(frame, (540, 470), 10, (0, 0, 255), -1)

        bikes = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85}]

        violations, summary = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        # Should generate a potential wrong-side candidate
        self.assertEqual(len(violations), 1)
        v = violations[0]
        self.assertEqual(v['violation_type'], 'wrong_side')
        self.assertEqual(v['title'], 'WRONG-SIDE RIDING — NEEDS REVIEW')
        self.assertFalse(v['detected'])
        self.assertTrue(v['is_candidate'])
        self.assertIsNone(v['confidence'])
        self.assertEqual(v['status'], 'potential_wrong_side')
        self.assertEqual(v['assessment_type'], 'image')
        self.assertIn('Upward', v['estimated_orientation'])
        self.assertEqual(v['permitted_direction'], 'downward')
        self.assertEqual(summary['status'], 'potential_wrong_side')

    # 3. Image with ambiguous motorcycle orientation
    def test_3_image_ambiguous_motorcycle_orientation(self):
        det = WrongSideDetector(permitted_direction='downward')
        # Blank uniform crop, square aspect ratio, no rider or plate
        frame = np.ones(self.frame_shape, dtype=np.uint8) * 128
        bike_bbox = [500, 350, 600, 450] # ar ~ 1.0 (ambiguous)
        bikes = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.75}]
        all_detections = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.75}]

        violations, summary = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        # Must not fabricate violation merely because orientation is difficult to estimate
        self.assertEqual(len(violations), 0)
        self.assertEqual(summary['status'], 'direction_consistent') # Permitted downward, but no conflict proved

    # 4. Image with no visible road context
    def test_4_image_no_visible_road_context(self):
        det = WrongSideDetector(permitted_direction='downward')
        # Pure flat background: no edges, no lane lines
        blank_frame = np.ones(self.frame_shape, dtype=np.uint8) * 50
        road_ctx = det.analyze_road_context(blank_frame)
        self.assertEqual(road_ctx['road_axis'], 'unknown')
        self.assertEqual(road_ctx['lines_count'], 0)

        bikes = [{'bbox': [200, 200, 300, 320], 'class_name': 'motorcycle', 'confidence': 0.80}]
        violations, summary = det.evaluate_image_wrong_side(blank_frame, bikes, bikes)
        # Without contraflow orientation evidence, no violation should be flagged
        self.assertEqual(len(violations), 0)

    # 5. Image with multiple motorcycles (scene consensus)
    def test_5_image_multiple_motorcycles_scene_consensus(self):
        # Permitted: auto
        det = WrongSideDetector(permitted_direction='auto')
        frame = self.road_frame.copy()

        # Bike 1 & Bike 2 facing downward (headlights)
        bike1_bbox = [250, 350, 330, 520]
        bike2_bbox = [450, 350, 530, 520]
        cv2.circle(frame, (290, 480), 10, (255, 255, 255), -1)
        cv2.circle(frame, (490, 480), 10, (255, 255, 255), -1)

        # Bike 3 facing upward (rear plate + taillight)
        bike3_bbox = [700, 350, 780, 520]
        plate3_bbox = [725, 480, 755, 505]
        cv2.circle(frame, (740, 470), 10, (0, 0, 255), -1)

        bikes = [
            {'bbox': bike1_bbox, 'class_name': 'motorcycle', 'confidence': 0.85},
            {'bbox': bike2_bbox, 'class_name': 'motorcycle', 'confidence': 0.87},
            {'bbox': bike3_bbox, 'class_name': 'motorcycle', 'confidence': 0.82}
        ]
        all_detections = bikes + [{'bbox': plate3_bbox, 'class_name': 'number_plate', 'confidence': 0.90}]

        violations, summary = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        # Opposing Bike 3 should be flagged based on scene consensus opposition
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]['bounding_box'], bike3_bbox)
        self.assertEqual(violations[0]['title'], 'WRONG-SIDE RIDING — NEEDS REVIEW')
        self.assertFalse(violations[0]['detected'])

    # 6. Motorcycle detection succeeds but orientation estimation fails / ambiguous
    def test_6_detection_succeeds_orientation_fails(self):
        det = WrongSideDetector(permitted_direction='auto')
        blank_frame = np.ones(self.frame_shape, dtype=np.uint8) * 100
        # Highly ambiguous crop
        bikes = [{'bbox': [300, 300, 400, 400], 'class_name': 'motorcycle', 'confidence': 0.92}]
        violations, summary = det.evaluate_image_wrong_side(blank_frame, bikes, bikes)
        self.assertEqual(len(violations), 0)
        self.assertEqual(summary['status'], 'direction_uncertain')
        self.assertIn("reliable road-direction evidence was unavailable", summary['message'])

    # 7. Video with a reliable wrong-side trajectory
    def test_7_video_reliable_wrong_side_trajectory(self):
        det = WrongSideDetector(permitted_direction='downward', min_displacement=20.0, min_frames=5)
        # Upward moving vehicle trajectory against permitted downward
        for i in range(10):
            det.update_track(99, [200, 500 - i * 25, 300, 600 - i * 25], frame_idx=i)
        
        viol = det.evaluate_wrong_side(99, [200, 275, 300, 375], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol)
        self.assertEqual(viol['violation_type'], 'wrong_side')
        self.assertEqual(viol['title'], 'WRONG-SIDE RIDING')
        self.assertTrue(viol['detected'])
        self.assertFalse(viol['is_candidate'])
        self.assertEqual(viol['assessment_type'], 'video')
        self.assertEqual(viol['track_id'], 99)
        self.assertEqual(viol['frame_number'], 9)
        self.assertIsNotNone(viol['confidence'])
        self.assertGreater(viol['confidence'], 0.70)

    # 8. Video with insufficient trajectory evidence
    def test_8_video_insufficient_trajectory_evidence(self):
        det = WrongSideDetector(permitted_direction='downward', min_displacement=30.0, min_frames=5)
        # Small movement (only 12px displacement, below min_displacement=30.0)
        for i in range(5):
            det.update_track(88, [200, 400 - i * 3, 300, 500 - i * 3], frame_idx=i)
        
        candidate = det.evaluate_wrong_side(88, [200, 388, 300, 488], self.frame_shape, frame_idx=4)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate['violation_type'], 'wrong_side')
        self.assertEqual(candidate['title'], 'WRONG-SIDE RIDING — NEEDS REVIEW')
        self.assertFalse(candidate['detected'])
        self.assertTrue(candidate['is_candidate'])
        self.assertIsNone(candidate['confidence'])
        self.assertEqual(candidate['review_status'], 'insufficient_evidence')

    # 9. Image result without track_id or evidence_frame
    def test_9_image_result_without_track_id_or_evidence_frame(self):
        det = WrongSideDetector(permitted_direction='downward')
        frame = self.road_frame.copy()
        bike_bbox = [500, 350, 580, 520]
        plate_bbox = [525, 480, 555, 505]
        all_detections = [
            {'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85},
            {'bbox': plate_bbox, 'class_name': 'number_plate', 'confidence': 0.90}
        ]
        cv2.circle(frame, (540, 470), 10, (0, 0, 255), -1)
        bikes = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85}]

        violations, _ = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        self.assertEqual(len(violations), 1)
        v = violations[0]
        self.assertIsNone(v['track_id'])
        self.assertIsNone(v['evidence_frame'])
        self.assertIsNone(v['frame_number'])
        self.assertIsNone(v['confidence'])
        self.assertEqual(v['assessment_type'], 'image')

    # 10. Wrong-side candidate passing through aggregator and appearing in results
    def test_10_wrong_side_candidate_through_aggregator_and_flask(self):
        det = WrongSideDetector(permitted_direction='downward')
        frame = self.road_frame.copy()
        bike_bbox = [500, 350, 580, 520]
        plate_bbox = [525, 480, 555, 505]
        all_detections = [
            {'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85},
            {'bbox': plate_bbox, 'class_name': 'number_plate', 'confidence': 0.90}
        ]
        cv2.circle(frame, (540, 470), 10, (0, 0, 255), -1)
        bikes = [{'bbox': bike_bbox, 'class_name': 'motorcycle', 'confidence': 0.85}]

        raw_violations, _ = det.evaluate_image_wrong_side(frame, bikes, all_detections)
        self.assertEqual(len(raw_violations), 1)

        # Aggregate through ViolationAggregator
        agg = ViolationAggregator()
        aggregated = agg.aggregate_frame(
            frame, [], [], raw_violations, [], frame_idx=0, timestamp_sec=0.0, is_video=False
        )
        self.assertEqual(len(aggregated), 1)
        cand = aggregated[0]
        self.assertEqual(cand['violation_type'], 'wrong_side')
        self.assertEqual(cand['title'], 'WRONG-SIDE RIDING — NEEDS REVIEW')
        self.assertEqual(cand['assessment_type'], 'image')
        self.assertIsNone(cand['track_id'])
        self.assertIsNone(cand['confidence'])
        self.assertIsNotNone(cand['evidence_path'])
        self.assertIsNotNone(cand['evidence_image'])
        self.assertTrue(os.path.exists(os.path.join(Config.BASE_DIR, cand['evidence_path'].lstrip('/'))))

        # Check DB insertion & retrieval
        job_id = f"test_job_ws_img_{uuid.uuid4().hex[:8]}"
        self.db.create_job(job_id, 'image', 'test.jpg', 'test.jpg')
        v_id = self.db.insert_violation(
            job_id=job_id,
            violation_type=cand['violation_type'],
            confidence=cand['confidence'],
            track_id=cand['track_id'],
            frame_number=0,
            timestamp_sec=0.0,
            bbox=cand['bounding_box'],
            rider_count=None,
            plate_text=None,
            plate_confidence=None,
            evidence_path=cand['evidence_path'],
            review_status=cand['review_status'],
            notes=cand['reason']
        )
        cand['id'] = v_id
        db_records = self.db.get_violations_by_job(job_id)
        self.assertEqual(len(db_records), 1)
        self.assertIsNone(db_records[0]['confidence']) # Nullable check

        # Verify results page renders standard violation card
        response = self.client.get(f'/results/{job_id}')
        self.assertEqual(response.status_code, 200)
        html = response.data.decode('utf-8')
        self.assertIn("WRONG-SIDE RIDING — NEEDS REVIEW", html)
        self.assertIn("Detected Violations (1)", html)
        self.assertIn("Uncertain", html)
        # Ensure separate road direction assessment panel is NOT present
        self.assertNotIn("Road Direction Assessment", html)

if __name__ == '__main__':
    unittest.main()
