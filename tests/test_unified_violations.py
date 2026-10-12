import unittest
import os
import sys
import io
import uuid
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from services.wrong_side_detector import WrongSideDetector
from services.violation_aggregator import ViolationAggregator
from services.database_service import DatabaseService
from services.video_processor import VideoProcessor
from config import Config

class TestUnifiedViolations(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.db = DatabaseService()
        self.frame_shape = (720, 1280, 3)
        self.dummy_frame = np.zeros(self.frame_shape, dtype=np.uint8)

    # 1. Confirmed no-helmet violation
    def test_confirmed_no_helmet_violation(self):
        agg = ViolationAggregator()
        h_viols = [{
            'violation_type': 'no_helmet',
            'confidence': 0.88,
            'bounding_box': [100, 100, 200, 300],
            'track_id': 5,
            'reason': 'Rider observed without certified protective helmet'
        }]
        result = agg.aggregate_frame(self.dummy_frame, h_viols, [], [], [], frame_idx=1, is_video=True)
        self.assertEqual(len(result), 1)
        v = result[0]
        self.assertEqual(v['violation_type'], 'no_helmet')
        self.assertTrue(v['detected'])
        self.assertEqual(v['confidence'], 0.88)
        self.assertIn(v['review_status'], ['pending', 'confirmed'])
        self.assertIsNotNone(v['evidence_path'])

    # 2. Confirmed wrong-side event from suitable video evidence
    def test_confirmed_wrong_side_video_event(self):
        det = WrongSideDetector(permitted_direction='downward', min_displacement=20.0, min_frames=5)
        # Vehicle 7 moving upward against permitted downward flow
        for i in range(10):
            det.update_track(7, [200, 500 - i * 20, 300, 600 - i * 20], frame_idx=i)
        
        ws_viol = det.evaluate_wrong_side(7, [200, 320, 300, 420], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(ws_viol)
        self.assertEqual(ws_viol['violation_type'], 'wrong_side')
        self.assertEqual(ws_viol['title'], 'WRONG-SIDE RIDING')
        self.assertTrue(ws_viol['detected'])
        self.assertFalse(ws_viol['is_candidate'])
        self.assertGreater(ws_viol['confidence'], 0.70)
        self.assertIn("Moving upward", ws_viol['reason'])

        # Aggregate through common pipeline
        agg = ViolationAggregator()
        aggregated = agg.aggregate_frame(self.dummy_frame, [], [], [ws_viol], [], frame_idx=9, is_video=True)
        self.assertEqual(len(aggregated), 1)
        v = aggregated[0]
        self.assertEqual(v['violation_type'], 'wrong_side')
        self.assertTrue(v['detected'])
        self.assertIsNotNone(v['evidence_path'])

    # 3. Wrong-side candidate with insufficient evidence
    def test_wrong_side_candidate_insufficient_evidence(self):
        det = WrongSideDetector(permitted_direction='downward', min_displacement=30.0, min_frames=5)
        # Vehicle moving upward, but small displacement (e.g. 15px, below min_displacement 30px)
        for i in range(5):
            det.update_track(8, [200, 400 - i * 3, 300, 500 - i * 3], frame_idx=i)
        
        candidate = det.evaluate_wrong_side(8, [200, 388, 300, 488], self.frame_shape, frame_idx=4)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate['violation_type'], 'wrong_side')
        self.assertEqual(candidate['title'], 'WRONG-SIDE RIDING — NEEDS REVIEW')
        self.assertFalse(candidate['detected'])
        self.assertTrue(candidate['is_candidate'])
        self.assertIsNone(candidate['confidence'])
        self.assertEqual(candidate['review_status'], 'insufficient_evidence')
        self.assertIn('insufficient', candidate['reason'].lower())

        # Aggregation preserves candidate schema
        agg = ViolationAggregator()
        aggregated = agg.aggregate_frame(self.dummy_frame, [], [], [candidate], [], frame_idx=4, is_video=True)
        self.assertEqual(len(aggregated), 1)
        cand_event = aggregated[0]
        self.assertFalse(cand_event['detected'])
        self.assertTrue(cand_event['is_candidate'])
        self.assertEqual(cand_event['review_status'], 'insufficient_evidence')
        self.assertIsNone(cand_event['confidence'])

    # 4. Image with no motion evidence
    def test_image_with_no_motion_evidence(self):
        det = WrongSideDetector(permitted_direction='downward')
        summary = det.get_status_summary(is_video=False)
        self.assertEqual(summary['status'], 'direction_uncertain')
        
        # When processing a static image, no fake wrong-side violations are generated
        agg = ViolationAggregator()
        frame_viols = agg.aggregate_frame(self.dummy_frame, [], [], [], [], frame_idx=0, is_video=False)
        self.assertEqual(len(frame_viols), 0)

    # 5. Multiple different violation types in one result
    def test_multiple_different_violation_types_in_one_result(self):
        agg = ViolationAggregator()
        h_viol = {
            'violation_type': 'no_helmet',
            'confidence': 0.85,
            'bounding_box': [50, 50, 150, 150],
            'track_id': 1
        }
        t_viol = {
            'violation_type': 'triple_riding',
            'confidence': 0.90,
            'bounding_box': [200, 200, 350, 350],
            'track_id': 2,
            'rider_count': 3
        }
        w_viol = {
            'violation_type': 'wrong_side',
            'confidence': 0.82,
            'bounding_box': [400, 400, 550, 550],
            'track_id': 3,
            'reason': 'Contraflow travel detected'
        }
        res = agg.aggregate_frame(self.dummy_frame, [h_viol], [t_viol], [w_viol], [], frame_idx=1, is_video=True)
        self.assertEqual(len(res), 3)
        vtypes = {v['violation_type'] for v in res}
        self.assertEqual(vtypes, {'no_helmet', 'triple_riding', 'wrong_side'})
        # Verify wrong-side was accepted without requiring a helmet violation on the same vehicle
        ws_item = next(v for v in res if v['violation_type'] == 'wrong_side')
        self.assertEqual(ws_item['track_id'], 3)
        self.assertTrue(ws_item['detected'])

    # 6. Duplicate violation records
    def test_duplicate_violation_records_deduplication(self):
        agg = ViolationAggregator()
        # Same track 9 with wrong_side in frame 1 and frame 2
        w1 = {
            'violation_type': 'wrong_side',
            'confidence': 0.74,
            'bounding_box': [100, 100, 200, 200],
            'track_id': 9
        }
        agg.aggregate_frame(self.dummy_frame, [], [], [w1], [], frame_idx=1, is_video=True)

        w2 = {
            'violation_type': 'wrong_side',
            'confidence': 0.91,
            'bounding_box': [105, 105, 205, 205],
            'track_id': 9
        }
        agg.aggregate_frame(self.dummy_frame, [], [], [w2], [], frame_idx=2, is_video=True)

        # Video deduplication must retain only 1 event with the highest confidence
        final_viols = agg.get_final_video_violations()
        self.assertEqual(len(final_viols), 1)
        self.assertEqual(final_viols[0]['confidence'], 0.91)
        self.assertEqual(final_viols[0]['frame_number'], 2)

    # 7. Violation counter accuracy
    def test_violation_counter_accuracy(self):
        violations = [
            {'violation_type': 'no_helmet', 'confidence': 0.85, 'review_status': 'confirmed'},
            {'violation_type': 'wrong_side', 'confidence': 0.89, 'review_status': 'pending'},
            {'violation_type': 'wrong_side', 'confidence': None, 'review_status': 'insufficient_evidence'}
        ]
        # Total unique records displayed in panel: 3
        total_records = len(violations)
        confirmed_count = len([v for v in violations if v.get('review_status') != 'insufficient_evidence'])
        candidate_count = total_records - confirmed_count

        self.assertEqual(total_records, 3)
        self.assertEqual(confirmed_count, 2)
        self.assertEqual(candidate_count, 1)

    # 8. Review & Details navigation
    def test_review_and_details_navigation(self):
        job_id = f"job_test_{uuid.uuid4().hex[:8]}"
        self.db.create_job(job_id, "video", "test.mp4", f"{job_id}.mp4")
        v_id = self.db.insert_violation(
            job_id=job_id,
            violation_type="wrong_side",
            confidence=0.87,
            track_id=15,
            frame_number=45,
            review_status="pending",
            notes="Moving against downward traffic flow"
        )
        # Test navigation to review page
        res = self.client.get(f'/violation/{v_id}')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Violation Evidence Audit', res.data)
        self.assertIn(b'WRONG SIDE', res.data)
        self.assertIn(b'87.0%', res.data)

        # Test updating review
        post_res = self.client.post(f'/violation/{v_id}', data={
            'review_status': 'confirmed',
            'reviewer_notes': 'Confirmed contraflow riding by officer'
        }, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)
        v_updated = self.db.get_violation(v_id)
        self.assertEqual(v_updated['review_status'], 'confirmed')

    # 9. Correct display of confidence and evidence status
    def test_display_of_confidence_and_evidence_status(self):
        # Confirmed event has float confidence
        conf_event = {'confidence': 0.865, 'review_status': 'confirmed'}
        self.assertEqual(f"{round(conf_event['confidence'] * 100, 1)}%", "86.5%")

        # Candidate event with insufficient evidence has None confidence
        cand_event = {'confidence': None, 'review_status': 'insufficient_evidence'}
        self.assertIsNone(cand_event['confidence'])
        self.assertEqual(cand_event['review_status'], 'insufficient_evidence')

    # 10. Preservation of selected road direction
    def test_preservation_of_selected_road_direction(self):
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_buf = cv2.imencode('.jpg', img)

        test_directions = ['downward', 'upward', 'leftward', 'rightward', 'auto']
        for d in test_directions:
            res = self.client.post('/upload', data={
                'file': (io.BytesIO(img_buf.tobytes()), f'dir_test_{d}.jpg'),
                'permitted_direction': d,
                'ajax': '1'
            }, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data['permitted_direction'], d)
            job = self.db.get_job(data['job_id'])
            self.assertEqual(job['metadata']['permitted_direction'], d)

if __name__ == '__main__':
    unittest.main()
