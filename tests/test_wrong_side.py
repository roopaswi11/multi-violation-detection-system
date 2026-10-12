import unittest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.wrong_side_detector import WrongSideDetector

class TestWrongSideDetector(unittest.TestCase):
    def setUp(self):
        self.frame_shape = (720, 1280, 3)

    def test_downward_permitted_direction(self):
        # Permitted: downward (Top to Bottom, dy > 0)
        det = WrongSideDetector(permitted_direction='downward', min_displacement=20.0, min_frames=5)
        
        # Vehicle 1 moving downward (compliant)
        for i in range(10):
            det.update_track(1, [200, 100 + i * 20, 300, 200 + i * 20], frame_idx=i)
        viol1 = det.evaluate_wrong_side(1, [200, 280, 300, 380], self.frame_shape, frame_idx=9)
        self.assertIsNone(viol1)

        # Vehicle 2 moving upward (contraflow violation)
        for i in range(10):
            det.update_track(2, [200, 500 - i * 20, 300, 600 - i * 20], frame_idx=i)
        viol2 = det.evaluate_wrong_side(2, [200, 320, 300, 420], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol2)
        self.assertEqual(viol2['violation_type'], 'wrong_side')
        self.assertIn("Moving upward", viol2['motion_desc'])

    def test_upward_permitted_direction(self):
        # Permitted: upward (Bottom to Top, dy < 0)
        det = WrongSideDetector(permitted_direction='upward', min_displacement=20.0, min_frames=5)
        
        # Vehicle 1 moving upward (compliant)
        for i in range(10):
            det.update_track(1, [200, 500 - i * 20, 300, 600 - i * 20], frame_idx=i)
        viol1 = det.evaluate_wrong_side(1, [200, 320, 300, 420], self.frame_shape, frame_idx=9)
        self.assertIsNone(viol1)

        # Vehicle 2 moving downward (contraflow violation)
        for i in range(10):
            det.update_track(2, [200, 100 + i * 20, 300, 200 + i * 20], frame_idx=i)
        viol2 = det.evaluate_wrong_side(2, [200, 280, 300, 380], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol2)
        self.assertEqual(viol2['violation_type'], 'wrong_side')
        self.assertIn("Moving downward", viol2['motion_desc'])

    def test_leftward_permitted_direction(self):
        # Permitted: leftward (Right to Left, dx < 0)
        det = WrongSideDetector(permitted_direction='leftward', min_displacement=20.0, min_frames=5)
        
        # Vehicle 1 moving leftward (compliant: x decreases)
        for i in range(10):
            det.update_track(1, [500 - i * 20, 200, 600 - i * 20, 300], frame_idx=i)
        viol1 = det.evaluate_wrong_side(1, [320, 200, 420, 300], self.frame_shape, frame_idx=9)
        self.assertIsNone(viol1)

        # Vehicle 2 moving rightward (contraflow violation: x increases)
        for i in range(10):
            det.update_track(2, [100 + i * 20, 200, 200 + i * 20, 300], frame_idx=i)
        viol2 = det.evaluate_wrong_side(2, [280, 200, 380, 300], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol2)
        self.assertEqual(viol2['violation_type'], 'wrong_side')
        self.assertIn("Moving rightward", viol2['motion_desc'])

    def test_rightward_permitted_direction(self):
        # Permitted: rightward (Left to Right, dx > 0)
        det = WrongSideDetector(permitted_direction='rightward', min_displacement=20.0, min_frames=5)
        
        # Vehicle 1 moving rightward (compliant: x increases)
        for i in range(10):
            det.update_track(1, [100 + i * 20, 200, 200 + i * 20, 300], frame_idx=i)
        viol1 = det.evaluate_wrong_side(1, [280, 200, 380, 300], self.frame_shape, frame_idx=9)
        self.assertIsNone(viol1)

        # Vehicle 2 moving leftward (contraflow violation: x decreases)
        for i in range(10):
            det.update_track(2, [500 - i * 20, 200, 600 - i * 20, 300], frame_idx=i)
        viol2 = det.evaluate_wrong_side(2, [320, 200, 420, 300], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol2)
        self.assertEqual(viol2['violation_type'], 'wrong_side')
        self.assertIn("Moving leftward", viol2['motion_desc'])

    def test_auto_direction_consensus_vs_contraflow(self):
        # Permitted: auto (infers dominant direction from consensus)
        det = WrongSideDetector(permitted_direction='auto', min_displacement=20.0, min_frames=5)
        
        # 5 vehicles moving downward establishing dominant flow
        for tid in [10, 11, 12, 14, 15]:
            for i in range(10):
                det.update_track(tid, [50 * tid, 100 + i * 25, 50 * tid + 80, 200 + i * 25], frame_idx=i)
        
        flow_angle = det.infer_dominant_flow()
        self.assertIsNotNone(flow_angle)
        self.assertAlmostEqual(flow_angle, 90.0, delta=15.0)

        # 6th vehicle moving upward against consensus
        for i in range(10):
            det.update_track(13, [500, 600 - i * 25, 580, 700 - i * 25], frame_idx=i)
        viol = det.evaluate_wrong_side(13, [500, 375, 580, 475], self.frame_shape, frame_idx=9)
        self.assertIsNotNone(viol)
        self.assertEqual(viol['violation_type'], 'wrong_side')
        self.assertIn("dominant traffic flow", viol['motion_desc'])

    def test_auto_direction_insufficient_evidence(self):
        # Auto mode with fewer than 3 moving vehicles
        det = WrongSideDetector(permitted_direction='auto', min_displacement=20.0, min_frames=5)
        det.update_track(1, [100, 100, 150, 150], frame_idx=0)
        summary = det.get_status_summary(is_video=True)
        self.assertEqual(summary['status'], 'direction_uncertain')
        self.assertIn("Wrong-side status could not be verified", summary['message'])

    def test_image_mode_does_not_invent_movement(self):
        # For still images, both auto and specified direction return direction_uncertain without inventing violations
        det_auto = WrongSideDetector(permitted_direction='auto')
        summary_auto = det_auto.get_status_summary(is_video=False)
        self.assertEqual(summary_auto['status'], 'direction_uncertain')
        self.assertIn("Wrong-side status could not be verified", summary_auto['message'])

        det_manual = WrongSideDetector(permitted_direction='downward')
        summary_manual = det_manual.get_status_summary(is_video=False)
        self.assertEqual(summary_manual['status'], 'direction_uncertain')
        self.assertIn("Downward Flow", summary_manual['selected_direction_label'])

    def test_stationary_motorcycle_not_flagged(self):
        det = WrongSideDetector(permitted_direction='downward', min_displacement=20.0, min_frames=5)
        track_id = 99
        for i in range(10):
            bbox = [200, 300 + (i % 2), 300, 400 + (i % 2)]
            det.update_track(track_id, bbox, frame_idx=i, timestamp=i * 0.1)

        latest_bbox = [200, 300, 300, 400]
        violation = det.evaluate_wrong_side(track_id, latest_bbox, self.frame_shape, frame_idx=9)
        self.assertIsNone(violation)

if __name__ == '__main__':
    unittest.main()
