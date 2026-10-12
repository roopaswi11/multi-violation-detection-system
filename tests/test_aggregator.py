import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.violation_aggregator import ViolationAggregator

class TestViolationAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = ViolationAggregator()
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_aggregate_and_deduplicate_video_frames(self):
        # Frame 1: track 12 has no-helmet with confidence 0.82
        h_viols_f1 = [{
            'violation_type': 'no_helmet',
            'confidence': 0.82,
            'bounding_box': [100, 100, 200, 300],
            'track_id': 12,
            'frame_number': 1
        }]
        res_f1 = self.aggregator.aggregate_frame(
            self.dummy_frame, h_viols_f1, [], [], [], frame_idx=1, is_video=True
        )
        self.assertEqual(len(res_f1), 1)

        # Frame 2: same track 12 has no-helmet with higher confidence 0.94
        h_viols_f2 = [{
            'violation_type': 'no_helmet',
            'confidence': 0.94,
            'bounding_box': [105, 105, 205, 305],
            'track_id': 12,
            'frame_number': 2
        }]
        res_f2 = self.aggregator.aggregate_frame(
            self.dummy_frame, h_viols_f2, [], [], [], frame_idx=2, is_video=True
        )

        # Check deduplicated final result across entire video
        final_video_viols = self.aggregator.get_final_video_violations()
        # Must be exactly 1 unified event, with highest confidence 0.94
        self.assertEqual(len(final_video_viols), 1)
        self.assertEqual(final_video_viols[0]['confidence'], 0.94)
        self.assertEqual(final_video_viols[0]['frame_number'], 2)

if __name__ == '__main__':
    unittest.main()
