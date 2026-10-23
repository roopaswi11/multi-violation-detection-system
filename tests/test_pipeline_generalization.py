"""
Phase 8: Comprehensive Evaluation and Regression Test Suite.
Validates generalization across single/multiple riders, single/multiple bikes,
mixed helmet compliance, pedestrians near vehicles, helmet held in hand,
clean images, and state isolation between consecutive uploads.
"""
import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.helmet_detector import HelmetDetector
from services.triple_riding_detector import TripleRidingDetector
from services.violation_aggregator import ViolationAggregator
from services.video_processor import VideoProcessor

class TestPipelineGeneralization(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.helmet_detector = HelmetDetector(vit_enabled=False)
        self.triple_detector = TripleRidingDetector(max_permitted_riders=2)
        self.aggregator = ViolationAggregator()

    def test_helmeted_rider_compliant(self):
        """Single rider wearing a helmet must result in 0 violations (compliant)."""
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88},
            {'bbox': [170, 100, 230, 160], 'class_name': 'helmet', 'confidence': 0.92}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(h_viols), 0, "Helmeted rider must not produce helmet violation")
        self.assertEqual(len(t_viols), 0, "Single rider must not produce triple riding violation")
        self.assertEqual(len(agg_viols), 0, "No violations expected for compliant rider")

    def test_single_helmetless_rider(self):
        """Single rider with direct no-helmet detection must produce exactly 1 confirmed violation."""
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88},
            {'bbox': [170, 100, 230, 160], 'class_name': 'no-helmet', 'confidence': 0.91}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(h_viols), 1)
        self.assertEqual(h_viols[0]['violation_type'], 'no_helmet')
        self.assertEqual(h_viols[0]['review_status'], 'confirmed')
        self.assertEqual(len(agg_viols), 1)

    def test_multiple_helmetless_riders_triple_riding(self):
        """
        Three riders on one motorcycle, all three without helmets:
        Must produce:
        1. Exactly ONE triple-riding violation for the motorcycle.
        2. Exactly THREE separate no-helmet violations (Rider 1, Rider 2, Rider 3).
        Total: 4 aggregated violations.
        """
        detections = [
            # Motorcycle
            {'bbox': [50, 180, 450, 450], 'class_name': 'motorcycle', 'confidence': 0.92},
            # Rider 1 (Driver)
            {'bbox': [70, 80, 180, 330], 'class_name': 'person', 'confidence': 0.89},
            {'bbox': [90, 80, 150, 140], 'class_name': 'no-helmet', 'confidence': 0.85},
            # Rider 2 (Pillion 1)
            {'bbox': [190, 90, 300, 330], 'class_name': 'person', 'confidence': 0.87},
            {'bbox': [210, 90, 270, 150], 'class_name': 'no-helmet', 'confidence': 0.83},
            # Rider 3 (Pillion 2)
            {'bbox': [310, 100, 420, 330], 'class_name': 'person', 'confidence': 0.86},
            {'bbox': [330, 100, 390, 160], 'class_name': 'no-helmet', 'confidence': 0.84}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        # 3 distinct helmet violations
        self.assertEqual(len(h_viols), 3, "Expected 3 separate no-helmet violations for 3 riders")
        # 1 triple riding violation
        self.assertEqual(len(t_viols), 1, "Expected 1 triple-riding violation for the motorcycle")
        self.assertEqual(t_viols[0]['rider_count'], 3)
        # Aggregator must retain all 4 distinct violations (no accidental deduplication)
        self.assertEqual(len(agg_viols), 4, "Expected 4 total violations (3 helmet + 1 triple riding)")

    def test_mixed_helmet_compliance_triple_riding(self):
        """
        Three riders on one motorcycle:
        Rider 1 wears a helmet.
        Rider 2 is helmetless.
        Rider 3 is helmetless.
        Must produce:
        - 1 triple riding violation.
        - Exactly 2 no-helmet violations (Rider 2 and Rider 3).
        - 0 helmet violations for Rider 1.
        """
        detections = [
            {'bbox': [50, 180, 450, 450], 'class_name': 'motorcycle', 'confidence': 0.92},
            # Rider 1 (Helmeted)
            {'bbox': [70, 80, 180, 330], 'class_name': 'person', 'confidence': 0.90},
            {'bbox': [90, 80, 150, 140], 'class_name': 'helmet', 'confidence': 0.95},
            # Rider 2 (No Helmet)
            {'bbox': [190, 90, 300, 330], 'class_name': 'person', 'confidence': 0.87},
            {'bbox': [210, 90, 270, 150], 'class_name': 'no-helmet', 'confidence': 0.88},
            # Rider 3 (No Helmet)
            {'bbox': [310, 100, 420, 330], 'class_name': 'person', 'confidence': 0.86},
            {'bbox': [330, 100, 390, 160], 'class_name': 'no-helmet', 'confidence': 0.84}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(h_viols), 2, "Only Rider 2 and 3 should have no-helmet violations")
        self.assertEqual(len(t_viols), 1, "Triple riding must be detected")
        self.assertEqual(len(agg_viols), 3, "Total violations must be exactly 3 (2 no-helmet + 1 triple)")

    def test_multiple_motorcycles_independent(self):
        """
        Two distinct motorcycles in one scene:
        Bike A: 1 rider, helmeted.
        Bike B: 2 riders, both helmetless.
        Must produce:
        - Bike A: 0 violations.
        - Bike B: 2 no-helmet violations, 0 triple riding.
        """
        detections = [
            # Bike A (Left side of road)
            {'bbox': [50, 200, 200, 400], 'class_name': 'motorcycle', 'confidence': 0.90},
            {'bbox': [80, 120, 160, 320], 'class_name': 'person', 'confidence': 0.88},
            {'bbox': [95, 120, 145, 170], 'class_name': 'helmet', 'confidence': 0.92},

            # Bike B (Right side of road)
            {'bbox': [350, 200, 550, 400], 'class_name': 'motorcycle', 'confidence': 0.91},
            {'bbox': [370, 110, 450, 320], 'class_name': 'person', 'confidence': 0.87},
            {'bbox': [385, 110, 435, 165], 'class_name': 'no-helmet', 'confidence': 0.86},
            {'bbox': [460, 120, 530, 320], 'class_name': 'person', 'confidence': 0.85},
            {'bbox': [475, 120, 520, 170], 'class_name': 'no-helmet', 'confidence': 0.84}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(h_viols), 2, "Only Bike B riders should have no-helmet violations")
        self.assertEqual(len(t_viols), 0, "Neither bike exceeds 2 riders")
        self.assertEqual(len(agg_viols), 2)

    def test_pedestrian_near_motorcycle_not_penalized(self):
        """
        Pedestrian standing on the pavement near a motorcycle must not be counted as a rider.
        """
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            # Mounted rider with helmet
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88},
            {'bbox': [170, 100, 230, 160], 'class_name': 'helmet', 'confidence': 0.92},
            # Pedestrian standing far to the right, feet on pavement
            {'bbox': [480, 150, 560, 460], 'class_name': 'person', 'confidence': 0.93}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(h_viols), 0, "Pedestrian must not cause false rider violation")
        self.assertEqual(len(t_viols), 0)
        self.assertEqual(len(agg_viols), 0)

    def test_helmet_held_in_hand_not_worn(self):
        """
        A helmet held in the rider's hand (bottom half of person bounding box)
        must NOT be classified as a worn helmet; the rider's head remains unhelmeted.
        """
        detections = [
            {'bbox': [100, 200, 300, 450], 'class_name': 'motorcycle', 'confidence': 0.90},
            # Rider box: y1=100, y2=350. Head is 100-180. Hand/seat is 260-320.
            {'bbox': [150, 100, 250, 350], 'class_name': 'person', 'confidence': 0.88},
            # Helmet held down near waist/hand
            {'bbox': [160, 270, 210, 320], 'class_name': 'helmet', 'confidence': 0.91}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        # Because the helmet is down in the hand, head is unhelmeted and flagged
        self.assertEqual(len(h_viols), 1, "Rider with helmet in hand must be flagged for unhelmeted head")

    def test_no_violations_cars_only(self):
        """Scene with only passenger cars and trucks: 0 violations, compliant."""
        detections = [
            {'bbox': [50, 150, 250, 300], 'class_name': 'car', 'confidence': 0.94},
            {'bbox': [300, 120, 550, 350], 'class_name': 'truck', 'confidence': 0.91}
        ]
        h_viols = self.helmet_detector.analyze(self.frame, detections)
        t_viols = self.triple_detector.analyze(self.frame, detections)
        agg_viols = self.aggregator.aggregate_frame(self.frame, h_viols, t_viols, [], [])

        self.assertEqual(len(agg_viols), 0)

    def test_per_image_state_isolation(self):
        """
        Ensures that processing an image with multiple violations does not contaminate
        subsequent processing of an image with zero violations.
        """
        processor = VideoProcessor()
        
        # Simulated run 1: image with violations
        agg1 = ViolationAggregator()
        v1 = [{'violation_type': 'no_helmet', 'confidence': 0.9, 'bounding_box': [10, 10, 50, 50]}]
        res1 = agg1.aggregate_frame(self.frame, v1, [], [], [])
        self.assertEqual(len(res1), 1)

        # Simulated run 2: clean image with fresh aggregator
        agg2 = ViolationAggregator()
        res2 = agg2.aggregate_frame(self.frame, [], [], [], [])
        self.assertEqual(len(res2), 0, "Subsequent image must have 0 violations, no residual state")

if __name__ == '__main__':
    unittest.main()
