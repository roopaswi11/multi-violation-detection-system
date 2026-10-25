import unittest
import numpy as np
import cv2
import os
from services.helmet_detector import HelmetDetector
from services.triple_riding_detector import TripleRidingDetector
from services.violation_aggregator import ViolationAggregator

class TestHelmetAndRiderAssociationRegression(unittest.TestCase):
    def setUp(self):
        self.helmet_detector = HelmetDetector()
        self.triple_detector = TripleRidingDetector()
        self.aggregator = ViolationAggregator()
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_case_1_two_helmetless_riders_on_one_scooter(self):
        """1. Two helmetless riders on one scooter: 2 NO HELMET violations, 0 triple riding."""
        bike = {'bbox': [100.0, 150.0, 400.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        driver = {'bbox': [180.0, 70.0, 290.0, 320.0], 'class_name': 'person', 'confidence': 0.89}
        pillion = {'bbox': [260.0, 60.0, 370.0, 310.0], 'class_name': 'person', 'confidence': 0.85}
        nh_driver = {'bbox': [200.0, 65.0, 270.0, 140.0], 'class_name': 'no helmet', 'confidence': 0.82}
        nh_pillion = {'bbox': [280.0, 55.0, 350.0, 130.0], 'class_name': 'no helmet', 'confidence': 0.80}

        dets = [bike, driver, pillion, nh_driver, nh_pillion]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)

        self.assertEqual(len(h_viols), 2)
        self.assertEqual(len(t_viols), 0)
        for v in h_viols:
            self.assertEqual(v['violation_type'], 'no_helmet')
            self.assertEqual(v['review_status'], 'confirmed')

    def test_case_2_two_helmeted_riders_on_one_scooter(self):
        """2. Two helmeted riders on one scooter: 0 violations (fully compliant)."""
        bike = {'bbox': [100.0, 150.0, 400.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        driver = {'bbox': [180.0, 70.0, 290.0, 320.0], 'class_name': 'person', 'confidence': 0.89}
        pillion = {'bbox': [260.0, 60.0, 370.0, 310.0], 'class_name': 'person', 'confidence': 0.85}
        h_driver = {'bbox': [200.0, 65.0, 270.0, 140.0], 'class_name': 'helmet', 'confidence': 0.85}
        h_pillion = {'bbox': [280.0, 55.0, 350.0, 130.0], 'class_name': 'helmet', 'confidence': 0.83}

        dets = [bike, driver, pillion, h_driver, h_pillion]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)

        self.assertEqual(len(h_viols), 0)
        self.assertEqual(len(t_viols), 0)

    def test_case_3_three_helmetless_riders_on_one_motorcycle(self):
        """3. Three helmetless riders on one motorcycle: 3 NO HELMET, 1 Triple Riding."""
        bike = {'bbox': [80.0, 150.0, 450.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        r1 = {'bbox': [150.0, 70.0, 240.0, 320.0], 'class_name': 'person', 'confidence': 0.88}
        r2 = {'bbox': [230.0, 65.0, 320.0, 310.0], 'class_name': 'person', 'confidence': 0.85}
        r3 = {'bbox': [300.0, 60.0, 390.0, 300.0], 'class_name': 'person', 'confidence': 0.82}
        nh1 = {'bbox': [160.0, 65.0, 220.0, 130.0], 'class_name': 'no helmet', 'confidence': 0.85}
        nh2 = {'bbox': [240.0, 60.0, 300.0, 125.0], 'class_name': 'no helmet', 'confidence': 0.83}
        nh3 = {'bbox': [310.0, 55.0, 370.0, 120.0], 'class_name': 'no helmet', 'confidence': 0.80}

        dets = [bike, r1, r2, r3, nh1, nh2, nh3]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)

        self.assertEqual(len(h_viols), 3)
        self.assertEqual(len(t_viols), 1)
        self.assertEqual(t_viols[0]['violation_type'], 'triple_riding')

    def test_case_4_three_helmeted_riders_on_one_motorcycle(self):
        """4. Three helmeted riders on one motorcycle: 0 NO HELMET, 1 Triple Riding."""
        bike = {'bbox': [80.0, 150.0, 450.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        r1 = {'bbox': [150.0, 70.0, 240.0, 320.0], 'class_name': 'person', 'confidence': 0.88}
        r2 = {'bbox': [230.0, 65.0, 320.0, 310.0], 'class_name': 'person', 'confidence': 0.85}
        r3 = {'bbox': [300.0, 60.0, 390.0, 300.0], 'class_name': 'person', 'confidence': 0.82}
        h1 = {'bbox': [160.0, 65.0, 220.0, 130.0], 'class_name': 'helmet', 'confidence': 0.85}
        h2 = {'bbox': [240.0, 60.0, 300.0, 125.0], 'class_name': 'helmet', 'confidence': 0.83}
        h3 = {'bbox': [310.0, 55.0, 370.0, 120.0], 'class_name': 'helmet', 'confidence': 0.80}

        dets = [bike, r1, r2, r3, h1, h2, h3]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)

        self.assertEqual(len(h_viols), 0)
        self.assertEqual(len(t_viols), 1)

    def test_case_5_mixed_helmet_compliance(self):
        """5. Mixed helmet compliance: Driver has helmet, pillion is helmetless."""
        bike = {'bbox': [100.0, 150.0, 400.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        driver = {'bbox': [180.0, 70.0, 290.0, 320.0], 'class_name': 'person', 'confidence': 0.89}
        pillion = {'bbox': [260.0, 60.0, 370.0, 310.0], 'class_name': 'person', 'confidence': 0.85}
        h_driver = {'bbox': [200.0, 65.0, 270.0, 140.0], 'class_name': 'helmet', 'confidence': 0.85}
        nh_pillion = {'bbox': [280.0, 55.0, 350.0, 130.0], 'class_name': 'no helmet', 'confidence': 0.80}

        dets = [bike, driver, pillion, h_driver, nh_pillion]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        self.assertEqual(len(h_viols), 1)
        self.assertIn('Pillion', h_viols[0]['title'])

    def test_case_6_multiple_motorcycles_and_nearby_pedestrians(self):
        """6. Multiple motorcycles and nearby pedestrians: Pedestrians not counted as riders."""
        bike1 = {'bbox': [50.0, 150.0, 250.0, 380.0], 'class_name': 'motorcycle', 'confidence': 0.85}
        rider1 = {'bbox': [90.0, 70.0, 190.0, 300.0], 'class_name': 'person', 'confidence': 0.85}
        
        bike2 = {'bbox': [350.0, 150.0, 550.0, 380.0], 'class_name': 'motorcycle', 'confidence': 0.85}
        rider2 = {'bbox': [390.0, 70.0, 490.0, 300.0], 'class_name': 'person', 'confidence': 0.85}
        
        # Pedestrian walking on sidewalk (y2 is ground level, away from bike footprint)
        pedestrian = {'bbox': [580.0, 120.0, 630.0, 420.0], 'class_name': 'person', 'confidence': 0.82}

        dets = [bike1, rider1, bike2, rider2, pedestrian]
        assoc = self.helmet_detector.associate_riders_to_motorcycles([bike1, bike2], [rider1, rider2, pedestrian])
        
        self.assertEqual(len(assoc[0]), 1)
        self.assertEqual(len(assoc[1]), 1)
        self.assertNotIn(pedestrian, assoc[0])
        self.assertNotIn(pedestrian, assoc[1])

    def test_case_7_helmeted_motorcyclist_next_to_helmetless_scooter(self):
        """7. A helmeted motorcyclist next to a helmetless scooter rider: Independent evaluation."""
        bike_mc = {'bbox': [10.0, 100.0, 110.0, 280.0], 'class_name': 'motorcycle', 'confidence': 0.80}
        rider_mc = {'bbox': [15.0, 50.0, 105.0, 250.0], 'class_name': 'person', 'confidence': 0.85}
        h_mc = {'bbox': [20.0, 45.0, 100.0, 255.0], 'class_name': 'helmet', 'confidence': 0.55}
        
        bike_sc = {'bbox': [180.0, 120.0, 450.0, 350.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        rider_sc = {'bbox': [220.0, 60.0, 340.0, 290.0], 'class_name': 'person', 'confidence': 0.88}
        nh_sc = {'bbox': [245.0, 55.0, 315.0, 135.0], 'class_name': 'no helmet', 'confidence': 0.80}

        dets = [bike_mc, rider_mc, h_mc, bike_sc, rider_sc, nh_sc]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        
        self.assertEqual(len(h_viols), 1)
        # Violation must belong to the scooter rider, not the motorcyclist
        self.assertGreater(h_viols[0]['bounding_box'][0], 150.0)

    def test_case_8_helmet_held_in_hand(self):
        """8. A helmet held in rider's hand/lower body: Treated as UNWORN (Violation)."""
        bike = {'bbox': [100.0, 150.0, 400.0, 400.0], 'class_name': 'motorcycle', 'confidence': 0.88}
        rider = {'bbox': [180.0, 70.0, 290.0, 320.0], 'class_name': 'person', 'confidence': 0.89}
        # Helmet held down near knee/seat level (y1 = 220)
        held_h = {'bbox': [210.0, 220.0, 260.0, 275.0], 'class_name': 'helmet', 'confidence': 0.75}

        dets = [bike, rider, held_h]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        self.assertEqual(len(h_viols), 1)
        self.assertIn('HELMET IN HAND', h_viols[0]['title'])

    def test_case_9_partially_occluded_and_distant_riders(self):
        """9. Partially occluded and distant riders: Handled cleanly without crashes."""
        bike = {'bbox': [200.0, 100.0, 300.0, 220.0], 'class_name': 'motorcycle', 'confidence': 0.70}
        rider = {'bbox': [220.0, 60.0, 280.0, 180.0], 'class_name': 'person', 'confidence': 0.75}
        dets = [bike, rider]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        self.assertIsInstance(h_viols, list)
    def test_case_10_image_with_no_violations(self):
        """10. Scene with only passenger cars: 0 violations, status compliant."""
        car = {'bbox': [100.0, 150.0, 350.0, 320.0], 'class_name': 'car', 'confidence': 0.90}
        dets = [car]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)
        self.assertEqual(len(h_viols), 0)
        self.assertEqual(len(t_viols), 0)

    def test_case_11_foreground_scooter_with_distant_background_persons(self):
        """11. Foreground scooter with distant background persons: Background persons must NOT be counted on foreground bike."""
        # Wide foreground scooter
        fg_bike = {'bbox': [120.0, 65.0, 394.0, 253.0], 'class_name': 'motorcycle', 'confidence': 0.85}
        fg_rider = {'bbox': [213.0, 30.0, 305.0, 266.0], 'class_name': 'person', 'confidence': 0.88}
        fg_helmet = {'bbox': [208.0, 16.0, 303.0, 242.0], 'class_name': 'helmet', 'confidence': 0.85}
        
        # Distant background persons overlapping horizontally with the wide scooter box
        bg_p1 = {'bbox': [145.0, 10.0, 198.0, 112.0], 'class_name': 'person', 'confidence': 0.56} # small distant person (h=102 vs 236)
        bg_p2 = {'bbox': [190.0, 20.0, 218.0, 78.0], 'class_name': 'person', 'confidence': 0.38}  # tiny distant person (h=58 vs 236)
        
        dets = [fg_bike, fg_rider, fg_helmet, bg_p1, bg_p2]
        h_viols = self.helmet_detector.analyze(self.frame, dets)
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)
        
        # Must produce 0 helmet violations and 0 triple riding violations
        self.assertEqual(len(h_viols), 0)
        self.assertEqual(len(t_viols), 0)
        
        # Verify assigned riders on fg_bike is exactly 1 (the foreground rider)
        assigned = self.triple_detector.associate_riders_to_motorcycles([fg_bike], [fg_rider, bg_p1, bg_p2])
        self.assertEqual(len(assigned[0]), 1)
        self.assertEqual(assigned[0][0]['bbox'], fg_rider['bbox'])

    def test_case_12_genuine_triple_riding_preserved(self):
        """12. Genuine triple riding on 1 motorcycle: Must reliably detect 1 triple riding violation."""
        bike = {'bbox': [104.0, 125.0, 470.0, 276.0], 'class_name': 'motorcycle', 'confidence': 0.80}
        r1 = {'bbox': [246.0, 0.0, 402.0, 276.0], 'class_name': 'person', 'confidence': 0.85} # driver
        r2 = {'bbox': [192.0, 7.0, 340.0, 276.0], 'class_name': 'person', 'confidence': 0.75}  # pillion 1
        r3 = {'bbox': [192.0, 5.0, 280.0, 184.0], 'class_name': 'person', 'confidence': 0.65}  # pillion 2
        
        dets = [bike, r1, r2, r3]
        t_viols = self.triple_detector.analyze(self.frame, dets, is_video=False)
        self.assertEqual(len(t_viols), 1)
        self.assertEqual(t_viols[0]['violation_type'], 'triple_riding')
        self.assertEqual(t_viols[0]['rider_count'], 3)

if __name__ == '__main__':
    unittest.main()
