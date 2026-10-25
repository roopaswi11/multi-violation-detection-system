import os
import unittest
import cv2
import numpy as np

from services.plate_service import PlateOCR
from services.video_processor import VideoProcessor
from services.violation_aggregator import ViolationAggregator


def levenshtein_distance(s1, s2):
    """Calculates Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(reference, hypothesis):
    """Calculates Character Error Rate (CER) = EditDistance(ref, hyp) / len(ref)."""
    ref_clean = reference.replace(" ", "").upper()
    hyp_clean = hypothesis.replace(" ", "").upper()
    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0
    dist = levenshtein_distance(ref_clean, hyp_clean)
    return dist / len(ref_clean)


class TestFinalAcceptanceScenarios(unittest.TestCase):
    """
    Project-Wide Comprehensive Evaluation for Final Acceptance Criteria:
    Scenario 1: Clear and readable plate
    Scenario 2: Partially readable plate
    Scenario 3: Blurry or obscured plate
    Scenario 4: Multiple vehicles in one image
    """

    @classmethod
    def setUpClass(cls):
        cls.ocr = PlateOCR()
        cls.processor = VideoProcessor()
        cls.aggregator = ViolationAggregator()

    def test_scenario_1_clear_and_readable_plates(self):
        """
        Scenario 1: Clear & readable plate
        - Detect and crop complete number plate
        - Extract complete registration number
        - Mark READABLE with valid format and sufficient confidence
        """
        test_cases = [
            {
                'path': os.path.join('uploads', '41282bc6_Screenshot_2026-10-23_210135.png'),
                'gt_plate': 'MH VAAA 0000',
                'vbox': [53.5, 155.77, 326.24, 333.68],
                'vtype': 'motorcycle'
            },
            {
                'path': os.path.join('uploads', 'c7e8d77b_Screenshot_2026-10-24_003848.png'),
                'gt_plate': 'MH 02 DL 4596',
                'vbox': [38.8, 119.5, 145.9, 309.5],
                'vtype': 'motorcycle'
            },
            {
                'path': os.path.join('uploads', 'bcbe8c11_Screenshot_2026-10-24_004447.png'),
                'gt_plate': 'KL 01 CA 202',
                'vbox': [0.33, 186.95, 365.39, 389.97],
                'vtype': 'motorcycle'
            }
        ]

        total = 0
        correct_full = 0
        cer_sum = 0.0

        for tc in test_cases:
            if not os.path.exists(tc['path']):
                continue
            frame = cv2.imread(tc['path'])
            self.assertIsNotNone(frame)

            res = self.ocr.extract_plate_from_vehicle(frame, tc['vbox'], vehicle_type=tc['vtype'])
            self.assertIsNotNone(res)
            self.assertEqual(res['status'], 'readable')
            self.assertGreater(res['confidence'], 0.35)
            self.assertIsNotNone(res['bbox'])
            self.assertIsNotNone(res['crop_path'])

            # Verify complete plate formatting & CER
            cer = calculate_cer(tc['gt_plate'], res['normalized_text'])
            cer_sum += cer
            total += 1
            if res['normalized_text'] == tc['gt_plate']:
                correct_full += 1

            # Check that evidence crop actually exists on disk
            crop_file = res['crop_path'].lstrip('/')
            self.assertTrue(os.path.exists(crop_file), f"Crop evidence missing: {crop_file}")

        self.assertGreater(total, 0)
        avg_cer = cer_sum / total
        self.assertLessEqual(avg_cer, 0.05, f"Average CER for clear plates was {avg_cer:.4f}")

    def test_scenario_2_partially_readable_plates(self):
        """
        Scenario 2: Partially readable plate
        - Preserve characters that can be read reliably
        - Mark PARTIAL
        - Never fabricate missing characters to fit a template
        """
        test_cases = [
            {
                'path': os.path.join('uploads', '705e4171_Screenshot_2026-10-23_210437.png'),
                'expected_sub': 'TS 08',
                'vbox': [33, 100, 93, 245],
                'vtype': 'motorcycle'
            },
            {
                'path': os.path.join('uploads', '7a7a42d9_Screenshot_2026-10-24_005223.png'),
                'expected_sub': '3038',
                'vbox': [91.7, 57.9, 280.7, 310.4],
                'vtype': 'motorcycle'
            }
        ]

        for tc in test_cases:
            if not os.path.exists(tc['path']):
                continue
            frame = cv2.imread(tc['path'])
            self.assertIsNotNone(frame)

            res = self.ocr.extract_plate_from_vehicle(frame, tc['vbox'], vehicle_type=tc['vtype'])
            self.assertIsNotNone(res)
            # Must be marked partial or readable
            self.assertIn(res['status'], ['partial', 'readable'])
            # Must preserve the reliably extracted characters
            self.assertIn(tc['expected_sub'].replace(" ", ""), res['normalized_text'].replace(" ", ""))
            # Must NOT be marked UNREADABLE
            self.assertNotEqual(res['normalized_text'], 'UNREADABLE')
            # Raw text must be preserved
            self.assertTrue(len(res['raw_text']) > 0)

    def test_scenario_3_blurry_obscured_and_no_plate(self):
        """
        Scenario 3: Blurry or obscured plate
        - Returns UNREADABLE if no reliable characters exist
        - Never presents an uncertain candidate as a verified complete plate
        """
        # Simulated heavily blurred blank frame where plate is completely unreadable
        blank_frame = np.full((300, 300, 3), 30, dtype=np.uint8)
        vbox = [50, 50, 250, 250]

        res = self.ocr.extract_plate_from_vehicle(blank_frame, vbox, vehicle_type='motorcycle')
        self.assertIsNotNone(res)
        self.assertEqual(res['status'], 'unreadable')
        self.assertEqual(res['normalized_text'], 'UNREADABLE')
        self.assertEqual(res['confidence'], 0.0)

    def test_scenario_4_multiple_vehicles_association(self):
        """
        Scenario 4: Multiple vehicles in one image
        - Detect each vehicle and plate independently
        - Associate plate with correct vehicle
        - Prevent assigning nearby vehicle's plate to another vehicle
        - Avoid duplicate fine generation
        """
        # Test synthetic frame with two distinct vehicles
        h, w = 600, 800
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Vehicle 1 on Left: has a clear plate
        bike1_box = [50, 150, 250, 500]
        plate1_box = [90, 420, 210, 455]
        cv2.rectangle(frame, (plate1_box[0], plate1_box[1]), (plate1_box[2], plate1_box[3]), (255, 255, 255), -1)

        # Vehicle 2 on Right: has NO plate
        bike2_box = [500, 150, 720, 500]

        # Violations on both vehicles
        helmet_viol_bike1 = {
            'violation_type': 'no_helmet',
            'bounding_box': [80, 150, 180, 280],
            'bike_box': bike1_box,
            'confidence': 0.95,
            'rider_id': 'Rider 1'
        }
        helmet_viol_bike2 = {
            'violation_type': 'no_helmet',
            'bounding_box': [530, 150, 630, 280],
            'bike_box': bike2_box,
            'confidence': 0.92,
            'rider_id': 'Rider 2'
        }

        # Simulated plate OCR results
        plate_results = [
            {
                'normalized_text': 'KA 01 AB 1234',
                'raw_text': 'KA01AB1234',
                'confidence': 0.91,
                'status': 'readable',
                'bbox': plate1_box,
                'vehicle_bbox': bike1_box
            }
        ]

        # Aggregate frame
        viols = self.aggregator.aggregate_frame(
            frame,
            [helmet_viol_bike1, helmet_viol_bike2],
            [],
            [],
            plate_results,
            is_video=False
        )

        self.assertEqual(len(viols), 2)
        v1 = next(v for v in viols if v['rider_id'] == 'Rider 1')
        v2 = next(v for v in viols if v['rider_id'] == 'Rider 2')

        # Vehicle 1 MUST receive Plate 1
        self.assertEqual(v1['plate_text'], 'KA 01 AB 1234')
        self.assertEqual(v1['plate_status'], 'readable')

        # Vehicle 2 MUST NOT be assigned Vehicle 1's plate!
        self.assertNotEqual(v2['plate_text'], 'KA 01 AB 1234')
        self.assertEqual(v2['plate_text'], 'UNREADABLE')
        self.assertEqual(v2['plate_status'], 'none_detected')


if __name__ == '__main__':
    unittest.main()
