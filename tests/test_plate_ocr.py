import unittest
import numpy as np
import cv2
import os
from services.plate_service import PlateOCR, INDIAN_STATE_CODES


class TestPlateOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = PlateOCR()

    def test_indian_state_codes_presence(self):
        """Verifies standard statutory Indian State and UT codes are loaded."""
        self.assertIn('TS', INDIAN_STATE_CODES)
        self.assertIn('DL', INDIAN_STATE_CODES)
        self.assertIn('MH', INDIAN_STATE_CODES)
        self.assertIn('KA', INDIAN_STATE_CODES)
        self.assertIn('TN', INDIAN_STATE_CODES)
        self.assertIn('UP', INDIAN_STATE_CODES)
        self.assertIn('GJ', INDIAN_STATE_CODES)
        self.assertIn('RJ', INDIAN_STATE_CODES)

    def test_spatial_ordering(self):
        """Tests that text segments are ordered top-to-bottom, left-to-right."""
        # Line 1: '08' at x=160, 'TS' at x=50, y=45
        # Line 2: 'EW9354' at x=100, y=90
        segments = [
            ([[130, 25], [190, 20], [195, 65], [135, 70]], '08', 0.92),
            ([[20, 40], [110, 30], [115, 80], [25, 90]], 'TS', 0.85),
            ([[40, 75], [190, 55], [195, 105], [45, 125]], 'EW9354', 0.88),
        ]
        ordered = self.ocr.order_text_segments(segments)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered[0]['text'], 'TS')
        self.assertEqual(ordered[1]['text'], '08')
        self.assertEqual(ordered[2]['text'], 'EW9354')

    def test_validate_standard_indian_formats(self):
        """Tests standard RTO Indian plate validation and formatting."""
        # Standard two-wheeler plate
        v1 = self.ocr.validate_and_normalize_indian_plate("TS08EW9354", 0.85)
        self.assertEqual(v1['status'], 'readable')
        self.assertEqual(v1['text'], 'TS 08 EW 9354')
        self.assertTrue(v1['is_valid_format'])

        # Four-wheeler Delhi plate
        v2 = self.ocr.validate_and_normalize_indian_plate("DL8SAB1234", 0.90)
        self.assertEqual(v2['status'], 'readable')
        self.assertEqual(v2['text'], 'DL 8 SAB 1234')

        # Maharashtra plate
        v3 = self.ocr.validate_and_normalize_indian_plate("MH12DE5678", 0.88)
        self.assertEqual(v3['status'], 'readable')
        self.assertEqual(v3['text'], 'MH 12 DE 5678')

        # Karnataka older format
        v4 = self.ocr.validate_and_normalize_indian_plate("KA011234", 0.80)
        self.assertEqual(v4['status'], 'readable')
        self.assertEqual(v4['text'], 'KA 01 1234')

    def test_bharat_series_validation(self):
        """Tests Central Government Bharat Series (BH) plate validation."""
        v = self.ocr.validate_and_normalize_indian_plate("22BH1234AA", 0.88)
        self.assertEqual(v['status'], 'readable')
        self.assertEqual(v['text'], '22 BH 1234 AA')
        self.assertTrue(v['is_valid_format'])

    def test_hologram_and_border_artifact_trimming(self):
        """Tests trimming of leading 'IND' and chromium hologram border marks."""
        # Leading IND country identifier
        v1 = self.ocr.validate_and_normalize_indian_plate("INDTS08EW9354", 0.85)
        self.assertEqual(v1['text'], 'TS 08 EW 9354')
        self.assertEqual(v1['status'], 'readable')

        # Stray border character 'B' before Telangana state code 'TS'
        v2 = self.ocr.validate_and_normalize_indian_plate("BTS08EW9354", 0.82)
        self.assertEqual(v2['text'], 'TS 08 EW 9354')
        self.assertEqual(v2['status'], 'readable')

    def test_contextual_disambiguation(self):
        """Tests contextual correction of character misreads (O/0, S/5, etc.)."""
        # 'T5' -> 'TS' (state code), 'O8' -> '08' (RTO district digits)
        v = self.ocr.validate_and_normalize_indian_plate("T5O8EW9354", 0.75)
        self.assertEqual(v['text'], 'TS 08 EW 9354')
        self.assertEqual(v['status'], 'readable')

    def test_unreadable_and_low_confidence_filtering(self):
        """Guarantees partial/noisy characters are never hallucinated as complete numbers."""
        # Broken 4-char fragment with low confidence
        v_low = self.ocr.validate_and_normalize_indian_plate("F4AS", 0.0014)
        self.assertEqual(v_low['status'], 'unreadable')
        self.assertEqual(v_low['text'], 'UNREADABLE')

        # Empty / whitespace
        v_empty = self.ocr.validate_and_normalize_indian_plate("", 0.0)
        self.assertEqual(v_empty['status'], 'unreadable')
        self.assertEqual(v_empty['text'], 'UNREADABLE')

    def test_locate_plate_candidates(self):
        """Tests candidate plate localization for motorcycles and four-wheelers."""
        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        # Add a simulated white rectangle for the plate
        cv2.rectangle(dummy_frame, (80, 180), (140, 220), (255, 255, 255), -1)

        bike_box = [50, 80, 170, 260]
        candidates = self.ocr.locate_plate_candidates(dummy_frame, bike_box, vehicle_type='motorcycle')
        self.assertTrue(len(candidates) > 0)
        
        # Verify bounding box format [x1, y1, x2, y2]
        for crop, box, ctype in candidates:
            self.assertEqual(len(box), 4)
            self.assertTrue(box[2] > box[0])
            self.assertTrue(box[3] > box[1])

    def test_example_image_plate_extraction(self):
        """End-to-end regression test on the supplied example image."""
        img_path = os.path.join("uploads", "705e4171_Screenshot_2026-10-23_210437.png")
        if not os.path.exists(img_path):
            self.skipTest(f"Example image not found at {img_path}")

        frame = cv2.imread(img_path)
        self.assertIsNotNone(frame)

        # Motorcycle bounding box from YOLO detection
        bike_box = [33, 100, 93, 245]
        res = self.ocr.extract_plate_from_vehicle(frame, bike_box, vehicle_type='motorcycle')

        self.assertIsNotNone(res)
        self.assertIn(res['status'], ['readable', 'partial'])
        # Plate must NOT be the broken hallucination 'F4AS'
        self.assertNotEqual(res['normalized_text'], 'F4AS')
        # Registration must contain Telangana state code 'TS' and district '08'
        self.assertIn('TS', res['normalized_text'])
        self.assertIn('08', res['normalized_text'])
        # High confidence extraction
        self.assertGreater(res['confidence'], 0.40)
        # Plate crop must have been saved for auditing
        self.assertIsNotNone(res['crop_path'])
        self.assertTrue(os.path.exists(res['crop_path'].lstrip('/')))


if __name__ == '__main__':
    unittest.main()
