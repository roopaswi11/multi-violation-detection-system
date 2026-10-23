import unittest
import os
import tempfile
import json
import sqlite3
from services.database_service import DatabaseService
from services.fine_service import FineService
from config import Config

class TestFineGeneration(unittest.TestCase):
    def setUp(self):
        # Create isolated temporary database for test runs
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        os.close(self.temp_db_fd)
        self.db = DatabaseService(db_path=self.temp_db_path)
        self.fine_service = FineService()

        # Seed test job
        self.job_id = "test_job_fine_001"
        self.db.create_job(
            job_id=self.job_id,
            file_type="image",
            original_filename="traffic_test.jpg",
            stored_filename="stored_test.jpg"
        )

    def tearDown(self):
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except Exception:
                pass

    def test_no_helmet_fine_generation(self):
        """1. Confirmed No-Helmet violation generates statutory ₹1,000 fine under Section 194D."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.95,
            review_status="confirmed",
            plate_text="DL8SAB1234",
            evidence_path="/outputs/evidence/ev_helmet.jpg"
        )
        violation = self.db.get_violation(v_id)
        
        fine = self.fine_service.generate_fine_for_violation(violation, db=self.db)
        
        self.assertIsNotNone(fine)
        self.assertTrue(fine['fine_id'].startswith("FINE-"))
        self.assertEqual(fine['violation_id'], v_id)
        self.assertEqual(fine['violation_type'], "no_helmet")
        self.assertEqual(fine['fine_amount'], 1000.0)
        self.assertEqual(fine['currency'], "INR")
        self.assertIn("194D", fine['legal_provision'])
        self.assertEqual(fine['registration_number'], "DL8SAB1234")
        self.assertEqual(fine['status'], "ISSUED")

    def test_triple_riding_fine_generation(self):
        """2. Confirmed Triple-Riding violation generates statutory ₹1,000 fine under Section 194C."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="triple_riding",
            confidence=0.92,
            rider_count=3,
            review_status="confirmed",
            plate_text="MH12DE5678",
            evidence_path="/outputs/evidence/ev_triple.jpg"
        )
        violation = self.db.get_violation(v_id)

        fine = self.fine_service.generate_fine_for_violation(violation, db=self.db)

        self.assertIsNotNone(fine)
        self.assertEqual(fine['violation_type'], "triple_riding")
        self.assertEqual(fine['fine_amount'], 1000.0)
        self.assertIn("194C", fine['legal_provision'])
        self.assertEqual(fine['registration_number'], "MH12DE5678")

    def test_wrong_direction_fine_generation_confirmed_video(self):
        """3. Confirmed Wrong-Direction violation from tracked movement generates statutory ₹5,000 fine under Section 184(f)."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="wrong_side",
            confidence=0.88,
            track_id=14,
            review_status="confirmed",
            notes="Contraflow trajectory confirmed across 12 frames (displacement: 85px)",
            evidence_path="/outputs/evidence/ev_wrong.jpg"
        )
        violation = self.db.get_violation(v_id)

        fine = self.fine_service.generate_fine_for_violation(violation, db=self.db)

        self.assertIsNotNone(fine)
        self.assertEqual(fine['violation_type'], "wrong_side")
        self.assertEqual(fine['fine_amount'], 5000.0)
        self.assertIn("184", fine['legal_provision'])
        self.assertEqual(fine['registration_number'], "UNREADABLE")

    def test_wrong_direction_rejected_on_unverified_still_image(self):
        """4. Wrong-direction on still image without verified movement or lane evidence must NOT generate a fine."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="wrong_side",
            confidence=0.40,
            review_status="insufficient_evidence",
            notes="Wrong-side status could not be verified because reliable road-direction evidence was unavailable."
        )
        violation = self.db.get_violation(v_id)

        fine = self.fine_service.generate_fine_for_violation(violation, db=self.db)

        # Must not generate fine for candidate or insufficient evidence
        self.assertIsNone(fine)

    def test_unconfirmed_violations_do_not_generate_fines(self):
        """5. Pending review, disputed, or false positive violations do NOT generate fines."""
        for status in ["pending", "disputed", "false_positive"]:
            v_id = self.db.insert_violation(
                job_id=self.job_id,
                violation_type="no_helmet",
                confidence=0.60,
                review_status=status
            )
            v = self.db.get_violation(v_id)
            fine = self.fine_service.generate_fine_for_violation(v, db=self.db)
            self.assertIsNone(fine, f"Violation with review_status '{status}' should not produce fine.")

    def test_multiple_violations_separate_fine_items_and_total(self):
        """6. Multiple distinct violations on one vehicle generate separate fine entries and correct combined total."""
        # Vehicle has 2 helmet violations (Driver + Pillion) + 1 Triple Riding
        v1 = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.96,
            rider_id="Rider 1 (Driver)",
            review_status="confirmed",
            plate_text="KA05HJ9999"
        )
        v2 = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.94,
            rider_id="Rider 2 (Pillion 1)",
            review_status="confirmed",
            plate_text="KA05HJ9999"
        )
        v3 = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="triple_riding",
            confidence=0.91,
            rider_count=3,
            review_status="confirmed",
            plate_text="KA05HJ9999"
        )

        fines, total_amount = self.fine_service.process_job_fines(self.job_id, db=self.db)

        self.assertEqual(len(fines), 3)
        self.assertEqual(total_amount, 3000.0) # 1000 + 1000 + 1000

        # Verify individual items are preserved with unique fine IDs and respective violation IDs
        fine_violation_ids = [f['violation_id'] for f in fines]
        self.assertEqual(set(fine_violation_ids), {v1, v2, v3})
        fine_amounts = [f['fine_amount'] for f in fines]
        self.assertEqual(fine_amounts, [1000.0, 1000.0, 1000.0])

    def test_unreadable_number_plate(self):
        """7. Unreadable or absent number plate records 'UNREADABLE' without fabricating plate numbers."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.89,
            review_status="confirmed",
            plate_text=None
        )
        violation = self.db.get_violation(v_id)
        fine = self.fine_service.generate_fine_for_violation(violation, db=self.db)
        
        self.assertIsNotNone(fine)
        self.assertEqual(fine['registration_number'], "UNREADABLE")

    def test_missing_fine_configuration_reports_required(self):
        """8. Unconfigured violation type reports 'FINE_CONFIGURATION_REQUIRED' without crashing or fabricating amount."""
        v = {
            'id': 9999,
            'job_id': self.job_id,
            'violation_type': 'unsupported_drone_flyby',
            'review_status': 'confirmed'
        }
        res = self.fine_service.generate_fine_for_violation(v, db=self.db)
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get('status'), 'FINE_CONFIGURATION_REQUIRED')

    def test_duplicate_processing_idempotency(self):
        """9. Reprocessing an image or re-running fine generation reuses existing fines without creating duplicates."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.95,
            review_status="confirmed"
        )
        v = self.db.get_violation(v_id)

        # First run
        fine1 = self.fine_service.generate_fine_for_violation(v, db=self.db)
        self.assertIsNotNone(fine1)

        # Reprocess / Second run on same violation
        fine2 = self.fine_service.generate_fine_for_violation(v, db=self.db)
        self.assertIsNotNone(fine2)

        # Must have same fine_id and not produce duplicate database records
        self.assertEqual(fine1['fine_id'], fine2['fine_id'])
        
        all_fines_for_job = self.db.get_fines_by_job(self.job_id)
        self.assertEqual(len(all_fines_for_job), 1)

    def test_dashboard_stats_include_fines(self):
        """10. Dashboard stats accurately reflect total fines issued and cumulative fine amount."""
        self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.95,
            review_status="confirmed"
        )
        self.fine_service.process_job_fines(self.job_id, db=self.db)

        stats = self.db.get_dashboard_stats()
        self.assertIn('total_fines_issued', stats)
        self.assertIn('total_fine_amount', stats)
        self.assertEqual(stats['total_fines_issued'], 1)
        self.assertEqual(stats['total_fine_amount'], 1000.0)

    def test_zero_confirmed_violations_total_fine_zero(self):
        """11. Zero confirmed violations returns total_fine_amount == 0.0 and 'TOTAL FINE: ₹0'."""
        # 1 unconfirmed pending violation
        self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.45,
            review_status="pending"
        )
        summary = self.fine_service.get_job_fine_summary(self.job_id, db=self.db)
        self.assertEqual(summary['confirmed_violations_count'], 0)
        self.assertEqual(summary['total_fines'], 0)
        self.assertEqual(summary['total_fine_amount'], 0.0)
        self.assertEqual(summary['display_status'], "TOTAL FINE: ₹0")
        self.assertFalse(summary['is_incomplete'])

    def test_single_confirmed_violation_total(self):
        """12. Single confirmed violation calculates exact fine amount and breakdown."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.95,
            review_status="confirmed",
            plate_text="AP09CD1111"
        )
        summary = self.fine_service.get_job_fine_summary(self.job_id, db=self.db)
        self.assertEqual(summary['confirmed_violations_count'], 1)
        self.assertEqual(summary['total_fine_amount'], 1000.0)
        self.assertEqual(summary['display_status'], "TOTAL FINE: ₹1,000")
        self.assertEqual(summary['primary_plate'], "AP09CD1111")
        self.assertFalse(summary['is_incomplete'])
        self.assertEqual(len(summary['breakdown']), 1)
        self.assertEqual(summary['breakdown'][0]['violation_type'], 'no_helmet')
        self.assertEqual(summary['breakdown'][0]['count'], 1)
        self.assertEqual(summary['breakdown'][0]['subtotal'], 1000.0)

    def test_multiple_mixed_violations_breakdown_and_total(self):
        """13. Multiple mixed violations (2 no-helmet + 1 triple-riding) calculates combined total and separate breakdowns."""
        self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.94,
            rider_id="Rider 1",
            review_status="confirmed",
            plate_text="TS07AB4321"
        )
        self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.91,
            rider_id="Rider 2",
            review_status="confirmed",
            plate_text="TS07AB4321"
        )
        self.db.insert_violation(
            job_id=self.job_id,
            violation_type="triple_riding",
            confidence=0.93,
            rider_count=3,
            review_status="confirmed",
            plate_text="TS07AB4321"
        )
        summary = self.fine_service.get_job_fine_summary(self.job_id, db=self.db)
        self.assertEqual(summary['confirmed_violations_count'], 3)
        self.assertEqual(summary['total_fine_amount'], 3000.0) # 2000 (no-helmet) + 1000 (triple-riding)
        self.assertEqual(summary['display_status'], "TOTAL FINE: ₹3,000")
        self.assertFalse(summary['is_incomplete'])

        breakdown_by_type = {b['violation_type']: b for b in summary['breakdown']}
        self.assertEqual(breakdown_by_type['no_helmet']['count'], 2)
        self.assertEqual(breakdown_by_type['no_helmet']['subtotal'], 2000.0)
        self.assertEqual(breakdown_by_type['triple_riding']['count'], 1)
        self.assertEqual(breakdown_by_type['triple_riding']['subtotal'], 1000.0)

    def test_unconfigured_rule_marks_total_incomplete(self):
        """14. Unconfigured fine rule sets total_fine_amount to None and marks is_incomplete == True."""
        v = {
            'id': 8888,
            'job_id': self.job_id,
            'violation_type': 'unconfigured_speeding_test',
            'review_status': 'confirmed'
        }
        summary = self.fine_service.get_job_fine_summary(self.job_id, violations=[v], db=self.db)
        self.assertEqual(summary['confirmed_violations_count'], 1)
        self.assertTrue(summary['is_incomplete'])
        self.assertIsNone(summary['total_fine_amount'])
        self.assertEqual(summary['display_status'], "TOTAL FINE: INCOMPLETE (Fine amount unavailable)")
        self.assertTrue(summary['breakdown'][0]['has_unavailable'])

    def test_recalculation_on_violation_status_change(self):
        """15. Changing violation review status from confirmed to disputed removes fine and recalculates total to ₹0."""
        v_id = self.db.insert_violation(
            job_id=self.job_id,
            violation_type="no_helmet",
            confidence=0.95,
            review_status="confirmed"
        )
        summary1 = self.fine_service.get_job_fine_summary(self.job_id, db=self.db)
        self.assertEqual(summary1['total_fine_amount'], 1000.0)

        # Admin disputes the violation
        self.db.update_violation_review(v_id, "disputed", "False positive reflection on closer inspection")
        self.db.delete_fine_by_violation(v_id)

        summary2 = self.fine_service.get_job_fine_summary(self.job_id, db=self.db)
        self.assertEqual(summary2['confirmed_violations_count'], 0)
        self.assertEqual(summary2['total_fine_amount'], 0.0)
        self.assertEqual(summary2['display_status'], "TOTAL FINE: ₹0")

if __name__ == '__main__':
    unittest.main()
