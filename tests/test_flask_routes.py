import unittest
import os
import sys
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app
from config import Config
from services.database_service import DatabaseService

class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.db = DatabaseService()

    def test_dashboard_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'TRAFFIC GUARD AI', response.data)
        self.assertIn(b'Traffic Monitoring Command Center', response.data)

    def test_upload_page_get(self):
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'New Traffic Detection', response.data)
        self.assertIn(b'ANALYZE', response.data)
        self.assertIn(b'Road Direction Settings', response.data)
        self.assertIn(b'Permitted Road Direction', response.data)
        self.assertIn(b'Automatic / Unknown', response.data)
        self.assertIn(b'Downward Flow (Top to Bottom)', response.data)
        self.assertIn(b'Upward Flow (Bottom to Top)', response.data)
        self.assertIn(b'Leftward Flow (Right to Left)', response.data)
        self.assertIn(b'Rightward Flow (Left to Right)', response.data)
        self.assertNotIn(b'Quick-Test with User Dataset Samples', response.data)
        self.assertNotIn(b'Pipeline Parameters', response.data)

    def test_history_page_get(self):
        response = self.client.get('/history')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Traffic Violation History Log', response.data)

    def test_api_stats(self):
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('total_jobs', data)
        self.assertIn('total_violations', data)
        self.assertIn('no_helmet_count', data)

    def test_invalid_file_extension_upload(self):
        data = {
            'file': (io.BytesIO(b"fake dangerous content"), 'test.exe')
        }
        response = self.client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid file format', response.data)

    def test_database_insert_and_review_workflow(self):
        # Create test job first with unique ID
        import uuid
        job_id = f"test_job_{uuid.uuid4().hex[:8]}"
        self.db.create_job(job_id, "image", "test.jpg", f"{job_id}_test.jpg")
        
        # Insert a sample violation
        v_id = self.db.insert_violation(
            job_id=job_id,
            violation_type="no_helmet",
            confidence=0.88,
            track_id=1,
            frame_number=10,
            review_status="pending",
            notes="Initial automated detection"
        )
        self.assertIsNotNone(v_id)

        # Retrieve
        v = self.db.get_violation(v_id)
        self.assertEqual(v['review_status'], 'pending')

        # Test POST review decision
        post_res = self.client.post(f'/violation/{v_id}', data={
            'review_status': 'confirmed',
            'reviewer_notes': 'Verified by officer'
        }, follow_redirects=True)
        self.assertEqual(post_res.status_code, 200)

        # Verify updated status
        v_updated = self.db.get_violation(v_id)
        self.assertEqual(v_updated['review_status'], 'confirmed')
        self.assertEqual(v_updated['reviewer_notes'], 'Verified by officer')

    def test_upload_with_permitted_direction_options(self):
        import cv2, numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, img_buf = cv2.imencode('.jpg', img)

        for direction in ['auto', 'downward', 'upward', 'leftward', 'rightward']:
            data = {
                'file': (io.BytesIO(img_buf.tobytes()), f'test_{direction}.jpg'),
                'permitted_direction': direction,
                'ajax': '1'
            }
            res = self.client.post('/upload', data=data, content_type='multipart/form-data')
            self.assertEqual(res.status_code, 200)
            json_data = res.get_json()
            self.assertTrue(json_data['success'])
            self.assertEqual(json_data['permitted_direction'], direction)
            self.assertIn('wrong_side_details', json_data)
            self.assertEqual(json_data['wrong_side_details']['selected_direction'], direction)

if __name__ == '__main__':
    unittest.main()

