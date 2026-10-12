import sqlite3
import os
import json
from datetime import datetime
from config import Config

class DatabaseService:
    def __init__(self, db_path=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    file_type TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    output_path TEXT,
                    status TEXT NOT NULL,
                    total_violations INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # Violations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    track_id INTEGER,
                    frame_number INTEGER,
                    timestamp_sec REAL,
                    bbox_json TEXT,
                    rider_count INTEGER,
                    plate_text TEXT,
                    plate_confidence REAL,
                    evidence_path TEXT,
                    review_status TEXT DEFAULT 'pending',
                    reviewer_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            ''')
            
            conn.commit()

    def create_job(self, job_id, file_type, original_filename, stored_filename, metadata=None):
        metadata_str = json.dumps(metadata or {})
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO jobs (id, file_type, original_filename, stored_filename, status, metadata_json)
                VALUES (?, ?, ?, ?, 'processing', ?)
            ''', (job_id, file_type, original_filename, stored_filename, metadata_str))
            conn.commit()

    def complete_job(self, job_id, output_path, total_violations, status='completed', metadata=None):
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if metadata is not None:
                cursor.execute('''
                    UPDATE jobs 
                    SET output_path = ?, total_violations = ?, status = ?, completed_at = ?, metadata_json = ?
                    WHERE id = ?
                ''', (output_path, total_violations, status, now, json.dumps(metadata), job_id))
            else:
                cursor.execute('''
                    UPDATE jobs 
                    SET output_path = ?, total_violations = ?, status = ?, completed_at = ?
                    WHERE id = ?
                ''', (output_path, total_violations, status, now, job_id))
            conn.commit()

    def insert_violation(self, job_id, violation_type, confidence, track_id=None,
                         frame_number=None, timestamp_sec=None, bbox=None,
                         rider_count=None, plate_text=None, plate_confidence=None,
                         evidence_path=None, review_status='pending', notes=None):
        bbox_str = json.dumps(bbox) if bbox is not None else None
        conf_val = float(confidence) if confidence is not None else 0.0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations (
                    job_id, violation_type, confidence, track_id,
                    frame_number, timestamp_sec, bbox_json, rider_count,
                    plate_text, plate_confidence, evidence_path, review_status, reviewer_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id, violation_type, conf_val, track_id,
                frame_number, timestamp_sec, bbox_str, rider_count,
                plate_text, plate_confidence, evidence_path, review_status, notes
            ))
            conn.commit()
            return cursor.lastrowid

    def get_job(self, job_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                job_dict = dict(row)
                job_dict['metadata'] = json.loads(job_dict['metadata_json'] or '{}')
                return job_dict
            return None

    def get_jobs(self, limit=20):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?', (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def get_violations_by_job(self, job_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM violations WHERE job_id = ? ORDER BY frame_number ASC, id ASC', (job_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item['bbox'] = json.loads(item['bbox_json']) if item['bbox_json'] else []
                if item.get('review_status') == 'insufficient_evidence' and item.get('confidence') == 0.0:
                    item['confidence'] = None
                results.append(item)
            return results

    def get_violations(self, violation_type=None, review_status=None, limit=50, offset=0):
        query = 'SELECT v.*, j.original_filename FROM violations v LEFT JOIN jobs j ON v.job_id = j.id WHERE 1=1'
        params = []
        if violation_type:
            query += ' AND v.violation_type = ?'
            params.append(violation_type)
        if review_status:
            query += ' AND v.review_status = ?'
            params.append(review_status)
        query += ' ORDER BY v.created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item['bbox'] = json.loads(item['bbox_json']) if item['bbox_json'] else []
                results.append(item)
            return results

    def get_violation(self, violation_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT v.*, j.original_filename, j.file_type 
                FROM violations v 
                LEFT JOIN jobs j ON v.job_id = j.id 
                WHERE v.id = ?
            ''', (violation_id,))
            row = cursor.fetchone()
            if row:
                item = dict(row)
                item['bbox'] = json.loads(item['bbox_json']) if item['bbox_json'] else []
                return item
            return None

    def update_violation_review(self, violation_id, review_status, notes=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE violations 
                SET review_status = ?, reviewer_notes = ? 
                WHERE id = ?
            ''', (review_status, notes, violation_id))
            conn.commit()
            return cursor.rowcount > 0

    def get_dashboard_stats(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) as cnt FROM jobs')
            total_jobs = cursor.fetchone()['cnt']
            
            cursor.execute('SELECT COUNT(*) as cnt FROM violations')
            total_violations = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE violation_type = 'no_helmet'")
            no_helmet_count = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE violation_type = 'triple_riding'")
            triple_riding_count = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE violation_type = 'wrong_side'")
            wrong_side_count = cursor.fetchone()['cnt']
            
            # Review counts
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE review_status = 'pending'")
            pending_review = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE review_status = 'confirmed'")
            confirmed_review = cursor.fetchone()['cnt']
            
            cursor.execute("SELECT COUNT(*) as cnt FROM violations WHERE review_status IN ('disputed', 'false_positive')")
            disputed_review = cursor.fetchone()['cnt']
            
            # Recent jobs
            cursor.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 5')
            recent_jobs = [dict(r) for r in cursor.fetchall()]
            
            return {
                'total_jobs': total_jobs,
                'total_violations': total_violations,
                'no_helmet_count': no_helmet_count,
                'triple_riding_count': triple_riding_count,
                'wrong_side_count': wrong_side_count,
                'pending_review': pending_review,
                'confirmed_review': confirmed_review,
                'disputed_review': disputed_review,
                'recent_jobs': recent_jobs
            }
