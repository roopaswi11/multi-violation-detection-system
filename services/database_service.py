import sqlite3
import os
import json
from datetime import datetime
from config import Config

class DatabaseService:
    def __init__(self, db_path=None, mongo_uri=None, mongo_db_name=None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.mongo_uri = mongo_uri or getattr(Config, 'MONGO_URI', None)
        self.mongo_db_name = mongo_db_name or getattr(Config, 'MONGO_DB_NAME', 'traffic_violation_db')
        self.mongo_client = None
        self.mongo_db = None
        self._init_mongo()
        self.init_db()

    def _init_mongo(self):
        """Attempts to initialize MongoDB connection if configured or available."""
        if self.mongo_uri:
            try:
                import pymongo
                self.mongo_client = pymongo.MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2000)
                # Verify connection
                self.mongo_client.admin.command('ping')
                self.mongo_db = self.mongo_client[self.mongo_db_name]
                print(f"[DatabaseService] Connected to MongoDB database: {self.mongo_db_name}")
                try:
                    self.mongo_db.fines.create_index("fine_id", unique=True)
                    self.mongo_db.fines.create_index("violation_id", unique=True)
                    self.mongo_db.fines.create_index("job_id")
                except Exception:
                    pass
            except Exception as e:
                print(f"[DatabaseService] MongoDB connection not active ({e}). Operating in SQLite mode.")
                self.mongo_client = None
                self.mongo_db = None

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
                    rider_id TEXT,
                    plate_text TEXT,
                    plate_confidence REAL,
                    evidence_path TEXT,
                    review_status TEXT DEFAULT 'pending',
                    reviewer_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            ''')

            # Fines table (Challans generated for confirmed violations)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fine_id TEXT UNIQUE NOT NULL,
                    violation_id INTEGER NOT NULL UNIQUE,
                    job_id TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    registration_number TEXT NOT NULL,
                    fine_amount REAL NOT NULL,
                    currency TEXT DEFAULT 'INR',
                    legal_provision TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    fine_rule_version TEXT NOT NULL,
                    violation_timestamp TEXT,
                    evidence_path TEXT,
                    status TEXT DEFAULT 'ISSUED',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (violation_id) REFERENCES violations (id),
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fines_job_id ON fines (job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fines_violation_id ON fines (violation_id)')
            
            # Migrate table if rider_id column is missing
            cursor.execute("PRAGMA table_info(violations)")
            columns = [row['name'] for row in cursor.fetchall()]
            if 'rider_id' not in columns:
                try:
                    cursor.execute("ALTER TABLE violations ADD COLUMN rider_id TEXT")
                except Exception:
                    pass

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

        if self.mongo_db is not None:
            try:
                self.mongo_db.jobs.insert_one({
                    '_id': job_id,
                    'job_id': job_id,
                    'file_type': file_type,
                    'original_filename': original_filename,
                    'stored_filename': stored_filename,
                    'status': 'processing',
                    'metadata': metadata or {},
                    'created_at': datetime.utcnow()
                })
            except Exception as e:
                print(f"[DatabaseService] Warning: MongoDB create_job write failed: {e}")

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

        if self.mongo_db is not None:
            try:
                update_fields = {
                    'output_path': output_path,
                    'total_violations': total_violations,
                    'status': status,
                    'completed_at': datetime.utcnow()
                }
                if metadata is not None:
                    update_fields['metadata'] = metadata
                self.mongo_db.jobs.update_one({'_id': job_id}, {'$set': update_fields})
            except Exception as e:
                print(f"[DatabaseService] Warning: MongoDB complete_job update failed: {e}")

    def insert_violation(self, job_id, violation_type, confidence, track_id=None,
                         frame_number=None, timestamp_sec=None, bbox=None,
                         rider_count=None, rider_id=None, plate_text=None, plate_confidence=None,
                         evidence_path=None, review_status='pending', notes=None):
        bbox_str = json.dumps(bbox) if bbox is not None else None
        conf_val = float(confidence) if confidence is not None else 0.0
        v_id = None
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations (
                    job_id, violation_type, confidence, track_id,
                    frame_number, timestamp_sec, bbox_json, rider_count, rider_id,
                    plate_text, plate_confidence, evidence_path, review_status, reviewer_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_id, violation_type, conf_val, track_id,
                frame_number, timestamp_sec, bbox_str, rider_count, rider_id,
                plate_text, plate_confidence, evidence_path, review_status, notes
            ))
            conn.commit()
            v_id = cursor.lastrowid

        if self.mongo_db is not None:
            try:
                self.mongo_db.violations.insert_one({
                    'violation_id': v_id,
                    'job_id': job_id,
                    'violation_type': violation_type,
                    'confidence': conf_val,
                    'track_id': track_id,
                    'frame_number': frame_number,
                    'timestamp_sec': timestamp_sec,
                    'bounding_box': bbox or [],
                    'rider_count': rider_count,
                    'rider_id': rider_id,
                    'plate_text': plate_text,
                    'plate_confidence': plate_confidence,
                    'evidence_path': evidence_path,
                    'review_status': review_status,
                    'notes': notes,
                    'created_at': datetime.utcnow()
                })
            except Exception as e:
                print(f"[DatabaseService] Warning: MongoDB insert_violation failed: {e}")

        return v_id

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
        query = '''
            SELECT v.*, j.original_filename, f.fine_id, f.fine_amount 
            FROM violations v 
            LEFT JOIN jobs j ON v.job_id = j.id 
            LEFT JOIN fines f ON v.id = f.violation_id 
            WHERE 1=1
        '''
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
            for j in recent_jobs:
                try:
                    j['metadata'] = json.loads(j.get('metadata_json') or '{}')
                except Exception:
                    j['metadata'] = {}

            # Fine stats
            cursor.execute('SELECT COUNT(*) as cnt, COALESCE(SUM(fine_amount), 0.0) as total_amt FROM fines')
            fine_row = cursor.fetchone()
            total_fines_issued = fine_row['cnt'] if fine_row else 0
            total_fine_amount = fine_row['total_amt'] if fine_row else 0.0
            
            return {
                'total_jobs': total_jobs,
                'total_violations': total_violations,
                'no_helmet_count': no_helmet_count,
                'triple_riding_count': triple_riding_count,
                'wrong_side_count': wrong_side_count,
                'pending_review': pending_review,
                'confirmed_review': confirmed_review,
                'disputed_review': disputed_review,
                'recent_jobs': recent_jobs,
                'total_fines_issued': total_fines_issued,
                'total_fine_amount': round(float(total_fine_amount), 2)
            }

    def create_fine(self, fine_dict):
        """
        Creates a fine (challan) record in SQLite and MongoDB with strict deduplication by violation_id.
        If a fine for the given violation_id already exists, returns the existing record.
        """
        existing = self.get_fine_by_violation(fine_dict['violation_id'])
        if existing:
            return existing

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO fines (
                    fine_id, violation_id, job_id, violation_type, registration_number,
                    fine_amount, currency, legal_provision, jurisdiction, fine_rule_version,
                    violation_timestamp, evidence_path, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fine_dict['fine_id'],
                fine_dict['violation_id'],
                fine_dict['job_id'],
                fine_dict['violation_type'],
                fine_dict.get('registration_number', 'UNREADABLE'),
                float(fine_dict['fine_amount']),
                fine_dict.get('currency', 'INR'),
                fine_dict['legal_provision'],
                fine_dict['jurisdiction'],
                fine_dict.get('fine_rule_version', '2019.1'),
                str(fine_dict.get('violation_timestamp', '')),
                fine_dict.get('evidence_path'),
                fine_dict.get('status', 'ISSUED')
            ))
            conn.commit()

        if self.mongo_db is not None:
            try:
                mongo_doc = dict(fine_dict)
                mongo_doc['_id'] = fine_dict['fine_id']
                mongo_doc['created_at'] = datetime.utcnow()
                self.mongo_db.fines.update_one(
                    {'violation_id': fine_dict['violation_id']},
                    {'$setOnInsert': mongo_doc},
                    upsert=True
                )
            except Exception as e:
                print(f"[DatabaseService] Warning: MongoDB create_fine failed: {e}")

        return self.get_fine(fine_dict['fine_id'])

    def get_fine(self, fine_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM fines WHERE fine_id = ?', (fine_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_fine_by_violation(self, violation_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM fines WHERE violation_id = ?', (violation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_fines_by_job(self, job_id):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM fines WHERE job_id = ? ORDER BY id ASC', (job_id,))
            return [dict(r) for r in cursor.fetchall()]

    def get_all_fines(self, limit=50, offset=0):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT f.*, j.original_filename, j.file_type 
                FROM fines f 
                LEFT JOIN jobs j ON f.job_id = j.id 
                ORDER BY f.created_at DESC LIMIT ? OFFSET ?
            ''', (limit, offset))
            return [dict(r) for r in cursor.fetchall()]

    def delete_fine_by_violation(self, violation_id):
        """Removes fine record when a violation is unconfirmed or disputed, keeping totals synchronized."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM fines WHERE violation_id = ?', (violation_id,))
            conn.commit()
            deleted = cursor.rowcount > 0

        if self.mongo_db is not None:
            try:
                self.mongo_db.fines.delete_many({'violation_id': violation_id})
            except Exception as e:
                print(f"[DatabaseService] Warning: MongoDB delete_fine failed: {e}")

        return deleted
