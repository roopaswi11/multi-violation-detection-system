import os
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file, flash
from werkzeug.utils import secure_filename

from config import Config
from services.database_service import DatabaseService
from services.detection_service import DetectionService
from services.video_processor import VideoProcessor
from services.fine_service import FineService

app = Flask(__name__)
app.config.from_object(Config)

# Initialize database & fine service
db = DatabaseService()
fine_service = FineService()

# Global detector instance (lazy-loaded or singleton)
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = DetectionService()
    return detector

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_EXTENSIONS

# ----------------- Static File Handlers (Secure) -----------------
@app.route('/outputs/<path:subpath>')
def serve_output(subpath):
    return send_from_directory(Config.OUTPUTS_DIR, subpath)

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(Config.UPLOADS_DIR, filename)

@app.route('/data/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory(Config.SAMPLES_DIR, filename)

# ----------------- Dashboard Pages -----------------
@app.route('/')
def dashboard():
    stats = db.get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.form.get('ajax') == '1' or request.accept_mimetypes.best == 'application/json'
    file = request.files.get('file')

    if not file or not file.filename:
        msg = "Please select an image or video file to analyze."
        if is_ajax:
            return jsonify({'error': msg}), 400
        flash(msg, "warning")
        return redirect(url_for('upload'))

    if not allowed_file(file.filename):
        msg = "Invalid file format. Supported: JPG, JPEG, PNG, WEBP, MP4, AVI, MOV, WEBM."
        if is_ajax:
            return jsonify({'error': msg}), 400
        flash(msg, "danger")
        return redirect(url_for('upload'))

    job_id = str(uuid.uuid4())[:8]
    orig_filename = secure_filename(file.filename)
    ext = orig_filename.rsplit('.', 1)[1].lower()
    stored_filename = f"{job_id}_{orig_filename}"
    target_path = os.path.join(Config.UPLOADS_DIR, stored_filename)
    file.save(target_path)
    file_type = 'image' if ext in Config.ALLOWED_IMAGE_EXTENSIONS else 'video'

    # Validate submitted permitted road direction
    VALID_DIRECTIONS = {
        'auto': 'auto',
        'unknown': 'auto',
        'downward': 'downward',
        'upward': 'upward',
        'leftward': 'leftward',
        'right_to_left': 'leftward',
        'rightward': 'rightward',
        'left_to_right': 'rightward'
    }
    raw_dir = request.form.get('permitted_direction', 'auto')
    direction = VALID_DIRECTIONS.get(raw_dir.strip().lower() if raw_dir else 'auto', 'auto')
    conf_thresh = Config.CONFIDENCE_THRESHOLD
    enable_vit = False

    # Record job in database
    db.create_job(job_id, file_type, orig_filename, stored_filename, metadata={
        'permitted_direction': direction,
        'conf_threshold': conf_thresh,
        'enable_vit': enable_vit
    })

    # Execute processing pipeline
    try:
        det = get_detector()
        processor = VideoProcessor(
            detector=det,
            permitted_direction=direction,
            enable_vit=enable_vit,
            conf_threshold=conf_thresh
        )

        if file_type == 'image':
            out_path, violations, detections = processor.process_image(target_path)
            diag = processor.get_diagnostics()
            confirmed_count = len([v for v in violations if v.get('review_status') == 'confirmed'])
            pending_count = len([v for v in violations if v.get('review_status') == 'pending'])
            candidate_count = len([v for v in violations if v.get('review_status') == 'insufficient_evidence'])
            for v in violations:
                v_id = db.insert_violation(
                    job_id=job_id,
                    violation_type=v['violation_type'],
                    confidence=v.get('confidence'),
                    track_id=v.get('track_id'),
                    frame_number=v.get('frame_number', 0),
                    timestamp_sec=v.get('timestamp_sec', 0.0),
                    bbox=v.get('bounding_box'),
                    rider_count=v.get('rider_count'),
                    rider_id=v.get('rider_id'),
                    plate_text=v.get('plate_text'),
                    plate_confidence=v.get('plate_confidence'),
                    evidence_path=v.get('evidence_path'),
                    review_status=v.get('review_status', 'pending'),
                    notes=v.get('reason')
                )
                v['id'] = v_id

            # Automatically calculate fines and fine summary for confirmed violations
            fine_summary = fine_service.get_job_fine_summary(job_id, violations=violations, db=db)
            fines = fine_summary['fines']
            total_fine_amount = fine_summary['total_fine_amount']

            rel_output = "/outputs/images/" + os.path.basename(out_path)
            meta = {
                'permitted_direction': direction,
                'permitted_direction_label': diag.get('permitted_direction_label', direction),
                'conf_threshold': conf_thresh,
                'enable_vit': enable_vit,
                'total_detections': len(detections),
                'confirmed_violations': confirmed_count,
                'pending_violations': pending_count,
                'candidate_violations': candidate_count,
                'total_fines': len(fines),
                'total_fine_amount': total_fine_amount,
                'fine_summary': fine_summary,
                'detection_status': diag.get('status', 'violations_found' if len(violations) > 0 else ('compliant' if len(detections) > 0 else 'no_objects_detected')),
                'diagnostic_message': diag.get('diagnostic_message', ''),
                'wrong_side_status': diag.get('wrong_side_status', 'Wrong-side status could not be verified because reliable road-direction evidence was unavailable.'),
                'wrong_side_details': diag.get('wrong_side_details', {}),
                'class_summary': diag.get('class_summary', {}),
                'primary_model': os.path.basename(getattr(det, 'model_path', 'default')),
                'base_model': os.path.basename(getattr(det, 'base_model_path', 'none')) if getattr(det, 'base_model_path', None) else None
            }
            db.complete_job(job_id, rel_output, len(violations), status='completed', metadata=meta)

        else: # video
            out_path, violations, total_frames = processor.process_video(target_path)
            diag = processor.get_diagnostics()
            confirmed_count = len([v for v in violations if v.get('review_status') == 'confirmed'])
            pending_count = len([v for v in violations if v.get('review_status') == 'pending'])
            candidate_count = len([v for v in violations if v.get('review_status') == 'insufficient_evidence'])
            for v in violations:
                v_id = db.insert_violation(
                    job_id=job_id,
                    violation_type=v['violation_type'],
                    confidence=v.get('confidence'),
                    track_id=v.get('track_id'),
                    frame_number=v.get('frame_number', 0),
                    timestamp_sec=v.get('timestamp_sec', 0.0),
                    bbox=v.get('bounding_box'),
                    rider_count=v.get('rider_count'),
                    rider_id=v.get('rider_id'),
                    plate_text=v.get('plate_text'),
                    plate_confidence=v.get('plate_confidence'),
                    evidence_path=v.get('evidence_path'),
                    review_status=v.get('review_status', 'pending'),
                    notes=v.get('reason')
                )
                v['id'] = v_id

            # Automatically calculate fines and fine summary for confirmed violations
            fine_summary = fine_service.get_job_fine_summary(job_id, violations=violations, db=db)
            fines = fine_summary['fines']
            total_fine_amount = fine_summary['total_fine_amount']

            rel_output = "/outputs/videos/" + os.path.basename(out_path)
            meta = {
                'permitted_direction': direction,
                'permitted_direction_label': diag.get('permitted_direction_label', direction),
                'conf_threshold': conf_thresh,
                'enable_vit': enable_vit,
                'total_frames': total_frames,
                'confirmed_violations': confirmed_count,
                'pending_violations': pending_count,
                'candidate_violations': candidate_count,
                'total_fines': len(fines),
                'total_fine_amount': total_fine_amount,
                'fine_summary': fine_summary,
                'detection_status': 'violations_found' if len(violations) > 0 else 'compliant',
                'diagnostic_message': f"Processed {total_frames} video frames. Identified {len(violations)} distinct traffic violations." if len(violations) > 0 else f"Processed {total_frames} video frames. No traffic violations observed.",
                'wrong_side_status': diag.get('wrong_side_status', 'Wrong-side status could not be verified because reliable road-direction evidence was unavailable.'),
                'wrong_side_details': diag.get('wrong_side_details', {}),
                'primary_model': os.path.basename(getattr(det, 'model_path', 'default'))
            }
            db.complete_job(job_id, rel_output, len(violations), status='completed', metadata=meta)

        if is_ajax:
            return jsonify({
                'success': True,
                'job_id': job_id,
                'file_type': file_type,
                'original_filename': orig_filename,
                'output_path': rel_output,
                'download_url': url_for('download', job_id=job_id),
                'results_url': url_for('results', job_id=job_id),
                'total_violations': len(violations),
                'confirmed_violations': confirmed_count,
                'candidate_violations': candidate_count,
                'total_fines': len(fines),
                'total_fine_amount': total_fine_amount,
                'fine_summary': fine_summary,
                'fines': fines,
                'violations': violations,
                'total_detections': len(detections) if file_type == 'image' else None,
                'diagnostics': diag,
                'permitted_direction': direction,
                'permitted_direction_label': diag.get('permitted_direction_label', direction),
                'wrong_side_status': diag.get('wrong_side_status', 'Wrong-side status could not be verified because reliable road-direction evidence was unavailable.'),
                'wrong_side_details': diag.get('wrong_side_details', {}),
                'status': 'completed'
            })

        return redirect(url_for('results', job_id=job_id))

    except Exception as e:
        import traceback
        traceback.print_exc()
        db.complete_job(job_id, None, 0, status=f"failed: {str(e)}")
        if is_ajax:
            return jsonify({'error': f"Processing failed: {str(e)}"}), 500
        flash(f"Processing failed: {str(e)}", "danger")
        return redirect(url_for('upload'))

@app.route('/results/<job_id>')
def results(job_id):
    job = db.get_job(job_id)
    if not job:
        flash("Job not found.", "danger")
        return redirect(url_for('dashboard'))
    violations = db.get_violations_by_job(job_id)
    fine_summary = fine_service.get_job_fine_summary(job_id, violations=violations, db=db)
    fines = fine_summary['fines']
    total_fine_amount = fine_summary['total_fine_amount']
    return render_template(
        'results.html',
        job=job,
        violations=violations,
        fines=fines,
        fine_summary=fine_summary,
        total_fine_amount=total_fine_amount
    )

@app.route('/history')
def history():
    vtype = request.args.get('type')
    status = request.args.get('status')
    violations = db.get_violations(violation_type=vtype, review_status=status, limit=100)
    return render_template('history.html', violations=violations, current_type=vtype, current_status=status)

@app.route('/violation/<int:violation_id>', methods=['GET', 'POST'])
def violation_details(violation_id):
    if request.method == 'POST':
        new_status = request.form.get('review_status', 'pending')
        notes = request.form.get('reviewer_notes', '')
        db.update_violation_review(violation_id, new_status, notes)
        v_rec = db.get_violation(violation_id)
        if new_status == 'confirmed':
            if v_rec:
                fine_service.generate_fine_for_violation(v_rec, db=db)
        else:
            db.delete_fine_by_violation(violation_id)
        flash("Violation review status updated successfully.", "success")
        return redirect(url_for('violation_details', violation_id=violation_id))

    violation = db.get_violation(violation_id)
    if not violation:
        flash("Violation record not found.", "danger")
        return redirect(url_for('history'))
    fine = db.get_fine_by_violation(violation_id)
    if not fine and violation.get('review_status') == 'confirmed':
        fine = fine_service.generate_fine_for_violation(violation, db=db)
    return render_template('violation_details.html', violation=violation, fine=fine)

@app.route('/download/<job_id>')
def download(job_id):
    job = db.get_job(job_id)
    if not job or not job.get('output_path'):
        flash("Output file not available for download.", "danger")
        return redirect(url_for('dashboard'))
    
    # Resolve local path
    rel_path = job['output_path'].lstrip('/')
    abs_path = os.path.join(Config.BASE_DIR, rel_path.replace('/', os.sep))
    if not os.path.exists(abs_path):
        flash("File not found on server.", "danger")
        return redirect(url_for('dashboard'))
    return send_file(abs_path, as_attachment=True)

# ----------------- JSON APIs -----------------
@app.route('/api/stats')
def api_stats():
    stats = db.get_dashboard_stats()
    return jsonify(stats)

@app.route('/api/job/<job_id>/status')
def api_job_status(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

@app.route('/api/job/<job_id>/fines')
def api_job_fines(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    summary = fine_service.get_job_fine_summary(job_id, db=db)
    return jsonify(summary)

@app.route('/api/fine/<fine_id>')
def api_fine_details(fine_id):
    fine = db.get_fine(fine_id)
    if not fine:
        return jsonify({'error': 'Fine record not found'}), 404
    return jsonify(fine)

@app.route('/api/fine-schedule')
def api_fine_schedule():
    return jsonify(fine_service.schedule)

@app.route('/api/fine-schedule/update', methods=['POST'])
def api_fine_schedule_update():
    data = request.get_json() or {}
    vtype = data.get('violation_type')
    if not vtype:
        return jsonify({'error': 'violation_type is required'}), 400
    success = fine_service.update_rule(
        violation_type=vtype,
        fine_amount=data.get('fine_amount'),
        legal_provision=data.get('legal_provision'),
        jurisdiction=data.get('jurisdiction'),
        active=data.get('active'),
        official_source=data.get('official_source')
    )
    if success:
        return jsonify({'success': True, 'message': f"Fine rule for {vtype} updated successfully.", 'schedule': fine_service.schedule})
    return jsonify({'error': 'Failed to persist fine schedule update'}), 500

if __name__ == '__main__':
    # Initialize detector and ensure database is ready
    get_detector()
    print("\n* Traffic Violation Detection Flask Web Dashboard started at http://127.0.0.1:5000\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
