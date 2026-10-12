# AI-Based Traffic Violation Detection System
### Using Ultralytics YOLO11, Vision Transformer (ViT), ByteTrack, EasyOCR, and Flask

A complete, production-grade, end-to-end traffic enforcement and surveillance web system built with Python, PyTorch, Ultralytics YOLO11, ByteTrack multi-object tracking, and Flask.

---

## 1. Project Overview & Three Workspace Datasets

An exhaustive audit of all three datasets provided in the workspace (`multi-voilation system/`) was conducted. All 12,518 images across the three datasets use the standard YOLO bounding box format with **100% image-label pairing and zero corrupt files**:

1. **Dataset 1 — IMVTDataset (`IMVTDataset/IMVTDataset/`):**
   - **Total Images:** 6,337 (Train: 5,785 | Val: 346 | Test: 206)
   - **Classes (10):** `b`, `driver`, `earphone`, `helmet`, `license_plate`, `mobile_phone`, `no_helmet`, `object`, `pillion`, `two_wheeler`
   - **Capabilities:** Detailed rider/pillion separation, distracted driving detection, motorcycle and plate localization.

2. **Dataset 2 — Traffic Violation Dataset (`traffic dataset.v4i.yolov11/`):**
   - **Total Images:** 5,675 (Train: 5,238 | Val: 219 | Test: 218)
   - **Classes (4):** `helmet`, `no helmet`, `number plate`, `triple riding`
   - **Capabilities:** Direct multi-rider triple riding detection, dense helmet compliance, number plate localization.

3. **Dataset 3 — Wrong Direction Dataset (`Wrong Direction.v4i.yolov11/`):**
   - **Total Images:** 506 (Train: 353 | Val: 144 | Test: 9)
   - **Classes (2):** `Right-Direction`, `Wrong-Direction`
   - **Capabilities:** Static camera lane contraflow detection, complementing video trajectory analysis.

Full audit documentation is available in [`dataset_audit_report.md`](file:///c:/Users/roopa/Documents/multi-voilation%20system/dataset_audit_report.md).

---

## 2. Trained Models & Performance Benchmarks

Transfer learning fine-tuning was performed using Ultralytics YOLO11n. Both trained model checkpoints are integrated into the application:

| Model | Task | Checkpoint | Validation Set | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| **Traffic Multi-Violation** | Helmet, Triple-Riding, Plates | `models/best.pt` | 219 images | 58.9% | 62.1% | **58.1%** | 36.1% |
| **Wrong Direction Classifier**| Road Contraflow Classification| `models/wrong_direction_best.pt` | 144 images | 89.7% | 85.2% | **92.9%** | 68.3% |
| *Baseline Pretrained* | *Un-fine-tuned YOLO11n on Direction* | `yolo11n.pt` | 144 images | 0.9% | 1.8% | *1.26%* | *0.6%* |

### Per-Class Highlights:
- **Number Plate Localization:** **98.6% mAP@0.5**
- **Triple-Riding Detection:** **78.6% mAP@0.5**
- **Right-Direction Traffic:** **96.1% mAP@0.5**
- **Wrong-Direction Contraflow:** **89.7% mAP@0.5**

Full training details and loss curves are documented in [`model_training_report.md`](file:///c:/Users/roopa/Documents/multi-voilation%20system/model_training_report.md).

---

## 3. Violation Detection Modules

- **A. No-Helmet Detection:** Dual-strategy pipeline:
  1. Direct detection from the fine-tuned model (`models/best.pt`).
  2. Spatial association: rider bounding box top 35% crop associated with helmet/no-helmet detections, with optional Vision Transformer (ViT) toggle.
- **B. Triple-Riding Detection:**
  1. Direct high-confidence detection via the trained multi-violation model (`78.6% mAP50`).
  2. Spatial rider-to-motorcycle IoU association counting riders ($\ge 3$), debounced across consecutive video frames.
- **C. Wrong-Side Riding Detection:**
  1. **Video:** ByteTrack multi-object tracking + trajectory vector analysis over consecutive frames against configurable road direction (`downward`, `upward`, `left_to_right`, `right_to_left`) with stationary vehicle jitter suppression.
  2. **Static Images:** Inference via the specialized `models/wrong_direction_best.pt` model (`92.9% mAP50`).
- **D. Number Plate Detection & OCR:**
  1. High-accuracy plate localization (`98.6% mAP50`).
  2. Automatic cropping with Bilateral filtering, CLAHE contrast enhancement, and EasyOCR text extraction.

---

## 4. System Architecture

```text
IMAGE / VIDEO UPLOAD (Flask Web Dashboard / CLI)
                     │
                     ▼
       Pre-processing & Validation
                     │
                     ▼
          Dual YOLO11 Detection
   ├── models/best.pt (Helmets, Plates, Triple Riding)
   └── models/wrong_direction_best.pt (Contraflow)
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
   [A. No-Helmet]  [B. Triple]  [C. Wrong-Side]
   Rider-Bike      Spatial      ByteTrack IDs
   Association     Counting     Trajectories
        │          (>2 Riders)  Direction Vector
   Head Region       │               │
   ViT Classifier    │               │
        │            │               │
        └────────────┼───────────────┘
                     │
                     ▼
        Number Plate Extraction & OCR
          (EasyOCR Engine + CLAHE)
                     │
                     ▼
            Violation Aggregator
   (Deduplication by Track ID, Evidence Cropping)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
Annotated Output Media      SQLite Database (traffic.db)
(Boxes, Trails, Badges)    (Jobs, Violations, Officer Audit)
        │                         │
        └────────────┬────────────┘
                     ▼
           Flask Web Dashboard
 (Live Charts, History Log, Review & Dispute Actions)
```

---

## 5. Directory Structure

```text
multi-voilation system/
├── app.py                     # Main Flask web application
├── config.py                  # System configuration (paths, thresholds, directions)
├── detect.py                  # Standalone CLI detection tool
├── plate_ocr.py               # Standalone number plate OCR extraction tool
├── violation_logic.py         # Standalone violation decision logic module
├── train.py                   # Generic training harness
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation
├── dataset_audit_report.md    # Phase 1 exhaustive dataset audit report
├── model_training_report.md   # Phase 5 model training and benchmark report
│
├── configs/                   # YOLO training dataset YAML configurations
│   ├── traffic_violation.yaml
│   ├── traffic_violation_train.yaml
│   ├── wrong_direction.yaml
│   └── imvt_dataset.yaml
│
├── scripts/                   # Data audit, preparation & training scripts
│   ├── audit_datasets.py      # Dataset inspection and integrity verification
│   ├── prepare_datasets.py    # Split verification and config generator
│   └── train_models.py        # Automated training and evaluation runner
│
├── models/                    # Model weights and checkpoints
│   ├── best.pt                # Trained multi-violation YOLO11 model
│   ├── wrong_direction_best.pt# Trained wrong-direction YOLO11 model
│   └── yolov11n.pt            # Base COCO fallback checkpoint
│
├── services/                  # Modular AI and backend services
│   ├── database_service.py    # SQLite schema, queries, metrics calculation
│   ├── detection_service.py   # YOLO11 detector wrapper and visualizer
│   ├── helmet_detector.py     # Rider-bike association & helmet compliance
│   ├── triple_riding_detector.py # Spatial rider counting per motorcycle
│   ├── wrong_side_detector.py # ByteTrack trajectory & contraflow rules
│   ├── tracking_service.py    # Multi-object tracking manager
│   ├── plate_service.py       # Number plate crop, preprocessing & OCR
│   ├── violation_aggregator.py # Multi-violation merger, deduplication & crops
│   └── video_processor.py     # Frame-by-frame video ingestion & encoding
│
├── templates/                 # Modern responsive HTML5 templates
│   ├── base.html              # Core navigation layout and footer
│   ├── index.html             # Command Center Dashboard with live charts
│   ├── upload.html            # Upload zone with 1-click sample selector
│   ├── results.html           # Media viewer, cards, evidence & download
│   ├── history.html           # Searchable and filterable violation registry
│   └── violation_details.html # Forensic inspection and officer audit review
│
├── static/
│   ├── css/style.css          # Dark-mode glassmorphic theme stylesheet
│   └── js/main.js             # Chart.js charts and dropzone interactions
│
├── outputs/                   # Generated output artifacts
│   ├── images/                # Annotated output images
│   ├── videos/                # Annotated output videos
│   └── evidence/              # Forensic high-resolution evidence crops
│
├── database/
│   └── traffic.db             # SQLite database
│
└── tests/                     # Automated unit and integration tests (19 tests, 100% pass)
    ├── test_detection.py
    ├── test_helmet_detector.py
    ├── test_triple_riding.py
    ├── test_wrong_side.py
    ├── test_aggregator.py
    └── test_flask_routes.py
```

---

## 6. How to Reproduce Dataset Preparation & Model Training

### Audit All 3 Datasets:
```bash
python scripts/audit_datasets.py
```

### Prepare Dataset Configs & Balanced Splits:
```bash
python scripts/prepare_datasets.py
```

### Train and Evaluate Models:
```bash
python scripts/train_models.py
```
This executes training of both models, evaluates checkpoints against official held-out validation sets, and saves `models/best.pt` and `models/wrong_direction_best.pt`.

---

## 7. Running the Flask Web Application

```bash
python app.py
```
Open your browser at `http://127.0.0.1:5000`.

### Key Pages:
- **Dashboard (`/`):** Real-time analytics, total jobs, violation breakdown, and interactive Chart.js visualizations.
- **Upload (`/upload`):** Upload images or MP4/AVI videos, customize permitted road direction, and use 1-click sample selector.
- **Results (`/results/<job_id>`):** Interactive annotated output player, individual violation cards, and license plate readings.
- **History (`/history`):** Complete searchable registry filterable by violation category and review status.
- **Forensic Review (`/violation/<id>`):** High-resolution evidence crop zoom, OCR readings, and officer decision action (`Confirm`, `Dispute`, `Pending`).

---

## 8. CLI Usage Examples

### Single Image Detection:
```bash
python detect.py --source "traffic dataset.v4i.yolov11/valid/images/overload-416-_jpg.rf.c5d66251ea6cd015dcad4fa8bbed9e74.jpg"
```

### Video Detection with ByteTrack:
```bash
python detect.py --source data/samples/test_traffic.mp4 --direction downward
```

### License Plate OCR:
```bash
python plate_ocr.py --image "traffic dataset.v4i.yolov11/valid/images/License-1003-_png.rf.4063f83446676822cf96659aab7e29fb.jpg"
```

---

## 9. Running Automated Tests

```bash
python -m unittest discover tests
```
Result: **19 tests passed** (`OK`).
