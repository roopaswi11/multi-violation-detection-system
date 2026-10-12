# Dataset Audit Report: Traffic Violation Detection System
**Date:** October 11, 2026  
**System:** YOLO11, ViT, ByteTrack, EasyOCR, Flask  
**Environment:** Windows (Python 3.13.2, PyTorch 2.14.1+cpu, Ultralytics 8.4.175)

---

## Executive Summary

An exhaustive audit of all three datasets added to the workspace (`multi-voilation system/`) was conducted using `scripts/audit_datasets.py`. All image files, annotation text files, class IDs, bounding boxes, and split directories were inspected and validated.

All three datasets use the **standard YOLO bounding box format** (`<class_id> <x_center> <y_center> <width> <height>` normalized to $[0, 1]$). There are **zero missing labels, zero corrupt images, and 100% image-label pairing** across all splits.

---

## Detailed Dataset Profiles

### Dataset 1: IMVTDataset
- **Location:** `IMVTDataset/IMVTDataset/`
- **Configuration File:** `IMVTDataset/IMVTDataset/data.yaml`
- **Total Images:** 6,337
- **Annotation Format:** YOLO normalized bounding boxes (`.txt`)
- **Number of Classes:** 10
- **Class Mapping:**
  - `0`: `b` (rare artifact class)
  - `1`: `driver` (motorcycle driver / rider)
  - `2`: `earphone` (driver distraction)
  - `3`: `helmet` (rider wearing helmet)
  - `4`: `license_plate` (vehicle registration plate)
  - `5`: `mobile_phone` (phone in hand while riding)
  - `6`: `no_helmet` (rider without helmet)
  - `7`: `object` (held luggage/cargo)
  - `8`: `pillion` (passenger rider behind driver)
  - `9`: `two_wheeler` (motorcycle / scooter)
- **Split Distribution:**
  - **Train:** 5,785 images, 5,785 label files (100% paired)
  - **Validation:** 346 images, 346 label files (100% paired)
  - **Test:** 206 images, 206 label files (100% paired)
- **Class Frequencies (Train):**
  - `two_wheeler`: 323
  - `driver`: 268
  - `pillion`: 59
  - `helmet`: 182
  - `no_helmet`: 213
  - `license_plate`: 214
  - `mobile_phone`: 33
  - `earphone`: 46
  - `b`: 1, `object`: 1
- **Supported Detection Tasks:**
  - Driver & Pillion rider detection and counting.
  - Two-wheeler motorcycle localization.
  - Helmet vs. No-helmet classification.
  - License plate localization.

---

### Dataset 2: traffic dataset.v4i.yolov11
- **Location:** `traffic dataset.v4i.yolov11/`
- **Configuration File:** `traffic dataset.v4i.yolov11/data.yaml`
- **Total Images:** 5,675
- **Annotation Format:** YOLO normalized bounding boxes (`.txt`)
- **Number of Classes:** 4
- **Class Mapping:**
  - `0`: `helmet`
  - `1`: `no helmet`
  - `2`: `number plate`
  - `3`: `triple riding` (direct multi-rider motorcycle violation)
- **Split Distribution:**
  - **Train:** 5,238 images, 5,238 label files (100% paired)
  - **Validation:** 219 images, 219 label files (100% paired)
  - **Test:** 218 images, 218 label files (100% paired)
- **Class Frequencies (Train):**
  - `no helmet`: 1,946 instances
  - `triple riding`: 1,128 instances
  - `helmet`: 1,833 instances
  - `number plate`: 3,042 instances
- **Supported Detection Tasks:**
  - High-density helmet vs. no-helmet detection.
  - Direct triple-riding detection (overloaded motorcycles).
  - High-density number plate localization.

---

### Dataset 3: Wrong Direction.v4i.yolov11
- **Location:** `Wrong Direction.v4i.yolov11/`
- **Configuration File:** `Wrong Direction.v4i.yolov11/data.yaml`
- **Total Images:** 506
- **Annotation Format:** YOLO normalized bounding boxes (`.txt`)
- **Number of Classes:** 2
- **Class Mapping:**
  - `0`: `Right-Direction` (vehicles traveling in permitted lane direction)
  - `1`: `Wrong-Direction` (contraflow vehicles traveling against lane flow)
- **Split Distribution:**
  - **Train:** 353 images, 353 label files (100% paired)
  - **Validation:** 144 images, 144 label files (100% paired)
  - **Test:** 9 images, 9 label files (100% paired)
- **Class Frequencies (Train):**
  - `Wrong-Direction`: 181 instances
  - `Right-Direction`: 304 instances
- **Supported Detection Tasks:**
  - Static camera lane directional detection for contraflow riding.
  - Complements video trajectory analysis with visual directional appearance.

---

## Integrity & Quality Summary Table

| Dataset | Format | Images | Labels | Split Check | Bbox Range | Missing Files | Corrupt Images |
|---|---|---|---|---|---|---|---|
| **IMVTDataset** | YOLO txt | 6,337 | 6,337 | Train (5785) / Val (346) / Test (206) | Valid $[0, 1]$ | 0 | 0 |
| **traffic_dataset_v4** | YOLO txt | 5,675 | 5,675 | Train (5238) / Val (219) / Test (218) | Valid $[0, 1]$ | 0 | 0 |
| **wrong_direction_v4**| YOLO txt | 506 | 506 | Train (353) / Val (144) / Test (9) | Valid $[0, 1]$ | 0 | 0 |
| **TOTAL** | | **12,518** | **12,518** | **100% Verified** | **All Valid** | **0** | **0** |

---

## Task Mapping & Model Strategy Recommendation

1. **Traffic Multi-Violation Detector (`traffic_dataset_v4`):**
   - The primary workhorse model for detecting `helmet`, `no helmet`, `number plate`, and `triple riding`.
   - Contains over 7,900 annotated violation instances.
2. **Rider & Vehicle Composition Detector (`IMVTDataset`):**
   - Provides granular breakdown between `two_wheeler`, `driver`, `pillion`, and distracted driving.
3. **Wrong Direction Detector (`wrong_direction_v4`):**
   - Provides static appearance verification for contraflow motorcycles, supplementing ByteTrack trajectory vector analysis.
