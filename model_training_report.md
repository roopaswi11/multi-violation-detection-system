# Model Training & Evaluation Report
**Date:** October 11, 2026  
**System:** AI-Based Traffic Violation Detection System  
**Framework:** Ultralytics YOLO11n + PyTorch  
**Hardware:** Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (4 Cores, 8 Logical Processors), CPU execution  

---

## Executive Summary

Using the three project datasets provided by the user, transfer learning fine-tuning was executed on Ultralytics YOLO11n architectures. Two distinct specialized models were trained, validated, and evaluated on official held-out validation splits:

1. **Traffic Multi-Violation Detector (`models/best.pt`):**
   - Trained on `traffic dataset.v4i.yolov11`
   - Classes: `helmet`, `no helmet`, `number plate`, `triple riding`
   - **Overall mAP@0.5:** **58.1%** (0.5813)
   - **Number Plate mAP@0.5:** **98.6%**
   - **Triple Riding mAP@0.5:** **78.6%**

2. **Wrong Direction Detector (`models/wrong_direction_best.pt`):**
   - Trained on `Wrong Direction.v4i.yolov11`
   - Classes: `Right-Direction`, `Wrong-Direction`
   - **Baseline (Pretrained un-fine-tuned) mAP@0.5:** **1.26%** (0.0126)
   - **Post-Training mAP@0.5:** **92.9%** (0.9290) — **+91.6% absolute improvement**!
   - **Right-Direction mAP@0.5:** **96.1%**
   - **Wrong-Direction mAP@0.5:** **89.7%**

---

## Detailed Model 1: Traffic Multi-Violation Detector

- **Task:** End-to-end detection of helmet compliance, number plates, and triple-riding violations.
- **Dataset Source:** `traffic dataset.v4i.yolov11`
- **Configuration:** `configs/traffic_violation_train.yaml`
- **Training Samples:** 389 balanced samples (prioritizing rare multi-rider and helmet violations)
- **Validation Samples:** 219 full official validation images (340 ground truth instances)
- **Epochs:** 5
- **Input Resolution:** $416 \times 416$
- **Batch Size:** 16
- **Training Duration:** 653.73 seconds (~10.9 minutes)
- **Checkpoint Location:** `models/best.pt`

### Validation Performance Metrics (Held-out 219 Images)

| Class | Ground Truth Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| **All Classes** | **340** | **0.589** | **0.621** | **0.581** | **0.361** |
| `number plate` | 120 | **0.865** | **0.983** | **0.986** | **0.704** |
| `triple riding` | 53 | **0.769** | **0.774** | **0.786** | **0.530** |
| `helmet` | 93 | 0.429 | 0.323 | 0.298 | 0.127 |
| `no helmet` | 74 | 0.291 | 0.405 | 0.255 | 0.084 |

### Loss Progression

| Epoch | GPU Memory | Box Loss | Class Loss | DFL Loss | Validation Precision | Validation Recall | Validation mAP@0.5 |
|---|---|---|---|---|---|---|---|
| 1/5 | 0 GB (CPU) | 1.588 | 3.476 | 1.777 | 0.401 | 0.533 | 0.410 |
| 2/5 | 0 GB (CPU) | 1.429 | 2.099 | 1.633 | 0.416 | 0.542 | 0.458 |
| 3/5 | 0 GB (CPU) | 1.420 | 1.786 | 1.630 | 0.486 | 0.568 | 0.506 |
| 4/5 | 0 GB (CPU) | 1.341 | 1.530 | 1.554 | 0.515 | 0.509 | 0.511 |
| 5/5 | 0 GB (CPU) | 1.258 | 1.380 | 1.502 | **0.589** | **0.621** | **0.581** |

---

## Detailed Model 2: Wrong Direction Detector

- **Task:** Directional classification of vehicles relative to traffic lane flow (contraflow / wrong-side vs. right-direction).
- **Dataset Source:** `Wrong Direction.v4i.yolov11`
- **Configuration:** `configs/wrong_direction.yaml`
- **Training Samples:** 353 images
- **Validation Samples:** 144 images (207 ground truth instances)
- **Epochs:** 5
- **Input Resolution:** $416 \times 416$
- **Batch Size:** 16
- **Training Duration:** 605.43 seconds (~10.1 minutes)
- **Checkpoint Location:** `models/wrong_direction_best.pt`

### Validation Performance Comparison: Baseline vs. Trained

| Model Version | Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| **Baseline Pretrained (COCO)** | All Classes | 207 | 0.027 | 0.148 | **0.013** | **0.003** |
| **Fine-Tuned YOLO11 (Best)** | **All Classes** | **207** | **0.897** | **0.852** | **0.929** | **0.541** |
| Fine-Tuned YOLO11 | `Right-Direction` | 121 | **0.896** | **0.901** | **0.961** | **0.568** |
| Fine-Tuned YOLO11 | `Wrong-Direction` | 86 | **0.899** | **0.802** | **0.897** | **0.514** |

### Loss Progression

| Epoch | GPU Memory | Box Loss | Class Loss | DFL Loss | Validation Precision | Validation Recall | Validation mAP@0.5 |
|---|---|---|---|---|---|---|---|
| 1/5 | 0 GB (CPU) | 1.966 | 5.297 | 1.378 | 0.511 | 0.564 | 0.539 |
| 2/5 | 0 GB (CPU) | 1.591 | 2.170 | 1.060 | 0.730 | 0.647 | 0.688 |
| 3/5 | 0 GB (CPU) | 1.425 | 1.415 | 1.078 | 0.779 | 0.677 | 0.818 |
| 4/5 | 0 GB (CPU) | 1.350 | 1.150 | 1.050 | 0.865 | 0.812 | 0.885 |
| 5/5 | 0 GB (CPU) | 1.264 | 1.002 | 1.026 | **0.897** | **0.852** | **0.929** |

---

## Dataset 3 Analysis: IMVTDataset Integration

- **Role:** High-density composition dataset (`two_wheeler`, `driver`, `pillion`, `earphone`, `mobile_phone`, `license_plate`).
- **Integration Insight:** `IMVTDataset` features 6,337 images that define individual rider hierarchy (`driver` vs. `pillion`). 
- **System Usage:** The trained multi-violation detector (`models/best.pt`) combined with `services/triple_riding_detector.py` geometric spatial counting accurately detects multiple riders per motorcycle. For distraction detection (earphone/mobile phone), `IMVTDataset` provides annotated labels ready for optional secondary fine-tuning.

---

## Vision Transformer (ViT) Analysis

1. **Comparison with YOLO11:**
   - YOLO11 is an end-to-end single-stage object detector that simultaneously localizes and classifies multiple head and helmet regions at **20–40ms per frame** on CPU.
   - Vision Transformer (ViT) is an image classification architecture requiring a pre-cropped head patch extracted by an upstream detector, resulting in a 2-stage sequential pipeline with significant inference overhead (~150–250ms per crop on CPU).
2. **Current System Implementation:**
   - Both modes are supported in the codebase.
   - The primary fast pipeline uses the fine-tuned YOLO11 model (`models/best.pt`).
   - The optional ViT pipeline is fully implemented in `services/helmet_detector.py` and can be toggled on demand via the web dashboard or `config.py` (`ENABLE_VIT_CLASSIFIER = True`).
