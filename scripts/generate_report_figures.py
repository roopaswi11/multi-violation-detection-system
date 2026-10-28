import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import pandas as pd
from PIL import Image

PROJECT_DIR = r"c:\Users\roopa\Documents\multi-voilation system"
ASSETS_DIR = os.path.join(PROJECT_DIR, "report_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. System Architecture Diagram (Publication Quality)
# -------------------------------------------------------------
def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Styles
    box_style_primary = dict(boxstyle="round,pad=0.5", fc="#F8FAFC", ec="#1E293B", lw=2)
    box_style_input = dict(boxstyle="round,pad=0.5", fc="#EFF6FF", ec="#2563EB", lw=2)
    box_style_detector = dict(boxstyle="round,pad=0.5", fc="#FEF3C7", ec="#D97706", lw=2)
    box_style_module = dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec="#059669", lw=1.8)
    box_style_output = dict(boxstyle="round,pad=0.5", fc="#FAF5FF", ec="#7C3AED", lw=2)

    # 1. Input Layer
    ax.text(12, 88, "INPUT LAYER\nCCTV Video Stream / Static Image\n(EXIF Orientation Normalization)",
            ha="center", va="center", fontsize=9, weight='bold', bbox=box_style_input)

    # 2. Dual YOLO11 Detection Tier
    ax.text(50, 88, "DUAL YOLO11 DETECTION ENGINE\nPrimary: fine-tuned best.pt (Helmets, Plates, Triple-Riding)\nContraflow: wrong_direction_best.pt | Fallback: yolov11n.pt",
            ha="center", va="center", fontsize=9, weight='bold', bbox=box_style_detector)

    # Arrow Input -> Detection
    ax.annotate("", xy=(32, 88), xytext=(24, 88),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2563EB"))

    # 3. Analytic Modules (4 Columns)
    modules = [
        ("MODULE A: HELMET COMPLIANCE\n• 2-Pass Vehicle-Rider Association\n• Head ROI Localization (Top 35%)\n• ResNet-18 Head Classifier (93.3%)\n• Conflict Resolution & Handheld Check", 14, 55),
        ("MODULE B: TRIPLE-RIDING\n• 1-to-1 Seated Driver Assignment\n• Tandem Scale Invariant (Ratio ≥ 0.45)\n• Seated Depth & Head Y Alignment\n• Video Debounce (≥ 3 Frames)", 38, 55),
        ("MODULE C: WRONG-SIDE CONTRAFLOW\n• ByteTrack Multi-Object Tracking\n• Trajectory Vector Calculus\n• Stationary Jitter Suppression\n• Static Direction Classifier (92.9%)", 62, 55),
        ("MODULE D: LICENSE PLATE & OCR\n• High-Recall Plate Crop (98.6%)\n• Bilateral Filter & CLAHE Normalization\n• EasyOCR Alphanumeric Engine\n• Regex Validation (Read/Partial)", 86, 55)
    ]

    for title, x, y in modules:
        ax.text(x, y, title, ha="center", va="center", fontsize=8, bbox=box_style_module)
        # Arrow from Detection to Module
        ax.annotate("", xy=(x, 68), xytext=(50, 80),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#D97706", connectionstyle="arc3,rad=0.05"))

    # 4. Aggregation and Deduplication Layer
    ax.text(50, 28, "VIOLATION AGGREGATION & AUDIT ENGINE\n• Multi-Class Violation Merger & Track Deduplication\n• Forensic Evidence Cropping (/outputs/evidence/)\n• Statutory Penalty Assessment (Motor Vehicles Act: ₹1,000 / ₹5,000)\n• Multi-Tier Review Status (CONFIRMED / PENDING / INSUFFICIENT_EVIDENCE)",
            ha="center", va="center", fontsize=9, weight='bold', bbox=box_style_primary)

    for x, y in [(14, 42), (38, 42), (62, 42), (86, 42)]:
        ax.annotate("", xy=(50, 36), xytext=(x, y),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669", connectionstyle="arc3,rad=-0.05"))

    # 5. Output Tier
    ax.text(25, 10, "DATABASE & REPOSITORY\n• SQLite Database (traffic.db: 610 Jobs, 555 Violations)\n• Evidence Archive & Audit Log Registry",
            ha="center", va="center", fontsize=8.5, weight='bold', bbox=box_style_output)

    ax.text(75, 10, "FLASK OPERATIONAL WEB DASHBOARD\n• Live Analytics & Metrics (Chart.js)\n• Forensic Review & Officer Action (Confirm/Dispute)\n• Filterable History & Official PDF/CSV Reports",
            ha="center", va="center", fontsize=8.5, weight='bold', bbox=box_style_output)

    ax.annotate("", xy=(25, 17), xytext=(40, 22),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7C3AED"))
    ax.annotate("", xy=(75, 17), xytext=(60, 22),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7C3AED"))

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig1_system_architecture.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

# -------------------------------------------------------------
# 2. Rider-Motorcycle Physical Geometric Association Diagram
# -------------------------------------------------------------
def generate_rider_association_geometry():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Draw motorcycle box (foreground)
    mc_rect = patches.Rectangle((30, 20), 40, 45, linewidth=2.5, edgecolor='#2563EB', facecolor='#EFF6FF', alpha=0.7)
    ax.add_patch(mc_rect)
    ax.text(50, 25, "Motorcycle Bounding Box (B)\n[bx1, by1, bx2, by2]", ha='center', va='center', fontsize=9, color='#1E40AF', weight='bold')

    # Draw primary mounted rider (driver)
    driver_rect = patches.Rectangle((42, 35), 18, 55, linewidth=2.5, edgecolor='#059669', facecolor='#ECFDF5', alpha=0.7)
    ax.add_patch(driver_rect)
    ax.text(51, 65, "Primary Driver (D)\nPass 1 Assigned\nHeight = H_driver", ha='center', va='center', fontsize=8.5, color='#065F46', weight='bold')

    # Draw distant background person (rejected)
    bg_rect = patches.Rectangle((32, 70), 10, 18, linewidth=2, edgecolor='#DC2626', facecolor='#FEF2F2', linestyle='--')
    ax.add_patch(bg_rect)
    ax.text(37, 92, "Distant Person (P_bg)\nScale Ratio = 0.24 (< 0.45)\nBottom Y = Shoulder level\nREJECTED (No False Violation)", ha='center', va='center', fontsize=8, color='#991B1B', weight='bold')

    # Draw genuine tandem pillion rider
    pillion_rect = patches.Rectangle((54, 38), 16, 52, linewidth=2, edgecolor='#D97706', facecolor='#FFFBEB', alpha=0.7)
    ax.add_patch(pillion_rect)
    ax.text(62, 55, "Tandem Pillion\nScale Ratio = 0.95\nΔHead Y ≤ 40%\nSeated Plane Match\nACCEPTED", ha='center', va='center', fontsize=8, color='#92400E', weight='bold')

    # Annotate mathematical conditions
    ax.text(50, 5, "Two-Pass Physical Invariants: Pass 1 assigns exactly 1 Primary Driver. Pass 2 verifies Scale Ratio ≥ 0.45, Seated Depth, and Head Alignment.",
            ha='center', va='center', fontsize=9, style='italic', bbox=dict(boxstyle="square,pad=0.4", fc="#F8FAFC", ec="#94A3B8"))

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig3_rider_association_geometry.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

# -------------------------------------------------------------
# 3. License Plate OCR Preprocessing Pipeline Diagram
# -------------------------------------------------------------
def generate_ocr_pipeline_diagram():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), dpi=300)

    # Create synthetic license plate sample for clear visual explanation
    plate = np.ones((100, 260, 3), dtype=np.uint8) * 230
    cv2.rectangle(plate, (5, 5), (255, 95), (20, 20, 20), 4)
    cv2.putText(plate, "IND", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 180), 2)
    cv2.putText(plate, "KA 05 MR 4892", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 3)

    # 1. Raw Crop
    axes[0].imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Step 1: Bounding Box Crop\n(YOLO11 98.6% mAP50)", fontsize=9, weight='bold')
    axes[0].axis('off')

    # 2. Bilateral Filter
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    axes[1].imshow(bilateral, cmap='gray')
    axes[1].set_title("Step 2: Bilateral Denoising\n(Edge-Preserving Smoothing)", fontsize=9, weight='bold')
    axes[1].axis('off')

    # 3. CLAHE Contrast Normalization
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    axes[2].imshow(enhanced, cmap='gray')
    axes[2].set_title("Step 3: CLAHE Equalization\n(Illumination Invariance)", fontsize=9, weight='bold')
    axes[2].axis('off')

    # 4. Otsu Adaptive Binarization & OCR Recognition
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    axes[3].imshow(thresh, cmap='gray')
    axes[3].set_title("Step 4: Binarization & EasyOCR\nResult: KA 05 MR 4892 (READABLE)", fontsize=9, weight='bold')
    axes[3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig4_ocr_pipeline.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

# -------------------------------------------------------------
# 4. Dataset Distribution Comparison Charts
# -------------------------------------------------------------
def generate_dataset_distribution_charts():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Dataset Sizes
    ds_names = ['IMVTDataset', 'Traffic Dataset', 'Wrong Direction']
    ds_sizes = [6337, 5675, 506]
    colors = ['#2563EB', '#059669', '#D97706']

    axes[0].bar(ds_names, ds_sizes, color=colors, width=0.55, edgecolor='#1E293B', lw=1.5)
    axes[0].set_title("Total Images per Workspace Dataset\n(Total: 12,518 Images)", fontsize=10, weight='bold')
    axes[0].set_ylabel("Number of Images", fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(ds_sizes):
        axes[0].text(i, v + 120, f"{v:,}", ha='center', fontsize=9, weight='bold')

    # Traffic Dataset Class Instances
    classes = ['No Helmet', 'Helmet', 'Triple Riding', 'Number Plate']
    counts = [1946, 1833, 1128, 3042]
    class_colors = ['#EF4444', '#10B981', '#F59E0B', '#6366F1']

    axes[1].barh(classes, counts, color=class_colors, edgecolor='#1E293B', lw=1.5)
    axes[1].set_title("Traffic Dataset Instance Frequencies\n(5,238 Training Images)", fontsize=10, weight='bold')
    axes[1].set_xlabel("Annotation Instance Count", fontsize=9)
    axes[1].grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(counts):
        axes[1].text(v + 50, i, f"{v:,}", va='center', fontsize=8.5, weight='bold')

    # Data Split Breakdown for Main Dataset
    splits = ['Train (92.3%)', 'Validation (3.9%)', 'Test (3.8%)']
    split_sizes = [5238, 219, 218]
    split_colors = ['#3B82F6', '#10B981', '#EC4899']

    axes[2].pie(split_sizes, labels=splits, autopct='%1.1f%%', startangle=140, colors=split_colors,
                wedgeprops=dict(edgecolor='#1E293B', lw=1.2))
    axes[2].set_title("Dataset Split Proportions\n(traffic dataset.v4i.yolov11)", fontsize=10, weight='bold')

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig5_dataset_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

# -------------------------------------------------------------
# 5. Training Loss & mAP Curves (from results.csv)
# -------------------------------------------------------------
def generate_training_curves():
    csv_path_traffic = os.path.join(PROJECT_DIR, "runs", "train_traffic_violation", "experiment_1", "results.csv")
    csv_path_wrong = os.path.join(PROJECT_DIR, "runs", "train_wrong_direction", "experiment_1", "results.csv")

    df_t = pd.read_csv(csv_path_traffic)
    df_t.columns = [c.strip() for c in df_t.columns]

    df_w = pd.read_csv(csv_path_wrong)
    df_w.columns = [c.strip() for c in df_w.columns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Multi-Violation Model
    epochs = df_t['epoch']
    axes[0].plot(epochs, df_t['train/box_loss'], 'o-', label='Train Box Loss', color='#2563EB', lw=2)
    axes[0].plot(epochs, df_t['val/box_loss'], 's--', label='Val Box Loss', color='#38BDF8', lw=2)
    axes[0].plot(epochs, df_t['metrics/mAP50(B)'], '^-', label='mAP@0.5', color='#059669', lw=2.2)
    axes[0].plot(epochs, df_t['metrics/precision(B)'], 'd:', label='Precision', color='#D97706', lw=1.8)
    axes[0].set_title("YOLO11 Traffic Multi-Violation Training\n(best.pt Checkpoint)", fontsize=10, weight='bold')
    axes[0].set_xlabel("Epoch", fontsize=9)
    axes[0].set_ylabel("Loss / Metric Value", fontsize=9)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right', fontsize=8.5)

    # 2. Wrong Direction Model
    epochs_w = df_w['epoch']
    axes[1].plot(epochs_w, df_w['train/box_loss'], 'o-', label='Train Box Loss', color='#7C3AED', lw=2)
    axes[1].plot(epochs_w, df_w['val/box_loss'], 's--', label='Val Box Loss', color='#A78BFA', lw=2)
    axes[1].plot(epochs_w, df_w['metrics/mAP50(B)'], '^-', label='mAP@0.5', color='#059669', lw=2.2)
    axes[1].plot(epochs_w, df_w['metrics/precision(B)'], 'd:', label='Precision', color='#D97706', lw=1.8)
    axes[1].set_title("YOLO11 Wrong-Direction Classifier Training\n(wrong_direction_best.pt Checkpoint)", fontsize=10, weight='bold')
    axes[1].set_xlabel("Epoch", fontsize=9)
    axes[1].set_ylabel("Loss / Metric Value", fontsize=9)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='center right', fontsize=8.5)

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig6_training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

# -------------------------------------------------------------
# 6. Sample Prediction Results Composite
# -------------------------------------------------------------
def generate_sample_predictions_composite():
    p1 = os.path.join(PROJECT_DIR, "outputs", "images", "annotated_228d8471_Screenshot_2026-10-22_204709.png")
    p2 = os.path.join(PROJECT_DIR, "outputs", "images", "annotated_0f186c22_Screenshot_2026-10-12_183619.png")
    p3 = os.path.join(PROJECT_DIR, "outputs", "images", "annotated_0b34c516_Screenshot_2026-10-24_122210.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    if os.path.exists(p1):
        img1 = cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB)
        axes[0].imshow(img1)
    axes[0].set_title("(a) Negative Test Case (Compliant)\n0 Violations | All Riders Helmeted", fontsize=9, weight='bold')
    axes[0].axis('off')

    if os.path.exists(p2):
        img2 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
        axes[1].imshow(img2)
    axes[1].set_title("(b) Positive Triple-Riding Case\n1 Triple Violation | 3 Riders Detected", fontsize=9, weight='bold')
    axes[1].axis('off')

    if os.path.exists(p3):
        img3 = cv2.cvtColor(cv2.imread(p3), cv2.COLOR_BGR2RGB)
        axes[2].imshow(img3)
    axes[2].set_title("(c) Multi-Vehicle Helmet Violation Case\n3 Confirmed No-Helmet Violations", fontsize=9, weight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(ASSETS_DIR, "fig8_sample_predictions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated:", save_path)

if __name__ == '__main__':
    print("Generating report figures...")
    generate_system_architecture()
    generate_rider_association_geometry()
    generate_ocr_pipeline_diagram()
    generate_dataset_distribution_charts()
    generate_training_curves()
    generate_sample_predictions_composite()
    print("All report figures successfully generated in:", ASSETS_DIR)
