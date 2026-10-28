"""
Extract Head Crops for Vision Transformer (ViT) Fine-Tuning.
Crops helmet and no-helmet regions from 'traffic dataset.v4i.yolov11' using YOLO annotations.
Outputs a balanced, structured dataset in 'data/vit_dataset' and zips it to 'vit_dataset.zip'.
"""
import os
import cv2
import zipfile
import shutil

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "traffic dataset.v4i.yolov11")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "vit_dataset")
ZIP_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vit_dataset.zip")

CLASS_NAMES = {0: "helmet", 1: "no_helmet"}
MAX_PER_CLASS_TRAIN = 1500
MAX_PER_CLASS_VAL = 300

def extract_crops():
    print(f"[ViT Extraction] Reading dataset from: {DATASET_ROOT}")
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset directory '{DATASET_ROOT}' does not exist.")
        return

    # Clean previous output
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val"]:
        for cls in ["helmet", "no_helmet"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    counts = {"train": {"helmet": 0, "no_helmet": 0}, "val": {"helmet": 0, "no_helmet": 0}}

    for split in ["train", "valid"]:
        target_split = "train" if split == "train" else "val"
        images_dir = os.path.join(DATASET_ROOT, split, "images")
        labels_dir = os.path.join(DATASET_ROOT, split, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue

        img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[ViT Extraction] Processing {len(img_files)} images from {split} split...")

        max_limit = MAX_PER_CLASS_TRAIN if target_split == "train" else MAX_PER_CLASS_VAL

        for img_name in img_files:
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            if not os.path.exists(label_path):
                continue

            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id not in CLASS_NAMES:
                    continue

                cls_name = CLASS_NAMES[cls_id]
                if counts[target_split][cls_name] >= max_limit:
                    continue

                cx, cy, bw, bh = map(float, parts[1:5])
                # Convert normalized xywh to pixel coordinates
                x1 = max(0, int((cx - bw / 2.0) * w))
                y1 = max(0, int((cy - bh / 2.0) * h))
                x2 = min(w, int((cx + bw / 2.0) * w))
                y2 = min(h, int((cy + bh / 2.0) * h))

                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < 15 or crop_h < 15:
                    continue

                # Add 10% context padding for better feature learning
                pad_x = int(crop_w * 0.10)
                pad_y = int(crop_h * 0.10)
                px1 = max(0, x1 - pad_x)
                py1 = max(0, y1 - pad_y)
                px2 = min(w, x2 + pad_x)
                py2 = min(h, y2 + pad_y)

                crop = img[py1:py2, px1:px2]
                if crop.size == 0:
                    continue

                out_filename = f"{base_name}_{idx}.jpg"
                out_path = os.path.join(OUTPUT_DIR, target_split, cls_name, out_filename)
                cv2.imwrite(out_path, crop)
                counts[target_split][cls_name] += 1

    print("\n[ViT Extraction Summary]")
    print(f"  Train -> Helmet: {counts['train']['helmet']} | No-Helmet: {counts['train']['no_helmet']}")
    print(f"  Val   -> Helmet: {counts['val']['helmet']} | No-Helmet: {counts['val']['no_helmet']}")

    # Create zip archive
    print(f"\n[ViT Extraction] Zipping dataset to: {ZIP_OUTPUT} ...")
    with zipfile.ZipFile(ZIP_OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.join(OUTPUT_DIR, ".."))
                zipf.write(abs_path, rel_path)

    zip_size_mb = os.path.getsize(ZIP_OUTPUT) / (1024 * 1024)
    print(f"[ViT Extraction] Successfully created '{os.path.basename(ZIP_OUTPUT)}' ({zip_size_mb:.2f} MB)")

if __name__ == "__main__":
    extract_crops()
