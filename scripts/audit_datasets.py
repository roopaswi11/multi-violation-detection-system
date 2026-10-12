"""
Comprehensive Dataset Audit Script for Traffic Violation Detection System.
Inspects all 3 datasets:
1. IMVTDataset
2. traffic dataset.v4i.yolov11
3. Wrong Direction.v4i.yolov11

Checks:
- Exact directory and file counts
- Annotation format & syntax
- Classes & class distribution
- Splits (train, val, test)
- Bounding box validity
- Missing images / missing labels / corrupt files
"""

import os
import sys
import glob
import yaml
from PIL import Image
from collections import Counter

def audit_split(img_dir, label_dir, class_names):
    report = {
        'total_images': 0,
        'total_labels': 0,
        'valid_pairs': 0,
        'missing_labels': [],
        'unpaired_labels': [],
        'empty_labels': 0,
        'invalid_boxes': 0,
        'corrupt_images': 0,
        'class_counts': Counter(),
        'image_shapes': Counter()
    }

    if not os.path.exists(img_dir):
        return report

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = {
        os.path.splitext(f)[0]: os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(img_exts)
    }
    report['total_images'] = len(img_files)

    label_files = {}
    if os.path.exists(label_dir):
        label_files = {
            os.path.splitext(f)[0]: os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.lower().endswith('.txt')
        }
    report['total_labels'] = len(label_files)

    for stem, img_path in img_files.items():
        # Check image readability
        try:
            with Image.open(img_path) as img:
                report['image_shapes'][img.size] += 1
        except Exception:
            report['corrupt_images'] += 1

        if stem not in label_files:
            report['missing_labels'].append(stem)
        else:
            report['valid_pairs'] += 1
            lbl_path = label_files[stem]
            try:
                with open(lbl_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if not lines:
                        report['empty_labels'] += 1
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            xc, yc, w, h = [float(p) for p in parts[1:5]]
                            # Check bbox coordinates
                            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                                report['invalid_boxes'] += 1
                            cname = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"Unknown_{cls_id}"
                            report['class_counts'][cname] += 1
                        else:
                            report['invalid_boxes'] += 1
            except Exception:
                report['invalid_boxes'] += 1

    for stem in label_files:
        if stem not in img_files:
            report['unpaired_labels'].append(stem)

    return report

def audit_dataset(name, root_dir, yaml_path):
    print(f"\n{'='*70}\nAUDITING DATASET: {name}\nPath: {root_dir}")
    if not os.path.exists(yaml_path):
        print(f"Error: YAML not found at {yaml_path}")
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    class_names = config.get('names', [])
    nc = config.get('nc', len(class_names))
    print(f"Classes ({nc}): {class_names}")

    # Detect splits
    possible_splits = [('train', 'train'), ('val', 'valid'), ('val', 'val'), ('test', 'test')]
    split_results = {}

    checked_dirs = set()
    for sname, dirname in possible_splits:
        img_dir = os.path.join(root_dir, dirname, 'images')
        lbl_dir = os.path.join(root_dir, dirname, 'labels')
        if not os.path.exists(img_dir):
            # Try direct folder without 'images' subfolder
            if os.path.exists(os.path.join(root_dir, dirname)) and not os.path.isdir(os.path.join(root_dir, dirname, 'images')):
                img_dir = os.path.join(root_dir, dirname)
                lbl_dir = os.path.join(root_dir, dirname)

        key = (sname, dirname)
        if (img_dir, lbl_dir) not in checked_dirs and os.path.exists(img_dir):
            checked_dirs.add((img_dir, lbl_dir))
            res = audit_split(img_dir, lbl_dir, class_names)
            split_results[f"{sname} ({dirname})"] = res
            print(f" - Split [{sname} - {dirname}]: {res['total_images']} images, {res['total_labels']} labels, {res['valid_pairs']} valid pairs")
            print(f"   Class counts: {dict(res['class_counts'])}")

    return {
        'name': name,
        'root_dir': root_dir,
        'yaml_path': yaml_path,
        'class_names': class_names,
        'nc': nc,
        'splits': split_results
    }

def main():
    base_dir = r"c:\Users\roopa\Documents\multi-voilation system"
    datasets_to_audit = [
        {
            'name': 'IMVTDataset',
            'root_dir': os.path.join(base_dir, 'IMVTDataset', 'IMVTDataset'),
            'yaml_path': os.path.join(base_dir, 'IMVTDataset', 'IMVTDataset', 'data.yaml')
        },
        {
            'name': 'traffic_dataset_v4',
            'root_dir': os.path.join(base_dir, 'traffic dataset.v4i.yolov11'),
            'yaml_path': os.path.join(base_dir, 'traffic dataset.v4i.yolov11', 'data.yaml')
        },
        {
            'name': 'wrong_direction_v4',
            'root_dir': os.path.join(base_dir, 'Wrong Direction.v4i.yolov11'),
            'yaml_path': os.path.join(base_dir, 'Wrong Direction.v4i.yolov11', 'data.yaml')
        }
    ]

    all_results = []
    for d in datasets_to_audit:
        res = audit_dataset(d['name'], d['root_dir'], d['yaml_path'])
        all_results.append(res)

    print("\n" + "="*70)
    print("DATASET AUDIT COMPLETE.")

if __name__ == '__main__':
    main()
