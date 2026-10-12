"""
Prepares balanced representative training subset for Traffic Multi-Violation Dataset
and executes training with validation on the full official validation split.
"""

import os
import shutil
import random
import yaml

def prepare_subset(num_samples=350):
    base_dir = os.path.abspath(r"c:\Users\roopa\Documents\multi-voilation system")
    src_train_img = os.path.join(base_dir, 'traffic dataset.v4i.yolov11', 'train', 'images')
    src_train_lbl = os.path.join(base_dir, 'traffic dataset.v4i.yolov11', 'train', 'labels')

    dst_dir = os.path.join(base_dir, 'data', 'traffic_subset')
    dst_img = os.path.join(dst_dir, 'images')
    dst_lbl = os.path.join(dst_dir, 'labels')
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    # Scan all labels to find samples of each class
    # 0: helmet, 1: no helmet, 2: number plate, 3: triple riding
    class_images = {0: [], 1: [], 2: [], 3: []}
    
    label_files = [f for f in os.listdir(src_train_lbl) if f.endswith('.txt')]
    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        img_name = stem + '.jpg'
        if not os.path.exists(os.path.join(src_train_img, img_name)):
            continue
        with open(os.path.join(src_train_lbl, lf), 'r') as f:
            classes_in_file = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))
            for c in classes_in_file:
                if c in class_images:
                    class_images[c].append(stem)

    print("Available images per class in training set:")
    for c, imgs in class_images.items():
        print(f" - Class {c}: {len(imgs)} candidate images")

    # Sample balanced set prioritizing triple riding (rarest/most critical)
    selected_stems = set()
    random.seed(42)

    # 1. Take all/many triple riding images
    tr_samples = random.sample(class_images[3], min(100, len(class_images[3])))
    selected_stems.update(tr_samples)

    # 2. Add no-helmet samples
    nh_samples = random.sample(class_images[1], min(100, len(class_images[1])))
    selected_stems.update(nh_samples)

    # 3. Add helmet samples
    h_samples = random.sample(class_images[0], min(100, len(class_images[0])))
    selected_stems.update(h_samples)

    # 4. Add number plate samples
    np_samples = random.sample(class_images[2], min(100, len(class_images[2])))
    selected_stems.update(np_samples)

    print(f"\nTotal unique balanced samples selected: {len(selected_stems)}")

    # Copy to subset directory
    for stem in selected_stems:
        shutil.copy(os.path.join(src_train_img, stem + '.jpg'), os.path.join(dst_img, stem + '.jpg'))
        shutil.copy(os.path.join(src_train_lbl, stem + '.txt'), os.path.join(dst_lbl, stem + '.txt'))

    # Generate YAML pointing train to subset and val to FULL official valid set!
    val_dir = os.path.join(base_dir, 'traffic dataset.v4i.yolov11', 'valid', 'images')
    test_dir = os.path.join(base_dir, 'traffic dataset.v4i.yolov11', 'test', 'images')
    
    cfg = {
        'path': base_dir.replace('\\', '/'),
        'train': os.path.relpath(dst_img, base_dir).replace('\\', '/'),
        'val': os.path.relpath(val_dir, base_dir).replace('\\', '/'),
        'test': os.path.relpath(test_dir, base_dir).replace('\\', '/'),
        'nc': 4,
        'names': ['helmet', 'no helmet', 'number plate', 'triple riding']
    }

    cfg_path = os.path.join(base_dir, 'configs', 'traffic_violation_train.yaml')
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"Generated training config: {cfg_path}")
    return cfg_path

if __name__ == '__main__':
    prepare_subset()
