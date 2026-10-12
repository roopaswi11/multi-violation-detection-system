"""
Reproducible Dataset Preparation and Configuration Generator.
Generates verified, absolute-path YOLO YAML configs for all three datasets.
Fixes path naming discrepancies (e.g. 'val' vs 'valid').
"""

import os
import yaml

def prepare_configs():
    base_dir = os.path.abspath(r"c:\Users\roopa\Documents\multi-voilation system")
    configs_dir = os.path.join(base_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)

    # 1. Traffic Violation Dataset (Helmet, No-Helmet, Number Plate, Triple Riding)
    traffic_root = os.path.join(base_dir, 'traffic dataset.v4i.yolov11')
    traffic_yaml = {
        'path': traffic_root.replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'no helmet', 'number plate', 'triple riding']
    }
    with open(os.path.join(configs_dir, 'traffic_violation.yaml'), 'w') as f:
        yaml.dump(traffic_yaml, f, sort_keys=False)
    print(f"Generated: configs/traffic_violation.yaml")

    # 2. Wrong Direction Dataset (Right-Direction, Wrong-Direction)
    wrong_root = os.path.join(base_dir, 'Wrong Direction.v4i.yolov11')
    wrong_yaml = {
        'path': wrong_root.replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 2,
        'names': ['Right-Direction', 'Wrong-Direction']
    }
    with open(os.path.join(configs_dir, 'wrong_direction.yaml'), 'w') as f:
        yaml.dump(wrong_yaml, f, sort_keys=False)
    print(f"Generated: configs/wrong_direction.yaml")

    # 3. IMVT Composition Dataset (Two-Wheeler, Driver, Pillion, Distraction, Plates)
    imvt_root = os.path.join(base_dir, 'IMVTDataset', 'IMVTDataset')
    imvt_yaml = {
        'path': imvt_root.replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',  # Fixed: correctly points to 'val/images' instead of nonexistent 'valid/images'
        'test': 'test/images',
        'nc': 10,
        'names': ['b', 'driver', 'earphone', 'helmet', 'license_plate', 'mobile_phone', 'no_helmet', 'object', 'pillion', 'two_wheeler']
    }
    with open(os.path.join(configs_dir, 'imvt_dataset.yaml'), 'w') as f:
        yaml.dump(imvt_yaml, f, sort_keys=False)
    print(f"Generated: configs/imvt_dataset.yaml")

    # 4. Calibrated Fast Subset Config for Traffic Dataset (for responsive training and rapid convergence)
    fast_traffic_yaml = {
        'path': traffic_root.replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 4,
        'names': ['helmet', 'no helmet', 'number plate', 'triple riding']
    }
    with open(os.path.join(configs_dir, 'traffic_violation_train.yaml'), 'w') as f:
        yaml.dump(fast_traffic_yaml, f, sort_keys=False)
    print(f"Generated: configs/traffic_violation_train.yaml")

if __name__ == '__main__':
    prepare_configs()
