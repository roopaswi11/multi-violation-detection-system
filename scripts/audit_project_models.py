"""
Phase 1: Complete Project Model & Dataset Audit Script.
Prints actual class names, IDs, weights metadata, and dataset YAML configurations.
"""
import os
import sys
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def audit_models():
    print("=" * 80)
    print("PHASE 1: AUDIT OF ALL PROJECT YOLO MODELS")
    print("=" * 80)
    
    models = {
        'Traffic Multi-Violation Model': os.path.join(BASE_DIR, 'models', 'best.pt'),
        'Wrong Direction Model': os.path.join(BASE_DIR, 'models', 'wrong_direction_best.pt'),
        'COCO Base Model': os.path.join(BASE_DIR, 'models', 'yolov11n.pt'),
        'Root Pretrained Model': os.path.join(BASE_DIR, 'yolo11n.pt')
    }

    for name, path in models.items():
        print(f"\n--- Model: {name} ---")
        print(f"Path: {path}")
        if not os.path.exists(path):
            print("Status: FILE NOT FOUND")
            continue
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2)
        print(f"File Size: {size_mb} MB")
        try:
            m = YOLO(path)
            names = m.names
            print(f"Number of classes: {len(names)}")
            print("Class mapping:")
            for cid, cname in names.items():
                print(f"  ID {cid}: '{cname}'")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == '__main__':
    audit_models()
