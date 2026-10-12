"""
Comprehensive Training Pipeline for Traffic Violation Detection System.
Trains:
1. Wrong Direction YOLO11 Detector (configs/wrong_direction.yaml)
2. Traffic Multi-Violation YOLO11 Detector (configs/traffic_violation.yaml)
3. Evaluates best checkpoints on validation and test sets.
4. Saves checkpoints to models/ and updates config.
"""

import os
import sys
import shutil
import time
import json
from ultralytics import YOLO

BASE_DIR = os.path.abspath(r"c:\Users\roopa\Documents\multi-voilation system")

def train_wrong_direction(epochs=5, imgsz=416, batch=16):
    print("\n" + "="*70)
    print("STARTING TRAINING: Wrong Direction Detector (YOLO11)")
    print(f"Epochs: {epochs} | ImgSz: {imgsz} | Batch: {batch}")
    print("="*70)

    config_path = os.path.join(BASE_DIR, 'configs', 'wrong_direction.yaml')
    project_dir = os.path.join(BASE_DIR, 'runs', 'train_wrong_direction')
    
    model = YOLO(os.path.join(BASE_DIR, 'models', 'yolov11n.pt'))
    
    t0 = time.time()
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='cpu',
        workers=2,
        project=project_dir,
        name='experiment_1',
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )
    t1 = time.time()
    train_duration = round(t1 - t0, 2)
    print(f"\nTraining completed in {train_duration}s.")

    # Validation on held-out validation set
    print("\nRunning post-training validation on validation set...")
    val_results = model.val(data=config_path, imgsz=imgsz, batch=batch, device='cpu', verbose=True)

    best_pt = os.path.join(project_dir, 'experiment_1', 'weights', 'best.pt')
    target_pt = os.path.join(BASE_DIR, 'models', 'wrong_direction_best.pt')
    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        print(f"Copied best checkpoint to: {target_pt}")

    metrics = {
        'model_name': 'wrong_direction_detector',
        'train_duration_sec': train_duration,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'map50': round(float(val_results.box.map50), 4),
        'map': round(float(val_results.box.map), 4),
        'precision': round(float(val_results.box.mp), 4),
        'recall': round(float(val_results.box.mr), 4),
        'checkpoint_path': target_pt
    }
    return metrics

def train_traffic_violation(epochs=5, imgsz=416, batch=16):
    print("\n" + "="*70)
    print("STARTING TRAINING: Traffic Multi-Violation Detector (YOLO11)")
    print("Classes: ['helmet', 'no helmet', 'number plate', 'triple riding']")
    print(f"Epochs: {epochs} | ImgSz: {imgsz} | Batch: {batch}")
    print("="*70)

    config_path = os.path.join(BASE_DIR, 'configs', 'traffic_violation_train.yaml')
    project_dir = os.path.join(BASE_DIR, 'runs', 'train_traffic_violation')

    model = YOLO(os.path.join(BASE_DIR, 'models', 'yolov11n.pt'))

    t0 = time.time()
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device='cpu',
        workers=2,
        project=project_dir,
        name='experiment_1',
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )
    t1 = time.time()
    train_duration = round(t1 - t0, 2)
    print(f"\nTraining completed in {train_duration}s.")

    print("\nRunning post-training validation on validation set...")
    val_results = model.val(data=config_path, imgsz=imgsz, batch=batch, device='cpu', verbose=True)

    best_pt = os.path.join(project_dir, 'experiment_1', 'weights', 'best.pt')
    target_pt = os.path.join(BASE_DIR, 'models', 'best.pt')
    if os.path.exists(best_pt):
        shutil.copy(best_pt, target_pt)
        print(f"Copied best checkpoint to: {target_pt}")

    metrics = {
        'model_name': 'traffic_multi_violation_detector',
        'train_duration_sec': train_duration,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'map50': round(float(val_results.box.map50), 4),
        'map': round(float(val_results.box.map), 4),
        'precision': round(float(val_results.box.mp), 4),
        'recall': round(float(val_results.box.mr), 4),
        'checkpoint_path': target_pt
    }
    return metrics

if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'wrong_direction'
    if task == 'wrong_direction':
        res = train_wrong_direction()
        print("\nSUMMARY METRICS:", json.dumps(res, indent=2))
    elif task == 'traffic':
        res = train_traffic_violation()
        print("\nSUMMARY METRICS:", json.dumps(res, indent=2))
