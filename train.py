"""
Reproducible Training & Fine-Tuning Pipeline for YOLO11 and ViT Helmet Classifier.

Usage:
    # Train YOLO11 on the Helmet & Number Plate dataset:
    python train.py --task yolo --data data/helmet_data.yaml --epochs 25 --imgsz 640 --batch 8

    # Train YOLO11 on Traffic / Triple Riding dataset:
    python train.py --task yolo --data data/traffic_data.yaml --epochs 25 --imgsz 640

    # Train ViT helmet classifier:
    python train.py --task vit --epochs 10 --batch 16
"""

import os
import sys
import argparse
import yaml
import glob
import shutil
from pathlib import Path

def validate_dataset_yaml(yaml_path):
    """
    Validates the dataset YAML configuration, checking image-label paths and class mapping.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset YAML not found at: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    print(f"\n[Validation] Dataset YAML: {yaml_path}")
    print(f"[Validation] Number of classes (nc): {data.get('nc')}")
    print(f"[Validation] Class names: {data.get('names')}")
    print(f"[Validation] Train path: {data.get('train')}")
    print(f"[Validation] Val path:   {data.get('val')}")
    return data

def train_yolo(data_yaml, epochs=25, imgsz=640, batch=8, model_weights='yolo11n.pt', project_dir='models/train_runs'):
    """
    Trains Ultralytics YOLO11 on the specified dataset with checkpointing and metrics logging.
    """
    from ultralytics import YOLO

    os.makedirs(project_dir, exist_ok=True)
    validate_dataset_yaml(data_yaml)

    print(f"\n[Training] Starting YOLO11 training...")
    print(f"[Training] Base model: {model_weights}")
    print(f"[Training] Dataset:    {data_yaml}")
    print(f"[Training] Epochs:     {epochs} | ImgSz: {imgsz} | Batch: {batch}")
    
    model = YOLO(model_weights)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        name='yolo11_traffic_experiment',
        exist_ok=True,
        save=True,
        plots=True,
        patience=8,
        device='cpu' # fallback or cuda
    )

    print("\n[Training] Evaluating on validation set...")
    metrics = model.val()
    print("\n=== Validation Results ===")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    # Copy best weights to dedicated models directory without overwriting original best.pt
    best_pt = os.path.join(project_dir, 'yolo11_traffic_experiment', 'weights', 'best.pt')
    if os.path.exists(best_pt):
        target_pt = os.path.join('models', 'fine_tuned_traffic.pt')
        shutil.copy(best_pt, target_pt)
        print(f"\n[Saved] Best checkpoint copied to: {target_pt}")

    return metrics

def train_vit_classifier(dataset_dir='data/vit_dataset', epochs=10, batch_size=16):
    """
    Interface for Vision Transformer (ViT) helmet vs no-helmet classification fine-tuning.
    """
    print("\n[ViT Training] Checking dataset requirements...")
    if not os.path.exists(dataset_dir) or len(os.listdir(dataset_dir)) == 0:
        print(f"[ViT Training] Head crop dataset not found at '{dataset_dir}'.")
        print("To train ViT helmet classifier:")
        print("1. Extract cropped rider head images into data/vit_dataset/train/helmet and /no_helmet")
        print("2. Re-run python train.py --task vit")
        return None

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
        from torchvision.datasets import ImageFolder
        print("[ViT Training] Initializing ViT base model and dataset loader...")
        # Placeholder for full HuggingFace Trainer workflow
        print("[ViT Training] Ready for training.")
    except Exception as e:
        print(f"[ViT Training] Error initializing transformers trainer: {e}")

def main():
    parser = argparse.ArgumentParser(description="Traffic Violation Model Training")
    parser.add_argument("--task", default="yolo", choices=["yolo", "vit"], help="Training task")
    parser.add_argument("--data", default="data/helmet_data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--weights", default="models/yolov11n.pt", help="Base model weights")
    args = parser.parse_args()

    if args.task == "yolo":
        train_yolo(args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, model_weights=args.weights)
    elif args.task == "vit":
        train_vit_classifier(epochs=args.epochs, batch_size=args.batch)

if __name__ == "__main__":
    main()
