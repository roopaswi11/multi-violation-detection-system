"""
Retrains the Traffic Multi-Violation detector (models/best.pt) on the FULL
`traffic dataset.v4i.yolov11` training split, at the app's inference size, on GPU if available.
Usage:
    python scripts/train_full.py                     # yolo11s, 60 epochs
    python scripts/train_full.py --model yolo11n.pt --epochs 40
    python scripts/train_full.py --no-install        # train only, keep models/best.pt
"""
import os, sys, shutil, argparse, tempfile
import torch, yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config import Config

DATASET_DIR = os.path.join(BASE_DIR, 'traffic dataset.v4i.yolov11')
CLASS_NAMES = ['helmet', 'no helmet', 'number plate', 'triple riding']

def write_data_yaml():
    data = {'path': DATASET_DIR.replace('\\', '/'), 'train': 'train/images', 'val': 'valid/images',
            'test': 'test/images', 'names': {i: n for i, n in enumerate(CLASS_NAMES)}}
    path = os.path.join(tempfile.gettempdir(), 'traffic_violation_full.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='yolo11s.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--imgsz', type=int, default=Config.IMAGE_SIZE)
    p.add_argument('--batch', type=int, default=-1, help='-1 = auto-fit to GPU memory')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--name', default='full_dataset')
    p.add_argument('--no-install', action='store_true')
    a = p.parse_args()
    device = 0 if torch.cuda.is_available() else 'cpu'
    batch = a.batch if device != 'cpu' or a.batch > 0 else 16
    data_yaml = write_data_yaml()
    project_dir = os.path.join(BASE_DIR, 'runs', 'train_traffic_violation')
    print(f"[train_full] device={device} model={a.model} epochs={a.epochs} imgsz={a.imgsz} batch={batch}")
    YOLO(a.model).train(data=data_yaml, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=batch,
                        device=device, workers=a.workers, project=project_dir, name=a.name, exist_ok=True, plots=True)
    best_pt = os.path.join(project_dir, a.name, 'weights', 'best.pt')
    best = YOLO(best_pt)
    for split in ('val', 'test'):
        r = best.val(data=data_yaml, split=split, imgsz=a.imgsz, device=device, plots=False, verbose=False)
        print(f"\n[train_full] {split} mAP@0.5={r.box.map50:.3f} mAP@0.5:0.95={r.box.map:.3f}")
        for i, n in best.names.items():
            print(f"    {n:14s} P={r.box.p[i]:.3f} R={r.box.r[i]:.3f} mAP50={r.box.ap50[i]:.3f}")
    if not a.no_install:
        shutil.copy(best_pt, Config.CUSTOM_MODEL_PATH)
        print(f"\n[train_full] Installed {best_pt} -> {Config.CUSTOM_MODEL_PATH}")

if __name__ == '__main__':
    main()
