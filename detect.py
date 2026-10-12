"""
CLI Detection Script for Traffic Violation Detection System.
Usage:
    python detect.py --source data/samples/sample.jpg
    python detect.py --source path/to/video.mp4 --conf 0.40 --direction downward
"""
import sys
import os
import argparse
import json

# Ensure services directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from services.detection_service import DetectionService
from services.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="AI Traffic Violation Detector")
    parser.add_argument("--source", required=True, help="Path to input image or video")
    parser.add_argument("--model", default=None, help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=Config.CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--direction", default="auto", choices=["auto", "downward", "upward", "leftward", "rightward", "left_to_right", "right_to_left"], help="Permitted traffic direction")
    parser.add_argument("--output", default=None, help="Custom output path")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"Error: Source file '{args.source}' does not exist.")
        sys.exit(1)

    detector = DetectionService(model_path=args.model, conf_threshold=args.conf)
    processor = VideoProcessor(detector=detector, permitted_direction=args.direction)

    ext = os.path.splitext(args.source)[1].lower().replace('.', '')
    if ext in Config.ALLOWED_IMAGE_EXTENSIONS:
        print(f"[detect.py] Processing image: {args.source}")
        out_path, violations, detections = processor.process_image(args.source, output_path=args.output)
        print(f"\n[detect.py] Processing Complete!")
        print(f"Output saved to: {out_path}")
        print(f"Total violations found: {len(violations)}")
        for v in violations:
            conf_str = f"{v['confidence']:.2f}" if v.get('confidence') is not None else "Uncertain / Review"
            print(f" - [{v['violation_type'].upper()}] Conf: {conf_str} | Reason: {v.get('reason')}")
            if v.get('plate_text'):
                print(f"   License Plate: {v['plate_text']} (Status: {v.get('plate_status')})")

    elif ext in Config.ALLOWED_VIDEO_EXTENSIONS:
        print(f"[detect.py] Processing video: {args.source}")
        out_path, violations, total_frames = processor.process_video(
            args.source, output_path=args.output,
            progress_callback=lambda f, t, p: print(f"Processing: frame {f}/{t} ({p:.1f}%)")
        )
        print(f"\n[detect.py] Video Processing Complete!")
        print(f"Output video saved to: {out_path}")
        print(f"Total unique violations: {len(violations)}")
        for v in violations:
            print(f" - [{v['violation_type'].upper()}] Track: {v.get('track_id')} | Frame: {v.get('frame_number')} | Conf: {v['confidence']:.2f}")
    else:
        print(f"Unsupported file format: {ext}")
        sys.exit(1)

if __name__ == "__main__":
    main()
