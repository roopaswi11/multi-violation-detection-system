
import cv2
from ultralytics import YOLO
from pathlib import Path

video_path = Path(
    r"C:\Users\roopa\Downloads\WhatsApp Video 2026-10-27 at 12.14.28 PM.mp4"
)
model_path = Path("models/yolov11n.pt")

print("1. Loading model...", flush=True)
model = YOLO(str(model_path))
print("2. Model loaded successfully.", flush=True)

print("3. Opening video...", flush=True)
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("ERROR: Cannot open video:", video_path)
    raise SystemExit(1)

print("Video opened successfully.")
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read the first frame.")
    raise SystemExit(1)

print("4. Running YOLO inference on CPU...", flush=True)
results = model.predict(frame, verbose=False, device="cpu")

print("5. INFERENCE SUCCESS", flush=True)
print("Detections:", len(results[0].boxes))
print("Detected classes:", [
    results[0].names[int(box.cls[0])]
    for box in results[0].boxes
])
