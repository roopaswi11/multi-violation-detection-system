import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'traffic-violation-secret-key-2026')
    
    # Storage Paths
    BASE_DIR = BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, 'yolov11n.pt')
    CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, 'best.pt')
    WRONG_DIRECTION_MODEL_PATH = os.path.join(MODELS_DIR, 'wrong_direction_best.pt')
    HEAD_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'head_classifier.pth')
    POSE_MODEL_PATH = os.path.join(BASE_DIR, 'yolo11n-pose.pt')
    
    UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    OUTPUTS_IMAGES_DIR = os.path.join(OUTPUTS_DIR, 'images')
    OUTPUTS_VIDEOS_DIR = os.path.join(OUTPUTS_DIR, 'videos')
    OUTPUTS_EVIDENCE_DIR = os.path.join(OUTPUTS_DIR, 'evidence')
    
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'traffic.db')
    FINE_SCHEDULE_PATH = os.path.join(BASE_DIR, 'configs', 'fine_schedule.json')
    DEFAULT_JURISDICTION = os.environ.get('DEFAULT_JURISDICTION', 'India (MV Act 2019)')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SAMPLES_DIR = os.path.join(DATA_DIR, 'samples')
    
    # Upload Settings
    ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
    
    # YOLO Detection Settings
    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45
    IMAGE_SIZE = 640
    DEVICE = 'cuda' if os.environ.get('FORCE_CUDA') == '1' else 'cpu'
    
    # Class IDs for standard COCO vs custom model
    # COCO: 0=person, 3=motorcycle
    COCO_MOTORCYCLE_CLASS = 3    
    # ViT Settings
    ENABLE_VIT_CLASSIFIER = True
    VIT_MODEL_NAME = os.path.join(BASE_DIR, "models", "vit_helmet_model")
    VIT_HEAD_CROP_RATIO = 0.30  # Top 30% of rider bbox for head/helmet check
    
    # Triple Riding Settings
    MAX_PERMITTED_RIDERS = 2
    TRIPLE_RIDING_IOU_THRESH = 0.25
    MIN_CONSECUTIVE_FRAMES_TRIPLE = 3  # Temporal debounce in video
    
    # Wrong-side Riding Settings
    # Permitted directions: 'auto' (automatic traffic consensus), 'downward', 'upward', etc.
    DEFAULT_PERMITTED_DIRECTION = 'auto'
    MIN_TRACK_DISPLACEMENT = 35.0  # Pixels displacement required before evaluating direction
    MIN_TRACK_FRAMES = 8          # Frames required to evaluate direction
    WRONG_SIDE_ANGLE_TOLERANCE = 100.0  # Degrees deviation from permitted direction
    
    # OCR Settings
    OCR_ENGINE = 'easyocr'  # 'easyocr' or 'tesseract'
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    MIN_OCR_CONFIDENCE = 0.30

    # Calibrated Per-Class Thresholds
    HELMET_CONF_THRESHOLD = float(os.environ.get('HELMET_CONF_THRESHOLD', 0.20))
    PLATE_CONF_THRESHOLD = float(os.environ.get('PLATE_CONF_THRESHOLD', 0.25))
    VEHICLE_CONF_THRESHOLD = float(os.environ.get('VEHICLE_CONF_THRESHOLD', 0.25))
    PERSON_CONF_THRESHOLD = float(os.environ.get('PERSON_CONF_THRESHOLD', 0.25))
    MIN_HEAD_PIXELS = 18

    # MongoDB Integration (Optional with SQLite fallback)
    MONGO_URI = os.environ.get('MONGO_URI', None)
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME', 'traffic_violation_db')

# Ensure all directories exist
for path in [
    Config.MODELS_DIR,
    Config.UPLOADS_DIR,
    Config.OUTPUTS_DIR,
    Config.OUTPUTS_IMAGES_DIR,
    Config.OUTPUTS_VIDEOS_DIR,
    Config.OUTPUTS_EVIDENCE_DIR,
    os.path.dirname(Config.DATABASE_PATH),
    Config.DATA_DIR,
    Config.SAMPLES_DIR
]:
    os.makedirs(path, exist_ok=True)
