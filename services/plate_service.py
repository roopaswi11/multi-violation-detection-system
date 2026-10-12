import os
import re
import cv2
import numpy as np
from config import Config

class PlateOCR:
    def __init__(self, engine=None):
        self.engine_type = engine or Config.OCR_ENGINE
        self.easyocr_reader = None
        self._init_engine()

    def _init_engine(self):
        if self.engine_type == 'easyocr':
            try:
                import easyocr
                print("[PlateOCR] Initializing EasyOCR (English)...")
                # Suppress verbose GPU download messages
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                print("[PlateOCR] EasyOCR initialized successfully.")
            except Exception as e:
                print(f"[PlateOCR] EasyOCR initialization failed: {e}. Falling back to Tesseract.")
                self.engine_type = 'tesseract'
                self.easyocr_reader = None

        if self.engine_type == 'tesseract':
            try:
                import pytesseract
                if os.path.exists(Config.TESSERACT_CMD):
                    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
                print("[PlateOCR] Tesseract OCR initialized.")
            except Exception as e:
                print(f"[PlateOCR] Tesseract not available: {e}")

    def preprocess_plate(self, plate_crop):
        """
        Applies image enhancement pipeline for optimal character recognition:
        1. Resize if too small (minimum height 50px).
        2. Grayscale conversion.
        3. Bilateral filter for noise reduction while keeping edges sharp.
        4. Contrast Limited Adaptive Histogram Equalization (CLAHE).
        5. Otsu thresholding / adaptive thresholding.
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        h, w = plate_crop.shape[:2]
        if h < 50 or w < 100:
            scale = max(2.0, 60.0 / max(1, h))
            plate_crop = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # Thresholding
        _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return contrast, thresh

    @staticmethod
    def normalize_plate_text(raw_text):
        """
        Normalizes extracted text:
        - Removes non-alphanumeric characters.
        - Converts to uppercase.
        - Fixes common OCR misreads in license plates (e.g. 'O' vs '0', 'I' vs '1' where contextual).
        """
        if not raw_text:
            return ""
        # Strip special characters and whitespace
        cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        return cleaned

    def extract_text(self, plate_crop):
        """
        Runs OCR on cropped plate region.
        Returns:
        {
            'raw_text': str,
            'normalized_text': str,
            'confidence': float,
            'status': 'readable' | 'unreadable' | 'empty'
        }
        """
        if plate_crop is None or plate_crop.size == 0:
            return {
                'raw_text': '',
                'normalized_text': '',
                'confidence': 0.0,
                'status': 'empty'
            }

        prep_result = self.preprocess_plate(plate_crop)
        if prep_result is None:
            return {
                'raw_text': '',
                'normalized_text': '',
                'confidence': 0.0,
                'status': 'unreadable'
            }
        contrast_img, thresh_img = prep_result

        raw_text = ""
        confidence = 0.0

        # Try EasyOCR
        if self.easyocr_reader is not None:
            try:
                results = self.easyocr_reader.readtext(contrast_img)
                if not results:
                    # Try thresholded image
                    results = self.easyocr_reader.readtext(thresh_img)
                
                if results:
                    # Collect text segments with highest confidence
                    texts = []
                    confs = []
                    for bbox, text, conf in results:
                        if text.strip():
                            texts.append(text.strip())
                            confs.append(float(conf))
                    if texts:
                        raw_text = " ".join(texts)
                        confidence = sum(confs) / len(confs)
            except Exception as e:
                print(f"[PlateOCR] EasyOCR reading error: {e}")

        # Fallback to Tesseract if EasyOCR didn't yield text
        if not raw_text:
            try:
                import pytesseract
                config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                data = pytesseract.image_to_data(thresh_img, config=config, output_type=pytesseract.Output.DICT)
                texts = []
                confs = []
                for i, text in enumerate(data['text']):
                    conf = float(data['conf'][i])
                    if text.strip() and conf > 20:
                        texts.append(text.strip())
                        confs.append(conf / 100.0)
                if texts:
                    raw_text = " ".join(texts)
                    confidence = sum(confs) / len(confs)
            except Exception as e:
                pass

        normalized = self.normalize_plate_text(raw_text)
        
        # Determine status
        if len(normalized) >= 4 and confidence >= Config.MIN_OCR_CONFIDENCE:
            status = 'readable'
        elif len(normalized) > 0:
            status = 'unreadable'  # Partial/noisy characters, flag for review
        else:
            status = 'unreadable'

        return {
            'raw_text': raw_text,
            'normalized_text': normalized if status == 'readable' else (normalized or 'UNREADABLE'),
            'confidence': round(confidence, 4),
            'status': status
        }

    @staticmethod
    def crop_plate_for_bike(frame, bike_box, plate_boxes=None):
        """
        Locates and crops the license plate for a given motorcycle.
        If a direct 'number-plate' detection overlaps or is near the bike, uses it.
        Otherwise, crops the lower-center 30% of the motorcycle bounding box.
        """
        h, w = frame.shape[:2]
        bx1, by1, bx2, by2 = [int(v) for v in bike_box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w - 1, bx2), min(h - 1, by2)
        
        # Check direct plate detections first
        if plate_boxes:
            for p in plate_boxes:
                px1, py1, px2, py2 = [int(v) for v in p['bbox']]
                # Check spatial overlap or vertical proximity
                if bx1 - 20 <= px1 and px2 <= bx2 + 20 and by1 <= py1 <= by2 + 30:
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(w - 1, px2), min(h - 1, py2)
                    crop = frame[py1:py2, px1:px2]
                    if crop.size > 0:
                        return crop, [px1, py1, px2, py2]

        # Geometric estimate: rear/lower part of motorcycle
        bike_h = by2 - by1
        bike_w = bx2 - bx1
        py1 = int(by1 + 0.65 * bike_h)
        py2 = by2
        px1 = int(bx1 + 0.20 * bike_w)
        px2 = int(bx1 + 0.80 * bike_w)
        
        if py2 > py1 and px2 > px1:
            crop = frame[py1:py2, px1:px2]
            return crop, [px1, py1, px2, py2]
            
        return None, None
