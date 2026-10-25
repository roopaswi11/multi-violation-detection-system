import os
import re
import cv2
import uuid
import numpy as np
from config import Config

# Standard Indian States and Union Territories (statutory RTO codes)
INDIAN_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN', 'MP',
    'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

# Contextual character disambiguation maps for high-security registration plates (HSRP)
DIGIT_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
LETTER_TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'E': '3', 'A': '4', 'Y': '4', 'S': '5', 'G': '6', 'B': '8'}


class PlateOCR:
    """
    Production-grade License Plate Detection and Multi-Stage OCR Pipeline.
    Supports:
    - Motorcycle, scooter, car, bus, truck, and autorickshaw candidate localization.
    - Two-line stacked plate splitting (motorcycles) and single-line plate handling (cars).
    - Image preprocessing variants: Bilateral filtering, CLAHE, Adaptive Gaussian thresholding, and sharpening.
    - Spatial text-box line ordering (top-to-bottom, left-to-right).
    - India-aware statutory registration format validation & candidate scoring.
    - Auditable plate evidence cropping and persistence.
    """
    def __init__(self, engine=None):
        self.engine_type = engine or Config.OCR_ENGINE
        self.easyocr_reader = None
        self._init_engine()

    def _init_engine(self):
        if self.engine_type == 'easyocr':
            try:
                import easyocr
                print("[PlateOCR] Initializing EasyOCR (English)...")
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

    @staticmethod
    def order_text_segments(results):
        """
        Groups OCR bounding boxes into lines based on vertical proximity,
        and sorts each line from left to right.
        """
        if not results:
            return []
        items = []
        for r in results:
            if len(r) == 3:
                bbox, text, conf = r
            elif len(r) == 2:
                bbox, text = r
                conf = 0.5
            else:
                continue

            clean_t = str(text).strip()
            if not clean_t:
                continue

            try:
                ys = [pt[1] for pt in bbox]
                xs = [pt[0] for pt in bbox]
                min_y, max_y = min(ys), max(ys)
                min_x, max_x = min(xs), max(xs)
                cy = (min_y + max_y) / 2.0
                cx = (min_x + max_x) / 2.0
                h = max(1.0, max_y - min_y)
            except Exception:
                cy, cx, h = 0.0, 0.0, 10.0

            items.append({
                'text': clean_t,
                'conf': float(conf),
                'cx': cx,
                'cy': cy,
                'h': h
            })

        if not items:
            return []

        # Sort initially by vertical center (cy)
        items.sort(key=lambda it: it['cy'])
        lines = []
        curr_line = [items[0]]
        curr_cy = items[0]['cy']
        avg_h = items[0]['h']

        for it in items[1:]:
            if abs(it['cy'] - curr_cy) < max(12.0, avg_h * 0.55):
                curr_line.append(it)
                curr_cy = sum(x['cy'] for x in curr_line) / len(curr_line)
            else:
                lines.append(curr_line)
                curr_line = [it]
                curr_cy = it['cy']
                avg_h = it['h']
        lines.append(curr_line)

        ordered = []
        for line in lines:
            line.sort(key=lambda it: it['cx'])
            ordered.extend(line)
        return ordered

    @staticmethod
    def validate_and_normalize_indian_plate(raw_text, ocr_conf=0.0):
        """
        Statutory format validator and candidate ranker for Indian registration plates:
        - Normalizes alphanumeric characters.
        - Trims leading border / hologram artifacts ('IND', stray border marks).
        - Validates standard statutory format: [State 2-char][RTO 1-2 digits][Series 0-3 chars][Number 1-4 digits]
        - Validates Bharat Series (BH): [Year 2 digits] BH [Number 4 digits] [Series 1-2 chars]
        - Performs contextual character disambiguation (O/0, I/1, S/5, B/8, A/4).
        """
        if not raw_text:
            return {
                'text': 'UNREADABLE',
                'clean': '',
                'raw': '',
                'confidence': 0.0,
                'status': 'unreadable',
                'is_valid_format': False,
                'score': 0.0
            }

        clean = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
        if not clean or len(clean) < 3 or (ocr_conf is not None and ocr_conf < 0.12 and len(clean) < 6):
            return {
                'text': 'UNREADABLE',
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'unreadable',
                'is_valid_format': False,
                'score': 0.0
            }

        # Trim hologram / country code border artifacts
        if len(clean) >= 5 and clean.startswith('IND'):
            clean = clean[3:]
        elif len(clean) >= 6 and clean[1:3] in INDIAN_STATE_CODES and clean[3:5].isdigit():
            clean = clean[1:]
        # Resolve state code prefix using statutory RTO codes and OCR confusion mapping
        if len(clean) >= 4:
            cand_state = PlateOCR.resolve_state_code(clean[:2])
            if cand_state in INDIAN_STATE_CODES:
                clean = cand_state + clean[2:]

        # 1. Check Bharat Series (BH)
        m_bh = re.match(r'^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$', clean)
        if m_bh:
            yr, bh, num, series = m_bh.groups()
            formatted = f"{yr} BH {num} {series}"
            return {
                'text': formatted,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'readable' if (ocr_conf or 0.0) >= 0.20 else 'partial',
                'is_valid_format': True,
                'score': 0.90 + (ocr_conf or 0.0) * 0.10
            }

        # 2. Check Direct Standard Indian Format
        # A. Format with series: State(2) + District(1-2) + Series(1-3) + Number(1-4)
        m_std = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$', clean)
        if m_std:
            state, dist, series, num = m_std.groups()
            is_valid_state = state in INDIAN_STATE_CODES
            formatted = f"{state} {dist} {series} {num}".strip().replace("  ", " ")
            has_full_num = len(num) >= 3
            is_readable = is_valid_state and (has_full_num or (ocr_conf or 0.0) >= 0.35)
            score = (ocr_conf or 0.0) * 0.35 + (0.45 if is_valid_state else 0.20) + (0.20 if has_full_num else 0.10)
            return {
                'text': formatted,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'readable' if is_readable else 'partial',
                'is_valid_format': is_valid_state,
                'score': score
            }

        # B. Format without series: State(2) + District(2) + Number(3-4)
        m_noser = re.match(r'^([A-Z]{2})([0-9]{2})([0-9]{3,4})$', clean)
        if m_noser:
            state, dist, num = m_noser.groups()
            is_valid_state = state in INDIAN_STATE_CODES
            formatted = f"{state} {dist} {num}".strip().replace("  ", " ")
            has_full_num = len(num) >= 3
            is_readable = is_valid_state and (has_full_num or (ocr_conf or 0.0) >= 0.35)
            score = (ocr_conf or 0.0) * 0.35 + (0.45 if is_valid_state else 0.20) + (0.20 if has_full_num else 0.10)
            return {
                'text': formatted,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'readable' if is_readable else 'partial',
                'is_valid_format': is_valid_state,
                'score': score
            }

        # C. Partial State + District code (e.g. TS 08, MH 12, DL 04)
        m_dist_prefix = re.match(r'^([A-Z]{2})([0-9]{1,2})$', clean)
        if m_dist_prefix:
            state, dist = m_dist_prefix.groups()
            is_valid_state = state in INDIAN_STATE_CODES
            formatted = f"{state} {dist}".strip()
            score = 0.40 + (ocr_conf or 0.0) * 0.35
            return {
                'text': formatted,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'partial',
                'is_valid_format': is_valid_state,
                'score': score
            }

        # 3. Contextual character disambiguation for 6 to 11 character candidates
        if 6 <= len(clean) <= 11:
            chars = list(clean)
            for i in [0, 1]:
                if chars[i] in DIGIT_TO_LETTER:
                    chars[i] = DIGIT_TO_LETTER[chars[i]]
            for i in [2, 3]:
                if i < len(chars) and chars[i] in LETTER_TO_DIGIT:
                    chars[i] = LETTER_TO_DIGIT[chars[i]]
            for i in range(max(4, len(chars) - 4), len(chars)):
                if chars[i] in LETTER_TO_DIGIT:
                    chars[i] = LETTER_TO_DIGIT[chars[i]]

            corrected = "".join(chars)
            m_corr = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{1,4})$', corrected)
            if not m_corr:
                m_corr = re.match(r'^([A-Z]{2})([0-9]{2})([0-9]{3,4})$', corrected)
            if m_corr:
                groups = m_corr.groups()
                if len(groups) == 4:
                    state, dist, series, num = groups
                    formatted = f"{state} {dist} {series} {num}".strip().replace("  ", " ")
                else:
                    state, dist, num = groups
                    formatted = f"{state} {dist} {num}".strip().replace("  ", " ")
                is_valid_state = state in INDIAN_STATE_CODES
                has_full_num = len(num) >= 3
                is_readable = is_valid_state and (has_full_num or (ocr_conf or 0.0) >= 0.35)
                score = (ocr_conf or 0.0) * 0.30 + (0.40 if is_valid_state else 0.15) + (0.20 if has_full_num else 0.10)
                return {
                    'text': formatted,
                    'clean': corrected,
                    'raw': raw_text,
                    'confidence': round(float(ocr_conf or 0.0), 4),
                    'status': 'readable' if is_readable else 'partial',
                    'is_valid_format': is_valid_state,
                    'score': score
                }

        # 4. State-code prefixed alphanumeric plate (e.g. MH VAAA 0000, MH 44 AA 0000)
        if len(clean) >= 6 and clean[:2] in INDIAN_STATE_CODES:
            state = clean[:2]
            rem = clean[2:]
            rem_chars = list(rem)
            for i in range(max(0, len(rem_chars) - 4), len(rem_chars)):
                if rem_chars[i] in LETTER_TO_DIGIT:
                    rem_chars[i] = LETTER_TO_DIGIT[rem_chars[i]]
            rem_fixed = "".join(rem_chars)

            m_split = re.match(r'^(.*?)(\d{1,4})$', rem_fixed)
            if m_split:
                mid, end_num = m_split.groups()
                formatted = f"{state} {mid} {end_num}".strip().replace("  ", " ")
            else:
                formatted = f"{state} {rem_fixed}".strip()

            score = 0.50 + (ocr_conf or 0.0) * 0.40
            return {
                'text': formatted,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'readable' if (ocr_conf or 0.0) >= 0.35 else 'partial',
                'is_valid_format': True,
                'score': score
            }

        # 5. Check partial State + District code candidate (e.g. TS 08, MH 12, DL 4C)
        m_dist = re.match(r'^([A-Z]{2})([0-9]{1,2})([A-Z]{0,2})$', clean)
        if m_dist:
            state, dist, tail = m_dist.groups()
            if state in INDIAN_STATE_CODES:
                formatted = f"{state} {dist} {tail}".strip()
                return {
                    'text': formatted,
                    'clean': clean,
                    'raw': raw_text,
                    'confidence': round(float(ocr_conf or 0.0), 4),
                    'status': 'partial',
                    'is_valid_format': True,
                    'score': 0.40 + (ocr_conf or 0.0) * 0.35
                }

        # 6. General partial candidate check
        if len(clean) >= 4 and (ocr_conf or 0.0) >= 0.20:
            spaced = clean
            if len(clean) >= 6 and clean[:2].isalpha() and not clean[2:].isalpha():
                m_p = re.match(r'^([A-Z]{2,4})(\d+.*)$', clean)
                if m_p:
                    spaced = f"{m_p.group(1)} {m_p.group(2)}"
            return {
                'text': spaced,
                'clean': clean,
                'raw': raw_text,
                'confidence': round(float(ocr_conf or 0.0), 4),
                'status': 'partial',
                'is_valid_format': False,
                'score': 0.25 + (ocr_conf or 0.0) * 0.25
            }

        # 6. Otherwise unreadable
        return {
            'text': 'UNREADABLE',
            'clean': clean,
            'raw': raw_text,
            'confidence': round(float(ocr_conf or 0.0), 4),
            'status': 'unreadable',
            'is_valid_format': False,
            'score': 0.0
        }

    @staticmethod
    def resolve_state_code(prefix):
        """Resolves OCR misread prefix to statutory Indian state code."""
        if not prefix or len(prefix) < 2:
            return prefix
        c1, c2 = prefix[0].upper(), prefix[1].upper()
        direct = c1 + c2
        if direct in INDIAN_STATE_CODES:
            return direct

        # Common digit/letter substitutions for position 0
        subs0 = [c1]
        if c1 in ['0', 'O', 'Q', 'D']:
            subs0.extend(['T', 'D', 'M', 'O', 'U'])
        elif c1 in ['1', 'I', 'L', '7', 'T']:
            subs0.extend(['T', 'J', 'I', 'U'])
        elif c1 in ['4', 'A']:
            subs0.extend(['A'])
        elif c1 in ['8', 'B']:
            subs0.extend(['B', 'R'])
        elif c1 in ['6', 'G']:
            subs0.extend(['G'])
        elif c1 in ['5', 'S']:
            subs0.extend(['S'])

        # Common substitutions for position 1
        subs1 = [c2]
        if c2 in ['0', 'O', 'D']:
            subs1.extend(['D', 'O', 'B', 'P'])
        elif c2 in ['5', 'S']:
            subs1.extend(['S'])
        elif c2 in ['8', 'B']:
            subs1.extend(['R', 'B'])
        elif c2 in ['1', 'I', 'L']:
            subs1.extend(['L', 'I', 'J', 'K'])
        elif c2 in ['4', 'A']:
            subs1.extend(['A', 'H'])

        for s0 in subs0:
            for s1 in subs1:
                cand = s0 + s1
                if cand in INDIAN_STATE_CODES:
                    return cand

        return direct

    @staticmethod
    def compute_box_iou(box1, box2):
        """Computes intersection-over-union between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        a2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        union = a1 + a2 - inter
        return float(inter) / float(union)

    @staticmethod
    def deskew_crop(crop):
        """Deskews slightly tilted plates using minimum area bounding rectangle."""
        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 30:
            return crop
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            coords = np.column_stack(np.where(thresh > 0))
            if len(coords) < 50:
                return crop
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) > 2.5 and abs(angle) < 30.0:
                h, w = crop.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(crop, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
        except Exception:
            pass
        return crop

    def preprocess_variants(self, plate_crop):
        """
        Creates image enhancement variants:
        - Upscales crop using Lanczos4 (ensuring characters are at least 30px high).
        - Deskews tilted plates.
        - Bilateral filtered CLAHE.
        - Unsharp masking (sharpening).
        - Adaptive Gaussian thresholding.
        - Otsu thresholding.
        """
        if plate_crop is None or plate_crop.size == 0:
            return []

        # Deskew if tilted
        deskewed = self.deskew_crop(plate_crop)

        h, w = deskewed.shape[:2]
        # CRAFT detector in EasyOCR requires characters to be sufficiently large
        scale = max(3.5, 140.0 / max(1, h))
        target_w = int(w * scale)
        target_h = int(h * scale)

        up = cv2.resize(deskewed, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

        variants = []

        # 1. Bilateral + CLAHE (primary contrast enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        contrast = clahe.apply(denoised)
        variants.append(('bi_clahe', contrast))

        # 2. Unsharp Masking (stroke sharpening)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        variants.append(('sharp', sharp))

        # 3. Adaptive Gaussian Thresholding
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
        variants.append(('adapt', adapt))

        # 4. Otsu Thresholding
        _, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(('otsu', otsu))

        # 5. Raw Grayscale (unmodified upscaled)
        variants.append(('raw_gray', gray))

        return variants

    def locate_plate_candidates(self, frame, vehicle_box, direct_plates=None, vehicle_type='motorcycle'):
        """
        Locates candidate plate regions for a vehicle:
        1. Checks direct YOLO plate detections with margin padding and IoU / proximity to vehicle.
        2. Generates domain-aware vehicle candidate zones (front mount, rear mount, fender, bumper).
        3. Applies multi-level threshold contour detection inside candidate zones to isolate bright plate rectangles.
        """
        candidates = []
        if frame is None or vehicle_box is None or len(vehicle_box) < 4:
            return candidates

        fh, fw = frame.shape[:2]
        bx1, by1, bx2, by2 = [int(v) for v in vehicle_box]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(fw, bx2), min(fh, by2)
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # 1. Check direct plate detections
        if direct_plates and isinstance(direct_plates, (list, tuple)):
            for p in direct_plates:
                if not isinstance(p, dict) or 'bbox' not in p:
                    continue
                px1, py1, px2, py2 = [int(v) for v in p['bbox']]
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(fw, px2), min(fh, py2)
                pw = px2 - px1
                ph = py2 - py1
                if pw <= 5 or ph <= 5:
                    continue

                pad_x = max(6, int(pw * 0.14))
                pad_y = max(5, int(ph * 0.14))
                c_px1 = max(0, px1 - pad_x)
                c_py1 = max(0, py1 - pad_y)
                c_px2 = min(fw, px2 + pad_x)
                c_py2 = min(fh, py2 + pad_y)

                cx = (px1 + px2) / 2.0
                cy = (py1 + py2) / 2.0
                if (bx1 - 35 <= cx <= bx2 + 35) and (by1 - 25 <= cy <= by2 + 50):
                    crop = frame[c_py1:c_py2, c_px1:c_px2]
                    if crop.size > 0:
                        candidates.append((crop, [c_px1, c_py1, c_px2, c_py2], 'direct_yolo'))

                        # If direct plate detection is excessively tall (aspect < 1.15) or large,
                        # also isolate internal rectangular plate contour
                        aspect = float(pw) / max(1, ph)
                        if aspect < 1.20 or ph > 0.35 * bh:
                            zgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            for tval in [130, 160, 100]:
                                _, zthresh = cv2.threshold(zgray, tval, 255, cv2.THRESH_BINARY)
                                zcnts, _ = cv2.findContours(zthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for zc in zcnts:
                                    zcx, zcy, zcw, zch = cv2.boundingRect(zc)
                                    z_aspect = float(zcw) / max(1, zch)
                                    if 1.3 <= z_aspect <= 6.5 and zcw * zch >= 300:
                                        r_px1 = max(0, c_px1 + zcx - 4)
                                        r_py1 = max(0, c_py1 + zcy - 4)
                                        r_px2 = min(fw, c_px1 + zcx + zcw + 4)
                                        r_py2 = min(fh, c_py1 + zcy + zch + 4)
                                        rcrop = frame[r_py1:r_py2, r_px1:r_px2]
                                        if rcrop.size > 0:
                                            candidates.append((rcrop, [r_px1, r_py1, r_px2, r_py2], 'direct_refined_contour'))

        # 2. Geometric zones based on vehicle type
        vtype = str(vehicle_type).lower().replace('_', ' ').strip()
        is_two_wheeler = vtype in ['motorcycle', 'bike', 'scooter', 'two wheeler', 'two_wheeler']

        if is_two_wheeler:
            zones = [
                # Front mount zone (below handlebar/headlight to front wheel hub)
                (max(0, int(by1 + 0.12 * bh)), min(fh, int(by1 + 0.65 * bh)), max(0, int(bx1 + 0.04 * bw)), min(fw, int(bx2 - 0.04 * bw))),
                # Rear mount zone (rear fender/mudguard)
                (max(0, int(by1 + 0.40 * bh)), min(fh, int(by2 + 0.05 * bh)), max(0, int(bx1 + 0.04 * bw)), min(fw, int(bx2 - 0.04 * bw))),
                # Full mid-to-lower vehicle zone
                (max(0, int(by1 + 0.15 * bh)), by2, max(0, bx1), min(fw, bx2))
            ]
        else:
            zones = [
                # Front bumper / grille
                (max(0, int(by1 + 0.45 * bh)), by2, max(0, int(bx1 + 0.05 * bw)), min(fw, int(bx2 - 0.05 * bw))),
                # Rear bumper / trunk
                (max(0, int(by1 + 0.35 * bh)), by2, max(0, int(bx1 + 0.05 * bw)), min(fw, int(bx2 - 0.05 * bw))),
                # Full lower zone
                (max(0, int(by1 + 0.30 * bh)), by2, max(0, bx1), min(fw, bx2))
            ]

        raw_contour_candidates = []
        for zy1, zy2, zx1, zx2 in zones:
            if zy2 <= zy1 or zx2 <= zx1:
                continue
            zcrop = frame[zy1:zy2, zx1:zx2]
            if zcrop.size == 0:
                continue

            # 3. Contour refinement inside zone: search for bright rectangular plate (white / yellow background)
            gray = cv2.cvtColor(zcrop, cv2.COLOR_BGR2GRAY)

            for thresh_val in [140, 160, 120, 100]:
                _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    area = cw * ch
                    aspect = float(cw) / max(1, ch)
                    if 1.3 <= aspect <= 7.0 and area >= 300 and cw >= 35 and ch >= 12:
                        pad = max(4, int(ch * 0.12))
                        px1 = max(0, zx1 + cx - pad)
                        py1 = max(0, zy1 + cy - pad)
                        px2 = min(fw, zx1 + cx + cw + pad)
                        py2 = min(fh, zy1 + cy + ch + pad)
                        refined = frame[py1:py2, px1:px2]
                        if refined.size > 0:
                            aspect_diff = min(abs(aspect - 4.5), abs(aspect - 2.0))
                            cscore = -aspect_diff * 2.0 + min(area, 10000) / 2000.0
                            raw_contour_candidates.append({
                                'crop': refined,
                                'box': [px1, py1, px2, py2],
                                'ctype': f'contour_t{thresh_val}',
                                'score': cscore
                            })

        # Sort contour candidates by score descending
        raw_contour_candidates.sort(key=lambda x: x['score'], reverse=True)

        final_candidates = list(candidates) # Direct YOLO candidates already included
        for cc in raw_contour_candidates:
            overlap = False
            for existing in final_candidates:
                if self.compute_box_iou(cc['box'], existing[1]) > 0.55:
                    overlap = True
                    break
            if not overlap:
                final_candidates.append((cc['crop'], cc['box'], cc['ctype']))
                if len(final_candidates) >= 4:
                    break

        if not final_candidates:
            for zy1, zy2, zx1, zx2 in zones[:1]:
                zcrop = frame[zy1:zy2, zx1:zx2]
                if zcrop.size > 0:
                    final_candidates.append((zcrop, [zx1, zy1, zx2, zy2], 'zone_estimate'))

        return final_candidates

    def extract_text(self, plate_crop):
        """
        Multi-stage OCR extraction on a cropped plate:
        - Runs preprocessed variants through EasyOCR / Tesseract.
        - Evaluates both full crop and two-line split (for motorcycle stacked plates).
        - Performs spatial text line ordering.
        - Validates through India-aware registration format validator.
        """
        if plate_crop is None or plate_crop.size == 0:
            return {
                'raw_text': '',
                'normalized_text': 'UNREADABLE',
                'confidence': 0.0,
                'status': 'unreadable'
            }

        ch, cw = plate_crop.shape[:2]
        crop_aspect = float(cw) / max(1, ch)
        variants = self.preprocess_variants(plate_crop)
        candidate_readings = []

        # 1. Evaluate with EasyOCR
        if self.easyocr_reader is not None:
            for vname, vimg in variants:
                try:
                    vh, vw = vimg.shape[:2]
                    # A. Full image pass
                    ocr_full = self.easyocr_reader.readtext(vimg, detail=1)
                    if ocr_full:
                        ordered = self.order_text_segments(ocr_full)
                        if ordered:
                            raw_text = " ".join(o['text'] for o in ordered)
                            avg_conf = sum(o['conf'] for o in ordered) / len(ordered)
                            val = self.validate_and_normalize_indian_plate(raw_text, avg_conf)
                            val['variant'] = f"full_{vname}"
                            candidate_readings.append(val)
                            if val.get('is_valid_format') and val.get('status') == 'readable' and avg_conf >= 0.70:
                                break

                    # B. Two-line split pass for stacked plates (aspect < 2.4)
                    if crop_aspect < 2.4 and vh >= 60:
                        top_img = vimg[0 : int(0.55 * vh), :]
                        bot_img = vimg[int(0.45 * vh) : vh, :]
                        r_top = self.easyocr_reader.readtext(top_img, detail=1)
                        r_bot = self.easyocr_reader.readtext(bot_img, detail=1)

                        ord_top = self.order_text_segments(r_top)
                        ord_bot = self.order_text_segments(r_bot)

                        if ord_top or ord_bot:
                            all_segs = ord_top + ord_bot
                            two_line_text = " ".join(s['text'] for s in all_segs)
                            two_line_conf = sum(s['conf'] for s in all_segs) / len(all_segs)
                            val_split = self.validate_and_normalize_indian_plate(two_line_text, two_line_conf)
                            val_split['variant'] = f"twoline_{vname}"
                            candidate_readings.append(val_split)
                except Exception as e:
                    pass

        # 2. Evaluate with Tesseract as fallback
        if not candidate_readings:
            try:
                import pytesseract
                for vname, vimg in variants:
                    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    data = pytesseract.image_to_data(vimg, config=config, output_type=pytesseract.Output.DICT)
                    texts = []
                    confs = []
                    for i, t in enumerate(data['text']):
                        c = float(data['conf'][i])
                        if t.strip() and c > 20:
                            texts.append(t.strip())
                            confs.append(c / 100.0)
                    if texts:
                        raw_text = " ".join(texts)
                        avg_conf = sum(confs) / len(confs)
                        val = self.validate_and_normalize_indian_plate(raw_text, avg_conf)
                        val['variant'] = f"tess_{vname}"
                        candidate_readings.append(val)
            except Exception:
                pass

        if not candidate_readings:
            return {
                'raw_text': '',
                'normalized_text': 'UNREADABLE',
                'confidence': 0.0,
                'status': 'unreadable'
            }

        # Select highest-scoring candidate reading
        candidate_readings.sort(key=lambda x: x['score'], reverse=True)
        best = candidate_readings[0]

        return {
            'raw_text': best['raw'],
            'normalized_text': best['text'],
            'confidence': best['confidence'],
            'status': best['status']
        }

    def extract_plate_from_vehicle(self, frame, vehicle_box, direct_plates=None, vehicle_type='motorcycle', evidence_dir=None):
        """
        Complete end-to-end multi-stage pipeline:
        1. Identifies candidate plate regions.
        2. Evaluates OCR on each candidate with spatial line ordering and statutory format validation.
        3. Saves evidence crop to evidence directory.
        4. Returns best validated reading.
        """
        evidence_dir = evidence_dir or Config.OUTPUTS_EVIDENCE_DIR
        os.makedirs(evidence_dir, exist_ok=True)

        if isinstance(direct_plates, str):
            vehicle_type = direct_plates
            direct_plates = None

        candidates = self.locate_plate_candidates(frame, vehicle_box, direct_plates=direct_plates, vehicle_type=vehicle_type)
        if not candidates:
            return {
                'raw_text': '',
                'normalized_text': 'UNREADABLE',
                'confidence': 0.0,
                'status': 'unreadable',
                'bbox': None,
                'crop_path': None
            }

        evaluated_candidates = []
        for crop, box, ctype in candidates:
            res = self.extract_text(crop)
            score = 0.0
            if res['status'] == 'readable':
                score = 0.65 + res['confidence'] * 0.35
            elif res['status'] == 'partial':
                score = 0.35 + res['confidence'] * 0.30
            else:
                score = 0.05

            if 'direct' in ctype:
                score += 0.10
            elif 'contour' in ctype:
                score += 0.05

            evaluated_candidates.append({
                'crop': crop,
                'box': box,
                'ctype': ctype,
                'res': res,
                'score': score
            })
            if res.get('status') == 'readable' and res.get('confidence', 0.0) >= 0.75:
                break

        evaluated_candidates.sort(key=lambda x: x['score'], reverse=True)
        best = evaluated_candidates[0]

        # Save evidence crop
        crop_path = None
        best_crop = best['crop']
        if best_crop is not None and best_crop.size > 0:
            uid = uuid.uuid4().hex[:8]
            fname = f"plate_{uid}.jpg"
            abs_p = os.path.join(evidence_dir, fname)
            cv2.imwrite(abs_p, best_crop)
            crop_path = f"/outputs/evidence/{fname}"

        return {
            'raw_text': best['res']['raw_text'],
            'normalized_text': best['res']['normalized_text'],
            'confidence': best['res']['confidence'],
            'status': best['res']['status'],
            'bbox': best['box'],
            'crop_path': crop_path
        }

    @staticmethod
    def crop_plate_for_bike(frame, bike_box, plate_boxes=None):
        """
        Backward-compatible helper: locates and returns candidate plate crop + bbox.
        """
        ocr_helper = PlateOCR.__new__(PlateOCR)
        candidates = ocr_helper.locate_plate_candidates(frame, bike_box, direct_plates=plate_boxes, vehicle_type='motorcycle')
        if candidates:
            for crop, box, ctype in candidates:
                if 'contour' in ctype:
                    return crop, box
            return candidates[0][0], candidates[0][1]
        return None, None
