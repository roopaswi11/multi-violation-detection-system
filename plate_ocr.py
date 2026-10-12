"""
Standalone Plate OCR script.
Can be executed directly: python plate_ocr.py --image path/to/image.jpg
"""
import sys
import os
import argparse
import cv2

# Ensure services directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services.plate_service import PlateOCR

def main():
    parser = argparse.ArgumentParser(description="Number Plate OCR Extraction")
    parser.add_argument("--image", required=True, help="Path to input image containing number plate")
    parser.add_argument("--engine", default="easyocr", choices=["easyocr", "tesseract"], help="OCR engine")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found.")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Unable to load image from '{args.image}'.")
        sys.exit(1)

    ocr = PlateOCR(engine=args.engine)
    result = ocr.extract_text(img)
    print("--- Number Plate OCR Results ---")
    print(f"Raw Text:        {result['raw_text']}")
    print(f"Normalized Text: {result['normalized_text']}")
    print(f"Confidence:      {result['confidence']:.4f}")
    print(f"Status:          {result['status']}")

if __name__ == "__main__":
    main()
