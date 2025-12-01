import cv2
import numpy as np
from PIL import Image
import pytesseract
import re 

class OCRPipeline:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )
        
        denoised = cv2.fastNlMeansDenoising(thresh, h=30)
        return denoised

    def clean_garbage(self, raw_lines):
        clean_lines = []
        for line in raw_lines:
            cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', line).strip()
            if len(cleaned) < 4:
                continue
            if not re.search(r'[aeiouAEIOU]', cleaned):
                continue
            if cleaned.isdigit():
                continue

            clean_lines.append(cleaned)
            
        return list(set(clean_lines)) 

    def get_text_from_image(self, processed_img):
        pil_img = Image.fromarray(processed_img)
        config = r'--psm 6' 
        raw_text = pytesseract.image_to_string(pil_img, config=config)
        return raw_text.split("\n")

    def extract_text(self, img_path):
        original_img = cv2.imread(img_path)
        rotated_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        all_raw_lines = []

        print("Scanning horizontal text...")
        processed_original = self.preprocess(original_img)
        all_raw_lines.extend(self.get_text_from_image(processed_original))

        print("Scanning vertical text...")
        processed_rotated = self.preprocess(rotated_img)
        all_raw_lines.extend(self.get_text_from_image(processed_rotated))
        final_clean_titles = self.clean_garbage(all_raw_lines)
        return final_clean_titles


if __name__ == "__main__":
    ocr = OCRPipeline()
    titles = ocr.extract_text("sample_shelf.jpg")

    print("\n--- Cleaned Book Titles ---")
    for title in titles:
        print(title)