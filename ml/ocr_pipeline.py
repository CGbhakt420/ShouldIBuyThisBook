import cv2
import numpy as np
from PIL import Image
import pytesseract

class OCRPipeline:
    def __init__(self):
        pass
    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        threshold = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,2
        )
        
        denoised = cv2.fastNlMeansDenoising(threshold, h=30)
        return denoised
    
    def extract_text(self, img_path):
        processed = self.preprocess(img_path)
        pil_img = Image.fromarray(processed)

        raw_text = pytesseract.image_to_string(pil_img)
        lines = [
            line.strip()
            for line in raw_text.split("\n")
            if len(line.strip()) >= 3
        ]

        return lines