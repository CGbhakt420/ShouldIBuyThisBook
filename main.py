from ml.ocr_pipeline import OCRPipeline

ocr = OCRPipeline()
titles = ocr.extract_text("sample_shelf.jpg")
print(titles)