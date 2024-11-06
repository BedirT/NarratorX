# narratorx/ocr.py

import pymupdf
from PIL import Image
from surya.model.detection.model import load_model as load_det_model
from surya.model.detection.model import load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr


def process_pdf(pdf_path, language):
    # Load the PDF
    doc = pymupdf.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    # Load models
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    # Run OCR
    langs = [language]
    predictions = run_ocr(
        images, [langs] * len(images), det_model, det_processor, rec_model, rec_processor
    )

    # Extract text
    pages_text = []
    for page_ocr_result in predictions:
        page_text = ""
        for line in page_ocr_result.text_lines:
            page_text += line.text + "\n"
        pages_text.append(page_text)

    # Combine pages
    full_text = "\n\n".join(pages_text)

    return full_text
