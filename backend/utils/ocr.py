# backend/utils/ocr.py
import io
import os
from typing import List

import pytesseract
from PIL import Image



def _preprocess(img: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR on scans."""
    try:
        import numpy as np
        import cv2
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        return Image.fromarray(thr)
    except Exception:
        return img


def _ocr_pil(img: Image.Image) -> str:
    return pytesseract.image_to_string(img).strip()


def _pdf_text_pypdf(file_bytes: bytes) -> str:
    """Extract embedded (selectable) text from PDF using pypdf (no OCR)."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: List[str] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n".join(parts).strip()


def _pdf_pages_to_images_pymupdf(file_bytes: bytes, dpi: int = 220) -> List[Image.Image]:
    """Rasterize PDF pages to images using PyMuPDF (no Poppler needed)."""
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def ocr_pdf_bytes(file_bytes: bytes) -> str:
    """
    Strategy:
      1) Try pypdf to extract embedded text (fast & clean).
      2) If empty/very short => assume scanned PDF; rasterize with PyMuPDF and OCR with Tesseract.
    """
    # 1) pypdf
    try:
        text = _pdf_text_pypdf(file_bytes)
        if len(text.strip()) >= 30:  # heuristic threshold
            return text
    except Exception:
        pass

    # 2) OCR fallback
    pages = _pdf_pages_to_images_pymupdf(file_bytes, dpi=220)
    texts: List[str] = []
    for img in pages:
        pre = _preprocess(img)
        texts.append(_ocr_pil(pre))
    return "\n".join(texts).strip()


def ocr_image_bytes(file_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(file_bytes))
    img = _preprocess(img)
    return _ocr_pil(img)


def ocr_any(uploaded_bytes: bytes, filename: str = "", content_type: str = "") -> str:
    """Auto-detect PDF vs image based on MIME or extension."""
    ext = os.path.splitext(filename or "")[1].lower()
    mime = (content_type or "").lower()
    if "pdf" in mime or ext == ".pdf":
        return ocr_pdf_bytes(uploaded_bytes)
    return ocr_image_bytes(uploaded_bytes)