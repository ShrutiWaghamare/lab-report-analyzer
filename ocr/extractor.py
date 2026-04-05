"""
OCR Extractor Module — Multi-Strategy Extraction
=================================================
Tries every available extraction method in order of accuracy.
Automatically picks the best result.

STRATEGY ORDER (per input type):

  Digital PDF (Apollo, Thyrocare, SRL etc.)
  ─────────────────────────────────────────
  1. pdfplumber   → reads actual table cells → best column accuracy
  2. PyMuPDF      → reads text layer → good for non-table PDFs
  3. pdf2image + Tesseract → OCR fallback for image-only PDFs

  Scanned PDF / Phone photo PDF
  ─────────────────────────────
  1. PyMuPDF      → check if text layer exists
  2. pdf2image + OpenCV + Tesseract → full OCR pipeline

  Image file (JPG, PNG, TIFF etc.)
  ─────────────────────────────────
  1. OpenCV preprocessing → Tesseract OCR

  Plain text file
  ───────────────
  1. Read directly

WHY THIS ORDER:
  pdfplumber reads actual PDF table structure (row/col objects).
  Column misalignment is impossible — you get value column and
  reference column as separate strings. This fixes the triglycerides
  150 vs 375 bug that happened with text-layer extraction.

  PyMuPDF is fast and accurate for text-heavy PDFs without tables.

  Tesseract is the last resort — slowest and most error-prone,
  but handles any image-based input.

Install:
  pip install pdfplumber pymupdf pdf2image pillow pytesseract opencv-python-headless
  System: sudo apt-get install tesseract-ocr poppler-utils
          (Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki)
"""

import io
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ── optional imports — each strategy degrades gracefully ─────────────────────

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
    logger.info("[Extractor] pdfplumber available ✓")
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("[Extractor] pdfplumber not installed. Install: pip install pdfplumber")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("[Extractor] PyMuPDF available ✓")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("[Extractor] PyMuPDF not installed. Install: pip install pymupdf")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
    logger.info("[Extractor] Tesseract/Pillow available ✓")
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("[Extractor] pytesseract/Pillow not installed. Install: pip install pytesseract pillow")

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
    logger.info("[Extractor] pdf2image available ✓")
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("[Extractor] pdf2image not installed. Install: pip install pdf2image")

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
    logger.info("[Extractor] OpenCV available ✓")
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("[Extractor] OpenCV not installed. Install: pip install opencv-python-headless")

# ─────────────────────────────────────────────────────────────────────────────


class OCRExtractor:
    """
    Unified multi-strategy extractor for lab report documents.

    Tries strategies in order of accuracy and returns the best result.
    Logs which strategy was used so you can see what happened.
    """

    TESS_CONFIG = r"--oem 3 --psm 6"
    MIN_MEANINGFUL_CHARS = 100
    MIN_ALPHA_CHARS = 20

    def __init__(self, dpi: int = 300, language: str = "eng"):
        self.dpi = dpi
        self.language = language

    # ─────────────────────────────── public API ──────────────────────────────

    def extract(self, file_path: Union[str, Path]) -> str:
        """Extract text from a file path."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()

        if ext == ".txt":
            return self._read_text_file(path)
        elif ext == ".pdf":
            return self._extract_pdf_from_path(path)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            return self._extract_image_from_path(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def extract_from_bytes(self, file_bytes: bytes, file_ext: str) -> str:
        """Extract text from raw bytes (used by FastAPI file upload)."""
        ext = file_ext.lower()

        if ext == ".txt":
            return file_bytes.decode("utf-8", errors="replace").strip()
        elif ext == ".pdf":
            return self._extract_pdf_from_bytes(file_bytes)
        elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"):
            return self._extract_image_from_bytes(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def extract_from_string(self, raw_text: str) -> str:
        return raw_text.strip()

    # ─────────────────────────────── PDF strategies ──────────────────────────

    def _extract_pdf_from_path(self, path: Path) -> str:
        """
        Try all PDF strategies in order, return the best result.
        Strategy 1: pdfplumber (table-aware) — fixes column misalignment
        Strategy 2: PyMuPDF (text layer) — fast, good for non-table PDFs
        Strategy 3: pdf2image + Tesseract (OCR) — handles scanned/image PDFs
        """
        # Strategy 1 — pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._pdfplumber_extract_path(path)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 1 (pdfplumber) succeeded — {len(text)} chars")
                    return text
                logger.info("[Extractor] Strategy 1 (pdfplumber) returned empty — trying PyMuPDF")
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 1 (pdfplumber) failed: {e}")

        # Strategy 2 — PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                text = self._pymupdf_extract_path(path)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 2 (PyMuPDF) succeeded — {len(text)} chars")
                    return text
                logger.info("[Extractor] Strategy 2 (PyMuPDF) returned empty — trying Tesseract OCR")
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 2 (PyMuPDF) failed: {e}")

        # Strategy 3 — pdf2image + Tesseract
        if PDF2IMAGE_AVAILABLE and TESSERACT_AVAILABLE:
            try:
                text = self._tesseract_extract_path(path)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 3 (Tesseract OCR) succeeded — {len(text)} chars")
                    return text
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 3 (Tesseract OCR) failed: {e}")

        raise RuntimeError(
            "All PDF extraction strategies failed. "
            "Install: pip install pdfplumber pymupdf pdf2image pytesseract"
        )

    def _extract_pdf_from_bytes(self, file_bytes: bytes) -> str:
        """Same strategy order but from bytes."""

        # Strategy 1 — pdfplumber
        if PDFPLUMBER_AVAILABLE:
            try:
                text = self._pdfplumber_extract_bytes(file_bytes)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 1 (pdfplumber) succeeded — {len(text)} chars")
                    return text
                logger.info("[Extractor] Strategy 1 (pdfplumber) empty — trying PyMuPDF")
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 1 (pdfplumber) failed: {e}")

        # Strategy 2 — PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                text = self._pymupdf_extract_bytes(file_bytes)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 2 (PyMuPDF) succeeded — {len(text)} chars")
                    return text
                logger.info("[Extractor] Strategy 2 (PyMuPDF) empty — trying Tesseract OCR")
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 2 (PyMuPDF) failed: {e}")

        # Strategy 3 — pdf2image + Tesseract
        if PDF2IMAGE_AVAILABLE and TESSERACT_AVAILABLE:
            try:
                text = self._tesseract_extract_bytes(file_bytes)
                if self._is_meaningful(text):
                    logger.info(f"[Extractor] Strategy 3 (Tesseract OCR) succeeded — {len(text)} chars")
                    return text
            except Exception as e:
                logger.warning(f"[Extractor] Strategy 3 (Tesseract OCR) failed: {e}")

        raise RuntimeError("All PDF extraction strategies failed.")

    # ─────────────────────────────── Strategy 1: pdfplumber ──────────────────

    def _pdfplumber_extract_path(self, path: Path) -> str:
        with pdfplumber.open(str(path)) as pdf:
            return self._pdfplumber_process(pdf)

    def _pdfplumber_extract_bytes(self, file_bytes: bytes) -> str:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            return self._pdfplumber_process(pdf)

    def _pdfplumber_process(self, pdf) -> str:
        """
        Extract text from pdfplumber PDF object.

        For each page:
          1. Extract tables first — gives structured row/col data.
             Each row becomes a tab-separated line so the parser
             gets  TRIGLYCERIDES | 375 | mg/dL | <150 | Enzymatic
             instead of misreading the reference column as the value.
          2. Extract plain text for headers, patient info, notes.
          3. Combine both.
        """
        all_pages = []

        for page in pdf.pages:
            page_parts = []

            # Table extraction — structured row/col data
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            cleaned = [
                                str(cell).strip() if cell is not None else ""
                                for cell in row
                            ]
                            if any(c for c in cleaned):
                                # Tab-separated so parser can split columns correctly
                                page_parts.append("\t".join(cleaned))

            # Plain text — catches patient info, comments, headers
            plain_text = page.extract_text()
            if plain_text and plain_text.strip():
                page_parts.append(plain_text.strip())

            if page_parts:
                all_pages.append("\n".join(page_parts))

        return "\n\n".join(all_pages).strip()

    # ─────────────────────────────── Strategy 2: PyMuPDF ─────────────────────

    def _pymupdf_extract_path(self, path: Path) -> str:
        import fitz
        doc = fitz.open(str(path))
        return self._pymupdf_process(doc)

    def _pymupdf_extract_bytes(self, file_bytes: bytes) -> str:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return self._pymupdf_process(doc)

    def _pymupdf_process(self, doc) -> str:
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text()
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages).strip()

    # ─────────────────────────────── Strategy 3: Tesseract ───────────────────

    def _tesseract_extract_path(self, path: Path) -> str:
        images = convert_from_path(str(path), dpi=self.dpi)
        return self._tesseract_process_images(images)

    def _tesseract_extract_bytes(self, file_bytes: bytes) -> str:
        images = convert_from_bytes(file_bytes, dpi=self.dpi)
        return self._tesseract_process_images(images)

    def _tesseract_process_images(self, images: List) -> str:
        """Run Tesseract on a list of PIL images with preprocessing."""
        page_texts = []
        for i, img in enumerate(images):
            logger.info(f"[Extractor] OCR page {i + 1}/{len(images)}")
            processed = self._preprocess_image(img)
            text = pytesseract.image_to_string(
                processed, lang=self.language, config=self.TESS_CONFIG
            )
            page_texts.append(text)
        return "\n\n".join(page_texts).strip()

    # ─────────────────────────────── Image extraction ────────────────────────

    def _extract_image_from_path(self, path: Path) -> str:
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract/Pillow not installed.")
        img = Image.open(path)
        return self._ocr_single_image(img)

    def _extract_image_from_bytes(self, file_bytes: bytes) -> str:
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("pytesseract/Pillow not installed.")
        img = Image.open(io.BytesIO(file_bytes))
        return self._ocr_single_image(img)

    def _ocr_single_image(self, image) -> str:
        processed = self._preprocess_image(image)
        text = pytesseract.image_to_string(
            processed, lang=self.language, config=self.TESS_CONFIG
        )
        logger.info(f"[Extractor] Image OCR complete — {len(text)} chars")
        return text.strip()

    # ─────────────────────────────── Image preprocessing ─────────────────────

    def _preprocess_image(self, pil_image) -> "Image":
        """
        Preprocess image before Tesseract.
        Uses OpenCV pipeline if available (better quality).
        Falls back to Pillow if OpenCV not installed.
        """
        if OPENCV_AVAILABLE:
            return self._preprocess_opencv(pil_image)
        else:
            return self._preprocess_pillow(pil_image)

    def _preprocess_opencv(self, pil_image) -> "Image":
        """
        OpenCV preprocessing pipeline:
        1. Upscale if image is too small (< 1500px wide)
        2. Grayscale
        3. Shadow removal via morphological background estimation
        4. Denoise with Non-Local Means
        5. Adaptive threshold (binarise)
        Best for: phone photos, scanned documents with uneven lighting
        """
        rgb = pil_image.convert("RGB")
        img = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

        # Upscale if too small
        h, w = img.shape[:2]
        if w < 1500:
            scale = 1500 / w
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Shadow removal
        kernel_size = max(gray.shape[0] // 20, 51)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.dilate(gray, kernel)
        gray = cv2.divide(
            gray.astype("float32"),
            background.astype("float32"),
            scale=255.0
        ).astype("uint8")

        # Denoise
        gray = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # Adaptive threshold
        gray = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 12
        )

        return Image.fromarray(gray)

    def _preprocess_pillow(self, pil_image) -> "Image":
        """
        Pillow-only fallback preprocessing.
        Best for: clean digital scans where OpenCV is not installed.
        """
        from PIL import ImageFilter, ImageEnhance
        img = pil_image.convert("L")
        w, h = img.size
        if w < 1500:
            scale = 1500 / w
            img = img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS
            )
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return img

    # ─────────────────────────────── utilities ────────────────────────────────

    def _read_text_file(self, path: Path) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    def _is_meaningful(self, text: str) -> bool:
        """Check if extracted text has enough real content to be useful."""
        if not text:
            return False
        stripped = text.strip()
        alpha = sum(1 for c in stripped if c.isalpha())
        return (
            len(stripped) >= self.MIN_MEANINGFUL_CHARS
            and alpha >= self.MIN_ALPHA_CHARS
        )