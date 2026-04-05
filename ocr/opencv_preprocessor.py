"""
OpenCV Image Preprocessor for Lab Reports
==========================================
Applies a preprocessing pipeline specifically tuned for lab report images
before they reach Tesseract OCR.

WHY OpenCV here (vs just Pillow):
  - Pillow: basic filters only (sharpen, contrast boost). Good for clean digital scans.
  - OpenCV: needed when the image has real-world problems:
      * Deskew        — report photographed at an angle (very common with phone cameras)
      * Shadow removal — uneven lighting from phone flash or overhead light hitting paper
      * Adaptive threshold — handles dark background / uneven brightness across the page
      * Noise removal — salt-and-pepper noise from old scanners or low-quality prints
      * Border crop    — removes black scanner borders that confuse Tesseract
      * Table detection— isolates tabular regions so the parser can handle them separately

This module is called by OCRExtractor before pytesseract runs.
Pillow preprocessing is the fallback when OpenCV is not installed.

Install:
    pip install opencv-python-headless numpy

System: no extra system packages needed (unlike poppler/tesseract).

Pipeline (applied in order):
    1. Load image as numpy array (BGR)
    2. Remove shadow / correct uneven illumination
    3. Convert to grayscale
    4. Deskew  (correct rotation)
    5. Denoise (remove scan noise)
    6. Adaptive threshold (binarise cleanly)
    7. Remove thin border artifacts
    8. Return as PIL Image for Tesseract

Usage:
    from ocr.opencv_preprocessor import OpenCVPreprocessor
    preprocessor = OpenCVPreprocessor()
    pil_image = preprocessor.preprocess(pil_image)   # drop-in before pytesseract
"""

import logging
import numpy as np
from typing import Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning(
        "OpenCV not installed. Install with: pip install opencv-python-headless\n"
        "Falling back to Pillow-only preprocessing."
    )


class OpenCVPreprocessor:
    """
    Full OpenCV preprocessing pipeline for lab report images.

    Designed for three real-world input types from Praveen Sehgal's report:
      Type A — Clean digital PDF page (pages 2-7): minimal processing needed
      Type B — Scanned document with slight skew / noise (pages 3, 5): deskew + denoise
      Type C — Phone photo of physical document (page 1 Aadhaar): full pipeline including
               shadow removal and aggressive threshold
    """

    def __init__(
        self,
        auto_detect_type: bool = True,
        deskew: bool = True,
        remove_shadow: bool = True,
        denoise: bool = True,
        adaptive_threshold: bool = True,
        crop_border: bool = True,
        min_dpi_upscale: int = 150,   # upscale if image seems low-res
    ):
        self.auto_detect_type = auto_detect_type
        self.deskew = deskew
        self.remove_shadow = remove_shadow
        self.denoise = denoise
        self.adaptive_threshold = adaptive_threshold
        self.crop_border = crop_border
        self.min_dpi_upscale = min_dpi_upscale

    # ─────────────────────────── main entry point ────────────────────────────

    def preprocess(self, pil_image: Image.Image) -> Image.Image:
        """
        Apply full preprocessing pipeline to a PIL image.

        Args:
            pil_image: Input PIL image (any mode).

        Returns:
            Preprocessed PIL image ready for Tesseract.
        """
        if not CV2_AVAILABLE:
            return self._pillow_fallback(pil_image)

        # Convert PIL → OpenCV (BGR numpy array)
        img = self._pil_to_cv2(pil_image)

        # Detect image type and choose pipeline intensity
        img_type = self._detect_image_type(img) if self.auto_detect_type else "scan"
        logger.debug(f"[OpenCV] Detected image type: {img_type}")

        # Upscale if too small
        img = self._upscale_if_needed(img)

        # Step 1: Shadow / uneven illumination removal
        # Critical for phone-photographed reports (shadow cast by hand or lighting)
        if self.remove_shadow and img_type in ("photo", "scan"):
            img = self._remove_shadow(img)

        # Step 2: Grayscale
        gray = self._to_grayscale(img)

        # Step 3: Deskew — straighten tilted pages
        # Essential for photos; optional for digital PDFs
        if self.deskew and img_type in ("photo", "scan"):
            gray = self._deskew(gray)

        # Step 4: Denoise
        # NLMeans denoising is slow but high quality; fast mode uses gaussian blur
        if self.denoise:
            gray = self._denoise(gray, fast=(img_type == "digital"))

        # Step 5: Adaptive threshold → binary image
        # Better than Otsu for documents with uneven background (shadow, yellowed paper)
        if self.adaptive_threshold:
            gray = self._adaptive_threshold(gray, img_type)

        # Step 6: Remove black border artifacts from scanners
        if self.crop_border:
            gray = self._crop_border(gray)

        return self._cv2_to_pil(gray)

    # ─────────────────────────── type detection ──────────────────────────────

    def _detect_image_type(self, img: np.ndarray) -> str:
        """
        Classify the image as 'digital', 'scan', or 'photo'.

        Heuristics:
          - 'digital': very uniform background, crisp edges, high contrast
          - 'scan':    mostly uniform but with scanner noise/artifacts
          - 'photo':   uneven background brightness, possible shadow gradient
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # Check background variance in corners (photos have high variance due to shadows)
        h, w = gray.shape
        corner_size = min(h, w) // 8
        corners = [
            gray[:corner_size, :corner_size],
            gray[:corner_size, w - corner_size:],
            gray[h - corner_size:, :corner_size],
            gray[h - corner_size:, w - corner_size:],
        ]
        corner_stds = [np.std(c) for c in corners]
        max_corner_std = max(corner_stds)

        # High variance in corners → uneven lighting → photo
        if max_corner_std > 30:
            return "photo"

        # Check overall brightness std — scans are mostly white with dark text
        overall_std = np.std(gray)
        if overall_std < 60:
            return "digital"

        return "scan"

    # ─────────────────────────── pipeline steps ──────────────────────────────

    def _remove_shadow(self, img: np.ndarray) -> np.ndarray:
        """
        Remove shadow / uneven illumination using morphological background estimation.

        How it works:
          1. Dilate image with a large kernel to smear out text → gives the background
          2. Divide original by background → normalises brightness across the page
          3. Scale to full 0-255 range

        This is the single most impactful step for phone-photographed lab reports.
        Example: Praveen's Aadhaar card photo (page 1) has heavy shadow in the corners.
        """
        # Work in grayscale for background estimation
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Large structuring element to cover entire text regions
        kernel_size = max(gray.shape[0] // 20, 50)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        # Dilate = morphological background
        background = cv2.dilate(gray, kernel)

        # Divide original by background (with float32 for precision)
        normalized = cv2.divide(
            gray.astype(np.float32),
            background.astype(np.float32),
            scale=255.0
        ).astype(np.uint8)

        # If input was BGR, convert back
        if len(img.shape) == 3:
            return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

        return normalized

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        """Convert to single-channel grayscale."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _deskew(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect and correct page rotation using Hough line transform.

        How it works:
          1. Detect edges (Canny)
          2. Find dominant lines (Hough)
          3. Calculate angle of dominant lines vs horizontal
          4. Rotate image to correct

        Typical skew in phone-photographed documents: 2-15 degrees.
        The correction only fires if angle > 0.5° to avoid unnecessary warping.
        """
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            logger.debug("[OpenCV] Deskew: no lines detected, skipping.")
            return gray

        # Collect angles of detected lines
        angles = []
        for line in lines[:50]:  # Use top 50 strongest lines
            rho, theta = line[0]
            # Convert to degrees from horizontal
            angle = np.degrees(theta) - 90
            # Only include near-horizontal lines (within ±45°)
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return gray

        # Use median angle (robust to outliers)
        median_angle = np.median(angles)

        # Only correct if skew is meaningful (>0.5°)
        if abs(median_angle) < 0.5:
            logger.debug(f"[OpenCV] Deskew: angle {median_angle:.2f}° — skipping (< 0.5°)")
            return gray

        logger.debug(f"[OpenCV] Deskewing by {median_angle:.2f}°")

        # Rotate image around centre
        h, w = gray.shape
        centre = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    def _denoise(self, gray: np.ndarray, fast: bool = False) -> np.ndarray:
        """
        Remove noise from scanned or photographed images.

        Two modes:
          fast=True  — Gaussian blur (fast, good for mostly-clean digital images)
          fast=False — Non-Local Means denoising (slower, better for grainy scans/photos)

        NLMeans: looks at patches across the whole image to decide what is noise.
        Much better than simple blur for text documents — preserves edge sharpness.
        """
        if fast:
            return cv2.GaussianBlur(gray, (3, 3), 0)

        # NLMeans parameters tuned for document text:
        #   h=10: filter strength (higher = more noise removed but text may blur)
        #   templateWindowSize=7: size of patch compared per pixel
        #   searchWindowSize=21: area searched for matching patches
        return cv2.fastNlMeansDenoising(
            gray,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _adaptive_threshold(self, gray: np.ndarray, img_type: str) -> np.ndarray:
        """
        Binarise image using adaptive thresholding.

        Why adaptive (not global Otsu):
          Otsu picks one threshold for the whole image.
          Adaptive picks a local threshold per region.
          This handles: yellowed paper, shadow gradients, mixed dark/light areas.

        Parameters differ by image type:
          - digital: small block size (fine local decisions, clean input)
          - scan/photo: larger block size (smoother decisions for noisy input)
        """
        if img_type == "digital":
            block_size = 15
            C = 8
        else:
            block_size = 31
            C = 12

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            C,
        )
        return binary

    def _crop_border(self, gray: np.ndarray) -> np.ndarray:
        """
        Remove black scanner borders.

        Scanners often produce thick black borders around the scanned page.
        These confuse Tesseract (it tries to OCR them as text / table borders).

        Strategy:
          1. Find the largest white contour (the actual page)
          2. Crop to that bounding box + small padding
        """
        # Invert binary image so the white page becomes black for contour detection
        inverted = cv2.bitwise_not(gray)

        # Find contours
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return gray

        # Find largest contour = the page itself
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Only crop if there's a meaningful border (>2% of image dimension)
        img_h, img_w = gray.shape
        if x < img_w * 0.02 and y < img_h * 0.02:
            return gray  # No meaningful border detected

        # Add small padding back
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(img_w - x, w + 2 * pad)
        h = min(img_h - y, h + 2 * pad)

        logger.debug(f"[OpenCV] Cropping border: {x},{y} {w}x{h}")
        return gray[y:y + h, x:x + w]

    def _upscale_if_needed(self, img: np.ndarray, target_width: int = 2000) -> np.ndarray:
        """
        Upscale small images to improve OCR accuracy.

        Tesseract accuracy degrades significantly below ~150 DPI.
        A standard A4 page at 150 DPI is ~1240px wide.
        We target 2000px width as a safe minimum for lab reports.
        """
        h, w = img.shape[:2]
        if w >= target_width:
            return img

        scale = target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.debug(f"[OpenCV] Upscaling {w}x{h} → {new_w}x{new_h}")
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # ─────────────────────────── table detection ─────────────────────────────

    def detect_table_regions(self, pil_image: Image.Image) -> list:
        """
        Detect rectangular table regions in a lab report image.

        Returns a list of (x, y, w, h) bounding boxes for each detected table.
        These can be passed to the parser to focus OCR on tabular content.

        How it works:
          1. Binarise image
          2. Detect horizontal lines using morphological erosion
          3. Detect vertical lines using morphological erosion
          4. Combine → table grid mask
          5. Find contours of grid regions

        Use case: Praveen's report has multiple distinct tables per page
        (Lipid Profile, CBC, RFT etc). Detecting them separately improves
        parser accuracy because each table has a consistent column structure.
        """
        if not CV2_AVAILABLE:
            logger.warning("[OpenCV] Table detection unavailable — OpenCV not installed.")
            return []

        img = self._pil_to_cv2(pil_image)
        gray = self._to_grayscale(img)

        # Binarise
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines
        h, w = binary.shape
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        # Combine lines → table grid mask
        table_mask = cv2.add(horizontal_lines, vertical_lines)

        # Dilate to connect nearby lines into solid regions
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        table_mask = cv2.dilate(table_mask, dilate_kernel, iterations=3)

        # Find contours of table regions
        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter: only keep regions large enough to be a real table
        min_area = (w * h) * 0.01  # at least 1% of image area
        table_regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, tw, th = cv2.boundingRect(cnt)
                table_regions.append((x, y, tw, th))

        table_regions.sort(key=lambda r: r[1])  # sort top-to-bottom
        logger.info(f"[OpenCV] Detected {len(table_regions)} table region(s)")
        return table_regions

    def crop_to_region(self, pil_image: Image.Image, region: tuple) -> Image.Image:
        """Crop PIL image to a detected region (x, y, w, h)."""
        x, y, w, h = region
        return pil_image.crop((x, y, x + w, y + h))

    # ─────────────────────────── utilities ───────────────────────────────────

    @staticmethod
    def _pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image → OpenCV BGR numpy array."""
        rgb = pil_image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV grayscale/BGR array → PIL Image."""
        if len(cv2_image.shape) == 2:
            return Image.fromarray(cv2_image)
        return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _pillow_fallback(pil_image: Image.Image) -> Image.Image:
        """Pillow-only preprocessing when OpenCV is not installed."""
        from PIL import ImageFilter, ImageEnhance

        img = pil_image.convert("RGB")
        img = img.convert("L")

        # Upscale if small
        w, h = img.size
        if w < 1000:
            scale = 1000 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        img = img.filter(ImageFilter.SHARPEN)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        return img
