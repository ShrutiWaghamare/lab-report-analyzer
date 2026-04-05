"""
Enhanced Extraction with pdfplumber Table Detection
===================================================

Global Strategy for All PDF Types:
  1. Digital PDFs (Apollo, Thyrocare, etc.)
     → pdfplumber: Extract tables (structured)
     → pdfplumber: Extract text (unstructured)
     → Combine both for complete data

  2. Scanned PDFs (poor quality, image-based)
     → Check for text layer first (fast)
     → If no text layer → Skip OCR (user feedback: fast over complete)

  3. Mixed PDFs (tables + text sections)
     → Extract tables from table sections
     → Extract text from non-table sections
     → Parser handles both formats

Result: Works with ANY lab format globally
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class TableAwareExtractor:
    """
    Extract lab report data from PDFs using pdfplumber table detection.
    
    Handles all PDF types:
      - Digital PDFs with structured tables
      - Digital PDFs with mixed tables + text
      - Scanned PDFs (if text layer exists)
      - Completely text-based layouts
    """

    def __init__(self, dpi: int = 300):
        self.dpi = dpi

    def extract_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF using intelligent strategy.

        Returns:
            (raw_text, metadata)
            - raw_text: Full extracted text
            - metadata: {
                "extraction_method": "pdfplumber_tables" | "pdfplumber_text" | "pymupdf_text",
                "has_tables": bool,
                "table_count": int,
                "has_text_layer": bool,
                "page_count": int,
              }
        """
        metadata = {
            "extraction_method": None,
            "has_tables": False,
            "table_count": 0,
            "has_text_layer": False,
            "page_count": 0,
            "pdf_type": None,
        }

        if not PDFPLUMBER_AVAILABLE:
            logger.warning("[TableAwareExtractor] pdfplumber not available, falling back to PyMuPDF")
            return self._extract_pymupdf(pdf_path, metadata)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                metadata["page_count"] = len(pdf.pages)

                # Strategy 1: Try to extract tables
                raw_text, tables_found = self._extract_tables_and_text(pdf, metadata)

                if tables_found:
                    metadata["extraction_method"] = "pdfplumber_tables"
                    metadata["has_tables"] = True
                    logger.info(
                        f"[TableAwareExtractor] Extracted {metadata['table_count']} tables + text, "
                        f"method=pdfplumber_tables, pages={metadata['page_count']}"
                    )
                else:
                    # Strategy 2: No tables, use text extraction
                    raw_text = self._extract_text(pdf, metadata)
                    metadata["extraction_method"] = "pdfplumber_text"
                    logger.info(
                        f"[TableAwareExtractor] Extracted text only (no tables), "
                        f"method=pdfplumber_text, pages={metadata['page_count']}"
                    )

                return raw_text, metadata

        except Exception as e:
            logger.warning(f"[TableAwareExtractor] pdfplumber failed: {e}, falling back to PyMuPDF")
            return self._extract_pymupdf(pdf_path, metadata)

    def _extract_tables_and_text(self, pdf, metadata: Dict) -> Tuple[str, bool]:
        """
        Extract tables (if present) + remaining text.

        Returns:
            (combined_text, tables_found_bool)
        """
        all_text = []
        pages_with_tables = []

        for page_num, page in enumerate(pdf.pages, 1):
            # Find tables on this page
            tables = page.extract_tables()

            if tables:
                pages_with_tables.append(page_num)
                # Convert tables to text format
                for table in tables:
                    table_text = self._table_to_text(table)
                    all_text.append(table_text)

                # Also get remaining text (between/after tables)
                page_text = page.extract_text()
                if page_text:
                    # Remove text that's already in tables (approximate)
                    all_text.append(page_text)
            else:
                # No tables on this page, get all text
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
                    metadata["has_text_layer"] = True

        metadata["table_count"] = len([t for page in pdf.pages if page.extract_tables() for t in page.extract_tables()])
        tables_found = len(pages_with_tables) > 0

        return "\n".join(all_text), tables_found

    def _extract_text(self, pdf, metadata: Dict) -> str:
        """Extract pure text from all pages (no tables)."""
        all_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
                metadata["has_text_layer"] = True

        return "\n".join(all_text)

    def _table_to_text(self, table: List[List[str]]) -> str:
        """
        Convert pdfplumber table to readable text format.

        Format each row to preserve column alignment info.
        """
        if not table:
            return ""

        # Find max column width for alignment
        lines = []
        for row in table:
            # Clean None values and join with delimiters
            cells = [str(cell).strip() if cell else "" for cell in row]
            lines.append(" | ".join(cells))

        return "\n".join(lines)

    def _extract_pymupdf(self, pdf_path: str, metadata: Dict) -> Tuple[str, Dict]:
        """Fallback: Extract using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            logger.error("[TableAwareExtractor] No extractor available!")
            return "", metadata

        try:
            import fitz
            doc = fitz.open(pdf_path)
            metadata["page_count"] = len(doc)
            metadata["extraction_method"] = "pymupdf_text"

            all_text = []
            for page in doc:
                text = page.get_text()
                if text:
                    all_text.append(text)
                    metadata["has_text_layer"] = True

            doc.close()
            logger.info(
                f"[TableAwareExtractor] PyMuPDF extraction, "
                f"pages={metadata['page_count']}, has_text={metadata['has_text_layer']}"
            )
            return "\n".join(all_text), metadata

        except Exception as e:
            logger.error(f"[TableAwareExtractor] PyMuPDF also failed: {e}")
            return "", metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Format-Aware Parser (handles tables + text)
# ─────────────────────────────────────────────────────────────────────────────

class FormatAwareParser:
    """
    Parse extracted text that may contain:
      - Table-formatted data (columns separated by |)
      - Text-formatted data (vertical column format)
      - Mixed layouts
    """

    def detect_layout_type(self, raw_text: str) -> str:
        """
        Detect if text is table-based, vertical, or mixed format.

        Returns:
            "table" | "vertical" | "text" | "mixed"
        """
        # Count pipe characters (table indicator)
        pipe_count = raw_text.count("|")
        total_lines = len(raw_text.split("\n"))

        pipe_density = pipe_count / max(total_lines, 1)

        if pipe_density > 0.3:
            return "table"
        elif pipe_density > 0.1:
            return "mixed"
        elif self._has_vertical_layout(raw_text):
            return "vertical"
        else:
            return "text"

    def _has_vertical_layout(self, raw_text: str) -> bool:
        """Check if layout is vertical (test name on one line, value on next)."""
        lines = raw_text.split("\n")
        numeric_only_lines = sum(1 for line in lines if re.match(r"^\s*[\d,\.]+\s*$", line.strip()))
        return numeric_only_lines > len(lines) * 0.1

    def parse_table_row(self, row: str) -> Dict:
        """
        Parse a single table row.

        Handles formats like:
          "Test Name | 123.45 | mg/dL | 100-150"
          "Test Name|123.45|mg/dL|100-150"
          "Test Name    |    123.45|mg/dL    |100-150"
        """
        # Split by pipe, clean spaces
        parts = [p.strip() for p in row.split("|")]

        if len(parts) < 2:
            return None

        return {
            "test_name": parts[0] if parts[0] else None,
            "value": parts[1] if len(parts) > 1 else None,
            "unit": parts[2] if len(parts) > 2 else None,
            "reference_range": parts[3] if len(parts) > 3 else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  PDF Type Detector
# ─────────────────────────────────────────────────────────────────────────────

class PDFTypeDetector:
    """
    Detect PDF type to choose optimal extraction strategy.
    """

    @staticmethod
    def detect_pdf_type(pdf_path: str) -> str:
        """
        Detect if PDF is:
          - "digital_structured" (has tables, text layer) → Best for pdfplumber
          - "digital_text" (pure text, no tables) → Good for pdfplumber/PyMuPDF
          - "scanned_no_text" (image-based, no text layer) → Need OCR

        Returns: "digital_structured" | "digital_text" | "scanned_no_text"
        """
        if not PDFPLUMBER_AVAILABLE:
            return "unknown"

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check first page for characteristics
                if len(pdf.pages) == 0:
                    return "unknown"

                first_page = pdf.pages[0]

                # Check for text layer
                text = first_page.extract_text()
                has_text = bool(text and len(text.strip()) > 100)

                # Check for tables
                tables = first_page.extract_tables()
                has_tables = bool(tables)

                if has_tables and has_text:
                    return "digital_structured"
                elif has_text:
                    return "digital_text"
                else:
                    return "scanned_no_text"

        except Exception as e:
            logger.warning(f"[PDFTypeDetector] Could not detect: {e}")
            return "unknown"
