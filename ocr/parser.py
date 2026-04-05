"""
Report Parser v3
================
Handles the vertical column format used by Apollo Diagnostics and similar labs
where PyMuPDF extracts each table cell as a separate line:

    HAEMOGLOBIN          ← test name line
    16.5                 ← value line
    g/dL                 ← unit line
    13-17                ← reference range line
    Spectrophotometer    ← method line (ignored)

This is completely different from the single-line format:
    HAEMOGLOBIN  16.5  g/dL  13-17

Both formats are detected and handled automatically.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── alias table ───────────────────────────────────────────────────────────────
TEST_ALIASES: Dict[str, str] = {
    "haemoglobin": "Hemoglobin", "hemoglobin": "Hemoglobin",
    "hb(haemoglobin)": "Hemoglobin", "hb": "Hemoglobin", "hgb": "Hemoglobin",
    "pcv": "PCV", "hct": "PCV", "packed cell volume": "PCV", "hematocrit": "PCV",
    "rbc count": "RBC Count", "rbc": "RBC Count",
    "mcv": "MCV", "mch": "MCH", "mchc": "MCHC",
    "rdw": "RDW", "r.d.w": "RDW", "rdw-cv": "RDW", "rdw cv": "RDW",
    "rdw-sd": "RDW SD", "rdw sd": "RDW SD",
    "wbc count": "WBC Count", "wbc": "WBC Count",
    "tlc": "WBC Count", "total leucocyte count": "WBC Count",
    "total leucocyte count (tlc)": "WBC Count",
    "platelet count": "Platelets", "platelets": "Platelets",
    "neutrophils": "Neutrophils", "lymphocytes": "Lymphocytes",
    "monocytes": "Monocytes", "eosinophils": "Eosinophils",
    "basophils": "Basophils", "mpv": "MPV",
    "hba1c": "HbA1c", "hba1c, glycated hemoglobin": "HbA1c",
    "glycated hemoglobin": "HbA1c", "hb a1c": "HbA1c",
    "glucose, fasting": "Glucose Fasting", "glucose, fasting , naf plasma": "Glucose Fasting",
    "fasting glucose": "Glucose Fasting", "fbs": "Glucose Fasting",
    "blood sugar fasting": "Glucose Fasting",
    "total cholesterol": "Total Cholesterol", "s. cholesterol": "Total Cholesterol",
    "triglycerides": "Triglycerides", "s. triglycerides": "Triglycerides",
    "hdl cholesterol": "HDL Cholesterol", "s. hdl cholesterol": "HDL Cholesterol",
    "ldl cholesterol": "LDL Cholesterol", "s. ldl cholesterol": "LDL Cholesterol",
    "vldl cholesterol": "VLDL", "vldl": "VLDL",
    "non-hdl cholesterol": "Non-HDL Cholesterol",
    "non hdl cholesterol": "Non-HDL Cholesterol",
    "tsh": "TSH", "tsh (ultrasensitive/4thgen)": "TSH",
    "t3": "T3", "tri-iodothyronine (t3, total)": "T3", "total t3": "T3",
    "t4": "T4", "thyroxine (t4, total)": "T4", "total t4": "T4",
    "sgpt": "SGPT (ALT)", "alt": "SGPT (ALT)", "sgpt(alt)": "SGPT (ALT)",
    "alanine aminotransferase (alt/sgpt)": "SGPT (ALT)",
    "alanine aminotransferase": "SGPT (ALT)",
    "sgot": "SGOT (AST)", "ast": "SGOT (AST)", "sgot(ast)": "SGOT (AST)",
    "aspartate aminotransferase (ast/sgot)": "SGOT (AST)",
    "aspartate aminotransferase": "SGOT (AST)",
    "alkaline phosphatase": "Alkaline Phosphatase", "alp": "Alkaline Phosphatase",
    "total bilirubin": "Total Bilirubin", "bilirubin, total": "Total Bilirubin",
    "direct bilirubin": "Direct Bilirubin",
    "bilirubin conjugated (direct)": "Direct Bilirubin",
    "indirect bilirubin": "Indirect Bilirubin", "bilirubin (indirect)": "Indirect Bilirubin",
    "gamma gt": "GAMMA GT", "ggt": "GAMMA GT",
    "albumin": "Albumin",
    "total protein": "Total Protein", "protein, total": "Total Protein",
    "globulin": "Globulin",
    "creatinine": "Creatinine", "serum creatinine": "Creatinine",
    "egfr": "eGFR", "estimated glomerular filtration rate": "eGFR",
    ".egfr - estimated glomerular filtration rate": "eGFR",
    "urea": "Blood Urea", "blood urea": "Blood Urea",
    "uric acid": "Uric Acid", "serum uric acid": "Uric Acid",
    "sodium": "Sodium", "sodium(na+)": "Sodium",
    "potassium": "Potassium", "potassium(k+)": "Potassium",
    "chloride": "Chloride", "chloride(cl-)": "Chloride",
    "calcium": "Calcium", "calcium(total)": "Calcium",
    "phosphorus": "Phosphorus", "phosphorus, inorganic": "Phosphorus",
    "esr": "ESR", "erythrocyte sedimentation rate": "ESR",
    "crp": "CRP", "c-reactive protein": "CRP",
    "hs-crp": "hs-CRP",
    "hs-crp (high sensitivity c-reactive protein)": "hs-CRP",
    "hs-crp (high sensitivity creactive protein)": "hs-CRP",
    "s. cholesterol": "Total Cholesterol",
    "s. triglycerides": "Triglycerides",
    "s. hdl cholesterol": "HDL Cholesterol",
    "s. ldl cholesterol": "LDL Cholesterol",
    "blood urea nitrogen": "Blood Urea Nitrogen",
}

CANONICAL_TO_KEY: Dict[str, str] = {
    "Hemoglobin": "hemoglobin", "RBC Count": "rbc_count", "PCV": "pcv",
    "MCV": "mcv", "MCH": "mch", "MCHC": "mchc", "RDW": "rdw", "RDW SD": "rdw_sd",
    "WBC Count": "wbc_count", "Platelets": "platelets",
    "Neutrophils": "neutrophils", "Lymphocytes": "lymphocytes",
    "Monocytes": "monocytes", "Eosinophils": "eosinophils", "Basophils": "basophils",
    "HbA1c": "hba1c", "Glucose Fasting": "glucose_fasting",
    "Total Cholesterol": "total_cholesterol", "HDL Cholesterol": "hdl_cholesterol",
    "LDL Cholesterol": "ldl_cholesterol", "Triglycerides": "triglycerides",
    "VLDL": "vldl", "Non-HDL Cholesterol": "non_hdl_cholesterol",
    "TSH": "tsh", "T3": "t3", "T4": "t4",
    "Creatinine": "creatinine", "eGFR": "egfr", "Uric Acid": "uric_acid",
    "Sodium": "sodium", "Potassium": "potassium", "Chloride": "chloride",
    "Calcium": "calcium", "Phosphorus": "phosphorus",
    "ESR": "esr", "CRP": "crp", "hs-CRP": "crp",
    "SGPT (ALT)": "sgpt", "SGOT (AST)": "sgot",
    "Alkaline Phosphatase": "alp",
    "Total Bilirubin": "total_bilirubin", "Direct Bilirubin": "direct_bilirubin",
    "Indirect Bilirubin": "indirect_bilirubin",
    "Albumin": "albumin", "Total Protein": "total_protein", "Globulin": "globulin",
    "Blood Urea": "blood_urea", "Blood Urea Nitrogen": "bun",
    "GAMMA GT": "gamma_gt", "MPV": "mpv",
}

PLAUSIBLE_BOUNDS = {
    "Hemoglobin": (3, 25), "PCV": (10, 70), "RBC Count": (1, 10),
    "MCV": (50, 130), "MCH": (15, 45), "MCHC": (25, 42), "RDW": (8, 30),
    "WBC Count": (0.5, 100000), "Platelets": (10, 1500000),
    "Neutrophils": (1, 100), "Lymphocytes": (1, 100),
    "HbA1c": (3, 20), "Glucose Fasting": (30, 700),
    "Total Cholesterol": (50, 600), "Triglycerides": (20, 2000),
    "HDL Cholesterol": (5, 200), "LDL Cholesterol": (10, 500),
    "TSH": (0.001, 100), "T3": (0.5, 500), "T4": (1, 30),
    "Creatinine": (0.3, 20), "eGFR": (5, 200), "Uric Acid": (1, 20),
    "Sodium": (110, 170), "Potassium": (2, 9), "Chloride": (70, 130),
    "CRP": (0.1, 500), "hs-CRP": (0.01, 100),
    "SGPT (ALT)": (1, 5000), "SGOT (AST)": (1, 5000),
    "Alkaline Phosphatase": (10, 2000),
    "Total Bilirubin": (0.1, 30), "Direct Bilirubin": (0.0, 20),
    "Albumin": (1, 7), "Total Protein": (3, 12), "Calcium": (5, 15),
}

# Lines to always skip
SKIP_PATTERNS = [
    r"^page\s+\d+", r"^test\s+name", r"^bio\.\s*ref", r"^method$",
    r"^comment", r"^note:", r"^interpretation", r"^reference\s+(range|group|interval)",
    r"^department\s+of", r"^collected\s*:", r"^received\s*:", r"^reported\s*:",
    r"^patient\s+name\s*:", r"^sponsor\s+name", r"^sin\s+no", r"^\d+\s*$",
    r"^in\s+vitro", r"^the\s+(test|assay|hba1c)", r"^for\s+pregnant",
    r"^non\s+diabetic", r"^prediabetes", r"^diabetes\s*$",
    r"^\*+\s*(end of report|terms|conditions)",
    r"^(spectrophotometer|calculated|electrical|flow cytometry|enzymatic|colorimetric|eclia|hplc|diazo|biuret|uricase|ise|god)",
    r"^(electronic pulse|immuno|turbid|god\s*-\s*pod|uv with|visible with|arsenazo|bromocresol)",
]


class ReportParser:
    """
    Parses lab report text into structured JSON.

    Automatically detects and handles two formats:
      Format A (vertical): each column on its own line  ← Apollo Diagnostics
      Format B (inline):   all columns on one line      ← Sarvodaya, generic
    """

    def parse(self, raw_text: str) -> Dict:
        lines = self._clean_lines(raw_text)
        patient_info = self._extract_patient_info(lines)

        # Detect format by checking if values appear on their own lines
        if self._is_vertical_format(lines):
            logger.info("[Parser v3] Detected vertical (multi-line) column format")
            test_results = self._parse_vertical(lines)
        else:
            logger.info("[Parser v3] Detected inline (single-line) column format")
            test_results = self._parse_inline(lines)

        logger.info(
            f"[Parser v3] name={patient_info['name']}  "
            f"age={patient_info['age']}  gender={patient_info['gender']}  "
            f"tests={len(test_results)}"
        )
        return {"patient_info": patient_info, "test_results": test_results}

    # ─────────────────── cleaning ────────────────────────────────────────────

    def _clean_lines(self, text: str) -> List[str]:
        text = text.replace("|", "  ")
        text = re.sub(r"[ \t]+", " ", text)
        lines = [l.strip() for l in text.splitlines()]
        return [l for l in lines if l]

    # ─────────────────── patient info ────────────────────────────────────────

    def _extract_patient_info(self, lines: List[str]) -> Dict:
        info = {"name": "Unknown", "age": "", "gender": ""}

        # Pre-join lines where the next line is a colon-continuation
        # e.g. "Patient Name" + ": Mr.ABHISHEK DATTA" → "Patient Name : Mr.ABHISHEK DATTA"
        joined = []
        i = 0
        while i < len(lines):
            if i + 1 < len(lines) and re.match(r"^:\s*\S", lines[i + 1]):
                joined.append(lines[i] + " " + lines[i + 1])
                i += 2
            else:
                joined.append(lines[i])
                i += 1

        for line in joined[:120]:
            lower = line.lower()

            # Name
            if info["name"] == "Unknown":
                m = re.search(
                    r"patient\s*(?:name)?\s*[:\-]\s*(?:mr\.?|mrs\.?|ms\.?|dr\.?)?\s*([A-Z][A-Za-z\s\.]+)",
                    line, re.IGNORECASE
                )
                if m:
                    name = m.group(1).strip().rstrip(".,")
                    if len(name) > 3 and "unknown" not in name.lower():
                        info["name"] = name

                # "Name : Mr. ABHISHEK DATTA"
                m2 = re.match(r"^name\s*[:\-]\s*(?:mr\.?|mrs\.?|ms\.?|dr\.?)?\s*([A-Z][A-Za-z\s\.]+)",
                               line, re.IGNORECASE)
                if m2 and info["name"] == "Unknown":
                    name = m2.group(1).strip().rstrip(".,")
                    if len(name) > 3:
                        info["name"] = name

            # Age
            if not info["age"]:
                # "Age/Gender : 28 Y 6 M" or "Age : 35 Yr" or inline "28 Y"
                m = re.search(r"age.*?[:\-]\s*(\d+)\s*[Yy]", line, re.IGNORECASE)
                if m:
                    info["age"] = f"{m.group(1)} years"
                else:
                    m2 = re.search(r"\b(\d{1,3})\s+(?:yr?s?|years?)\b", line, re.IGNORECASE)
                    if m2:
                        info["age"] = f"{m2.group(1)} years"

            # Gender
            if not info["gender"]:
                if re.search(r"\bfemale\b", lower):
                    info["gender"] = "female"
                elif re.search(r"\b(male|/m\b)", lower):
                    if not re.search(r"\b(method|min|mg|ml|mm)\b", lower):
                        info["gender"] = "male"

        return info

    # ─────────────────── format detection ────────────────────────────────────

    def _is_vertical_format(self, lines: List[str]) -> bool:
        """
        Detect if values are on their own lines (Apollo vertical format).
        Heuristic: look for a pattern of test-name line followed by a
        standalone number line.
        """
        matches = 0
        for i, line in enumerate(lines[:-1]):
            if self._is_test_name_line(line.lower()):
                next_line = lines[i + 1].strip()
                if re.match(r"^[\d,]+\.?\d*$", next_line.replace(",", "")):
                    matches += 1
        return matches >= 3

    def _is_test_name_line(self, lower: str) -> bool:
        for alias in TEST_ALIASES:
            if lower.strip() == alias or lower.strip().startswith(alias):
                return True
        return False

    # ─────────────────── vertical parser (Apollo format) ─────────────────────

    def _parse_vertical(self, lines: List[str]) -> List[Dict]:
        """
        State machine that walks line by line.
        State: WAITING → found test name → expecting value → got value → ...
        """
        results = []
        seen: set = set()
        i = 0

        while i < len(lines):
            line = lines[i]
            lower = line.lower().strip()

            # Find a test name on this line
            canonical = self._match_alias_exact(lower)
            if canonical and canonical not in seen:
                # Look ahead for value (skip blank/noise lines, max 3 ahead)
                value, unit, ref_range, consumed = self._lookahead_value(lines, i + 1)
                if value is not None and self._plausible(canonical, value):
                    key = CANONICAL_TO_KEY.get(canonical, canonical.lower().replace(" ", "_"))
                    results.append({
                        "test_name": key,
                        "display_name": canonical,
                        "value": value,
                        "unit": unit,
                        "reference_range": ref_range,
                    })
                    seen.add(canonical)
                    i += consumed + 1
                    continue
            i += 1

        return results

    def _match_alias_exact(self, lower: str) -> Optional[str]:
        """Match the line exactly or as a prefix against known aliases."""
        # Exact match first
        if lower in TEST_ALIASES:
            return TEST_ALIASES[lower]
        # Prefix match (handles cases like "HAEMOGLOBIN" matching "haemoglobin")
        for alias, canonical in sorted(TEST_ALIASES.items(), key=lambda x: -len(x[0])):
            if lower == alias:
                return canonical
        return None

    def _lookahead_value(
        self, lines: List[str], start: int
    ) -> Tuple[Optional[float], str, str, int]:
        """
        From position start, look for:
          line 0: numeric value (possibly with commas like 7,220)
          line 1: unit
          line 2: reference range

        Returns (value, unit, ref_range, lines_consumed)
        """
        if start >= len(lines):
            return None, "", "", 0

        # Skip noise/header lines
        offset = 0
        while start + offset < len(lines):
            candidate = lines[start + offset].strip()
            if self._should_skip(candidate):
                offset += 1
                if offset > 2:
                    return None, "", "", 0
                continue
            break

        pos = start + offset
        if pos >= len(lines):
            return None, "", "", 0

        # Parse value
        value = self._parse_number(lines[pos].strip())
        if value is None:
            return None, "", "", 0

        consumed = offset + 1  # consumed the value line

        # Next line = unit?
        unit = ""
        ref_range = ""
        if pos + 1 < len(lines):
            next_line = lines[pos + 1].strip()
            if self._looks_like_unit(next_line):
                unit = next_line
                consumed += 1
                # Next = ref range?
                if pos + 2 < len(lines):
                    rr = lines[pos + 2].strip()
                    if self._looks_like_ref_range(rr):
                        ref_range = rr
                        consumed += 1
            elif self._looks_like_ref_range(next_line):
                ref_range = next_line
                consumed += 1

        return value, unit, ref_range, consumed

    # ─────────────────── inline parser (Sarvodaya / single-line) ─────────────

    def _parse_inline(self, lines: List[str]) -> List[Dict]:
        results = []
        seen: set = set()

        for line in lines:
            if self._should_skip(line):
                continue
            parsed = self._parse_inline_row(line)
            if parsed and parsed["test_name"] not in seen:
                results.append(parsed)
                seen.add(parsed["test_name"])

        return results

    def _parse_inline_row(self, line: str) -> Optional[Dict]:
        lower = line.lower()
        best_canonical = None
        best_end = 0
        best_len = 0

        for alias, canonical in TEST_ALIASES.items():
            m = re.search(r'\b' + re.escape(alias), lower)
            if m and len(alias) > best_len:
                best_canonical = canonical
                best_end = m.end()
                best_len = len(alias)

        if not best_canonical:
            return None

        remainder = line[best_end:].strip(" :,\t")
        value, unit, ref_range = self._extract_inline_value(remainder, best_canonical)
        if value is None:
            return None
        if not self._plausible(best_canonical, value):
            return None

        key = CANONICAL_TO_KEY.get(best_canonical, best_canonical.lower().replace(" ", "_"))
        return {
            "test_name": key,
            "display_name": best_canonical,
            "value": value,
            "unit": unit,
            "reference_range": ref_range,
        }

    def _is_reference_range_token(self, tok: str) -> bool:
        """Detect if a token is part of a reference range (e.g., '200-499', '<200', '≥150')"""
        tok_clean = tok.strip(".,;:")
        # Pattern 1: contains dash like "200-499" or "200–499" (with unicode dash)
        if re.search(r"[\d][-–—][\d]", tok_clean):
            return True
        # Pattern 2: starts with comparison operator like "<200" or ">150"
        if re.match(r"^[<>≤≥]", tok_clean):
            return True
        return False
    
    def _is_range_prefix_pattern(self, tokens: list, idx: int) -> bool:
        """
        Check if tokens[idx] is a range threshold that should be skipped (not the actual value).
        E.g., in ['150', '200-499', '375'], tokens[0]='150' should be skipped.
        But in ['375', '≥240'], tokens[0]='375' should NOT be skipped (it's the value).
        
        Rule: Skip only if the current AND next tokens both look like range components.
        """
        if idx >= len(tokens) - 1:
            return False
        
        current_tok = tokens[idx].strip(".,;:")
        next_tok = tokens[idx + 1].strip(".,;:")
        
        # Only skip if current number is small and next is a clear range pattern
        # This handles cases like "150 200-499" where 150 is a threshold
        try:
            current_num = float(current_tok.replace(",", ""))
        except ValueError:
            return False
        
        # If current is small (< 250) and next is a clear range pattern, skip it
        # This helps with "150 200-499 375" pattern
        if current_num < 250 and self._is_reference_range_token(next_tok):
            # But don't skip if next is a single ≥/< value; that would be too aggressive
            if re.search(r"[\d][-–—][\d]", next_tok):
                # Next is a range like "200-499", so current is likely a threshold
                return True
        
        return False

    def _extract_inline_value(
        self, remainder: str, canonical: str
    ) -> Tuple[Optional[float], str, str]:
        cleaned = re.sub(r"\([^)]*\)", " ", remainder)
        tokens = cleaned.split()

        value = None
        unit = ""
        ref_start = 0
        all_candidates = []  # Track all numeric candidates for validation

        for i, token in enumerate(tokens):
            tok = token.strip(".,;:")
            if tok.upper() in ("H", "L", "HIGH", "LOW", "*", "A"):
                continue
            
            # FIX: Skip tokens that look like reference ranges (e.g., '200-499', '<200')
            if self._is_reference_range_token(tok):
                continue
            
            # FIX: Skip numbers that are range thresholds (e.g., '150' in '150 200-499 375')
            if self._is_range_prefix_pattern(tokens, i):
                continue
            
            num = self._parse_number(tok)
            if num is not None:
                # Skip small numbers that are likely footnotes
                if num <= 10 and canonical not in (
                    "TSH", "T4", "T3", "CRP", "hs-CRP", "Direct Bilirubin",
                    "Indirect Bilirubin", "Total Bilirubin", "Albumin",
                    "Total Protein", "Globulin", "Calcium", "Phosphorus",
                    "Potassium", "eGFR", "Blood Urea",
                ):
                    next_tok = tokens[i + 1].strip(".,;:").lower() if i + 1 < len(tokens) else ""
                    if not self._looks_like_unit(next_tok):
                        continue
                
                # Track all viable candidates
                all_candidates.append((num, i, tok))
                
                if value is None:  # Keep first valid value for now
                    value = num
                    ref_start = i + 1
                    if i + 1 < len(tokens) and self._looks_like_unit(tokens[i + 1].strip(".,;:")):
                        unit = tokens[i + 1].strip(".,;:")
                        ref_start = i + 2

        # FIX: If we have multiple value candidates, pick the one that's most plausible for this test
        if len(all_candidates) > 1 and canonical in PLAUSIBLE_BOUNDS:
            lo, hi = PLAUSIBLE_BOUNDS[canonical]
            # Filter candidates to those in plausible range
            plausible_candidates = [c for c in all_candidates if lo <= c[0] <= hi]
            if plausible_candidates:
                # Pick the last (rightmost) plausible value - usually the actual value after reference ranges
                best_candidate = plausible_candidates[-1]
                value = best_candidate[0]
                # Recalculate ref_start based on best candidate position
                ref_start = best_candidate[1] + 1
                # Check if next token is unit
                if ref_start - best_candidate[1] + best_candidate[1] < len(tokens):
                    next_idx = ref_start - 1
                    if next_idx + 1 < len(tokens):
                        next_tok_str = tokens[next_idx + 1].strip(".,;:")
                        if self._looks_like_unit(next_tok_str):
                            unit = next_tok_str
                            ref_start = next_idx + 2
                        elif ref_start < len(tokens):
                            unit = ""

        if value is None:
            return None, "", ""

        ref_range = " ".join(tokens[ref_start:]).strip()
        ref_range = re.sub(r"\s*[-–—]\s*", "-", ref_range)
        return value, unit, ref_range

    # ─────────────────── helpers ──────────────────────────────────────────────

    def _should_skip(self, line: str) -> bool:
        lower = line.lower().strip()
        if len(line) < 2:
            return True
        for pat in SKIP_PATTERNS:
            if re.search(pat, lower):
                return True
        return False

    def _parse_number(self, tok: str) -> Optional[float]:
        tok = tok.replace(",", "").strip()
        tok = re.sub(r"^[<>≤≥]", "", tok).strip()
        try:
            return float(tok)
        except ValueError:
            return None

    def _looks_like_unit(self, tok: str) -> bool:
        known = {
            "g/dl", "mg/dl", "mmol/l", "%", "fl", "pg", "10^6/ul",
            "million/cu.mm", "cells/cu.mm", "thou/mm3", "10^3/cmm",
            "mm/hr", "mg/l", "ug/dl", "ng/dl", "ng/ml", "uiu/ml",
            "ml/min/1.73m²", "ml/min", "lakh/cumm", "cells/cumm",
            "u/l", "iu/l", "meq/l", "g/dl", "mg%",
        }
        t = tok.lower().rstrip(".,;:")
        if t in known:
            return True
        if re.match(r"^[a-zμµ%\^²³/\.0-9]+$", t) and len(t) <= 20:
            if any(c in t for c in ["/", "%", "^"]):
                return True
        return False

    def _looks_like_ref_range(self, line: str) -> bool:
        """Does this line look like a reference range? e.g. 13-17 or 4000-10000 or <200"""
        stripped = line.strip()
        if re.match(r"^[<>≤≥]?\s*[\d,]+\.?\d*\s*[-–]\s*[\d,]+\.?\d*$", stripped):
            return True
        if re.match(r"^[<>≤≥]\s*[\d,]+\.?\d*$", stripped):
            return True
        return False

    def _plausible(self, canonical: str, value: float) -> bool:
        if canonical in PLAUSIBLE_BOUNDS:
            lo, hi = PLAUSIBLE_BOUNDS[canonical]
            return lo <= value <= hi
        return True