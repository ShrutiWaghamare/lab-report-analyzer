"""
Test Evaluator Module
Evaluates test values vs reference ranges and assigns severity levels.

FIXES APPLIED:
  1. _is_special_reference_format — now also catches complex/garbage strings
     that pdfplumber extracts from multi-column lipid tables like
     '200 200-239 ≥ 240 ≥ 240'. These are no longer sent to _handle_special_reference.
  2. _handle_special_reference — wraps float() in try/except so any
     unparseable reference range returns "unknown" instead of crashing.
  3. _clean_reference_range — new helper that extracts just the first
     numeric threshold from any messy reference range string.
  4. _parse_reference_range — new helper that tries to extract (min, max)
     from any format: '13-17', '< 200', '> 40', '8.0 - 23.0', etc.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Severity(Enum):
    NORMAL = "Normal"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


@dataclass
class TestResult:
    test_name: str
    value: float
    unit: str
    reference_range: str
    status: str
    severity: Severity
    interpretation: str


class TestEvaluator:
    """Evaluates individual test results against reference ranges."""

    def __init__(self):
        self.reference_ranges = self._load_reference_ranges()

    def _load_reference_ranges(self) -> Dict:
        """Load comprehensive reference ranges for lab tests."""
        return {
            # Complete Blood Count
            "hemoglobin":       {"male": (13.0, 17.0), "female": (12.0, 16.0), "unit": "g/dL"},
            "rbc_count":        {"male": (4.7, 6.1),   "female": (4.2, 5.4),   "unit": "10^6/μl"},
            "pcv":              {"male": (41, 50),      "female": (36, 44),     "unit": "%"},
            "mcv":              {"normal": (80, 100),   "unit": "fl"},
            "mch":              {"normal": (27, 31),    "unit": "pg"},
            "mchc":             {"normal": (32, 36),    "unit": "g/dL"},
            "rdw":              {"normal": (11.5, 14.5),"unit": "%"},
            "rdw_cv":           {"normal": (11.6, 14.0),"unit": "%"},
            "rdw_sd":           {"normal": (35.1, 43.9),"unit": "fl"},
            "wbc_count":        {"normal": (4.0, 11.0), "unit": "10^3/μl"},
            "platelets":        {"normal": (150, 450),  "unit": "10^3/μl"},
            "neutrophils":      {"normal": (40, 70),    "unit": "%"},
            "lymphocytes":      {"normal": (20, 40),    "unit": "%"},

            # Diabetes markers
            "hba1c":            {
                "normal":      (0, 5.6),
                "prediabetes": (5.7, 6.4),
                "diabetes":    (6.5, 20),
                "unit": "%",
            },
            "glucose_fasting":  {
                "normal":      (70, 99),
                "prediabetes": (100, 125),
                "diabetes":    (126, 500),
                "unit": "mg/dL",
            },

            # Kidney function
            "creatinine":       {"male": (0.74, 1.35), "female": (0.59, 1.04), "unit": "mg/dL"},
            "egfr":             {"normal": (90, 200),  "unit": "ml/min/1.73 sq m"},
            "uric_acid":        {"male": (3.4, 7.0),   "female": (2.4, 6.0),   "unit": "mg/dL"},

            # Lipid profile
            "total_cholesterol": {
                "desirable":  (0, 200),
                "borderline": (200, 239),
                "high":       (240, 500),
                "unit": "mg/dL",
            },
            "hdl_cholesterol":   {
                "male_low":   (0, 40),
                "female_low": (0, 50),
                "good":       (40, 200),
                "unit": "mg/dL",
            },
            "ldl_cholesterol":   {
                "optimal":     (0, 100),
                "near_optimal":(100, 129),
                "borderline":  (130, 159),
                "high":        (160, 189),
                "very_high":   (190, 500),
                "unit": "mg/dL",
            },
            "triglycerides":     {
                "normal":    (0, 150),
                "borderline":(150, 199),
                "high":      (200, 499),
                "very_high": (500, 1000),
                "unit": "mg/dL",
            },

            # Thyroid
            "t3":  {"normal": (80, 200),  "unit": "ng/dL"},
            "t4":  {"normal": (5.0, 12.0),"unit": "μg/dL"},
            "tsh": {"normal": (0.4, 4.0), "unit": "μIU/mL"},

            # Electrolytes
            "sodium":   {"normal": (136, 145), "unit": "mmol/L"},
            "potassium":{"normal": (3.5, 5.1), "unit": "mmol/L"},
            "chloride": {"normal": (98, 107),  "unit": "mmol/L"},

            # Inflammation
            "esr": {
                "male_young":   (0, 15),
                "male_old":     (0, 20),
                "female_young": (0, 20),
                "female_old":   (0, 30),
                "unit": "mm/hr",
            },
        }

    # ------------------------------------------------------------------ public

    def evaluate_tests(
        self, test_results: List[Dict], age: int, gender: str
    ) -> List[TestResult]:
        evaluated = []
        for test in test_results:
            try:
                evaluated.append(self._evaluate_single_test(test, age, gender))
            except Exception as e:
                logger.warning(
                    f"[TestEvaluator] Skipping {test.get('test_name', '?')}: {e}"
                )
                evaluated.append(TestResult(
                    test_name=test.get("test_name", "Unknown"),
                    value=test.get("value", 0),
                    unit=test.get("unit", ""),
                    reference_range=test.get("reference_range", ""),
                    status="unknown",
                    severity=Severity.NORMAL,
                    interpretation="Could not evaluate — invalid reference range format",
                ))
        return evaluated

    def _evaluate_single_test(
        self, test: Dict, age: int, gender: str
    ) -> TestResult:
        test_name = test["test_name"].lower().replace(" ", "_")
        value = test["value"]
        unit = test.get("unit", "")
        reference_range = test.get("reference_range", "")

        test_name_mapping = {
            "rdw": "rdw_cv",
            "ldl_cholesterol": "ldl_cholesterol",
            "hdl_cholesterol": "hdl_cholesterol",
            "total_cholesterol": "total_cholesterol",
            "triglycerides": "triglycerides",
            "t3": "t3", "t4": "t4", "tsh": "tsh",
            "hemoglobin": "hemoglobin",
            "rbc_count": "rbc_count", "pcv": "pcv",
            "mcv": "mcv", "mch": "mch", "mchc": "mchc",
            "wbc_count": "wbc_count", "platelets": "platelets",
            "neutrophils": "neutrophils", "lymphocytes": "lymphocytes",
            "hba1c": "hba1c", "glucose_fasting": "glucose_fasting",
            "creatinine": "creatinine", "egfr": "egfr",
            "uric_acid": "uric_acid",
            "sodium": "sodium", "potassium": "potassium", "chloride": "chloride",
            "esr": "esr",
        }

        mapped_name = test_name_mapping.get(test_name, test_name)

        # FIX: check for special format ONLY on clean simple strings
        # Messy pdfplumber strings like '200 200-239 ≥ 240 ≥ 240' are NOT special
        if self._is_simple_threshold(reference_range):
            return self._handle_special_reference(test, value, unit, reference_range)

        # Known test with internal reference ranges
        if mapped_name in self.reference_ranges:
            ref_range = self.reference_ranges[mapped_name]
            status, severity = self._determine_status_severity(
                mapped_name, value, age, gender, ref_range
            )
            interpretation = self._get_interpretation(mapped_name, status, severity)
            return TestResult(
                test_name=test["test_name"],
                value=value, unit=unit,
                reference_range=reference_range,
                status=status, severity=severity,
                interpretation=interpretation,
            )

        # Unknown test — try to parse reference range from the string itself
        parsed = self._parse_reference_range(reference_range)
        if parsed:
            min_val, max_val = parsed
            status, severity = self._compare_to_range(value, min_val, max_val)
            interpretation = self._get_interpretation(mapped_name, status, severity)
            return TestResult(
                test_name=test["test_name"],
                value=value, unit=unit,
                reference_range=reference_range,
                status=status, severity=severity,
                interpretation=interpretation,
            )

        # Completely unknown — just store it without flagging
        return TestResult(
            test_name=test["test_name"],
            value=value, unit=unit,
            reference_range=reference_range,
            status="unknown",
            severity=Severity.NORMAL,
            interpretation="Test not in reference database",
        )

    # ------------------------------------------------------------------ FIX: reference range helpers

    def _is_simple_threshold(self, reference_range: str) -> bool:
        """
        FIX: Returns True ONLY for clean simple threshold strings like:
          '< 200'   '>40'   '<= 5.6'   '> 60'
        Returns False for messy pdfplumber strings like:
          '200 200-239 ≥ 240 ≥ 240'   '130 – 159'   '8.0 - 23.0'
        This prevents _handle_special_reference from receiving unparseable input.
        """
        if not reference_range:
            return False
        s = reference_range.strip()
        # Must start with < or > and be followed by optional space + a number only
        return bool(re.match(r'^[<>][=]?\s*[\d.]+\s*$', s))

    def _handle_special_reference(
        self, test: Dict, value: float, unit: str, reference_range: str
    ) -> TestResult:
        """
        Handle clean threshold strings: '< 200', '> 40', etc.
        FIX: wrapped in try/except — any parse failure returns 'unknown'
        instead of crashing the whole pipeline.
        """
        test_name = test["test_name"]
        try:
            s = reference_range.strip()
            # Remove <=  >= and just get the number
            num_str = re.sub(r'^[<>][=]?\s*', '', s)
            threshold = float(num_str)

            if '<' in s:
                if value > threshold:
                    deviation = (value - threshold) / threshold
                    severity = self._deviation_to_severity(deviation)
                    status = "high"
                else:
                    status = "normal"
                    severity = Severity.NORMAL
            else:  # >
                if value < threshold:
                    deviation = (threshold - value) / threshold
                    severity = self._deviation_to_severity(deviation)
                    status = "low"
                else:
                    status = "normal"
                    severity = Severity.NORMAL

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(
                f"[TestEvaluator] Could not parse reference range "
                f"'{reference_range}' for {test_name}: {e}"
            )
            status = "unknown"
            severity = Severity.NORMAL

        interpretation = self._get_interpretation(test_name.lower(), status, severity)
        return TestResult(
            test_name=test_name,
            value=value, unit=unit,
            reference_range=reference_range,
            status=status, severity=severity,
            interpretation=interpretation,
        )

    def _parse_reference_range(
        self, reference_range: str
    ) -> Optional[Tuple[float, float]]:
        """
        FIX: Try to extract (min, max) from any reference range string format.
        Handles:
          '13-17'          → (13, 17)
          '8.0 - 23.0'     → (8.0, 23.0)
          '3.5 - 5'        → (3.5, 5.0)
          '4000-10000'     → (4000, 10000)
          '130 – 159'      → (130, 159)  ← pdfplumber en-dash
          '0.270-4.20'     → (0.27, 4.20)
        Returns None if it cannot parse.
        """
        if not reference_range:
            return None

        # Normalise unicode dashes and spaces
        s = reference_range.strip()
        s = s.replace('–', '-').replace('—', '-').replace('−', '-')
        s = re.sub(r'\s*-\s*', '-', s)

        # Extract all numbers
        numbers = re.findall(r'[\d]+\.?[\d]*', s)
        if len(numbers) >= 2:
            try:
                lo = float(numbers[0])
                hi = float(numbers[1])
                if lo <= hi:
                    return lo, hi
            except ValueError:
                pass

        return None

    def _compare_to_range(
        self, value: float, min_val: float, max_val: float
    ) -> Tuple[str, Severity]:
        """Compare value to (min, max) and return status + severity."""
        if min_val <= value <= max_val:
            return "normal", Severity.NORMAL
        if value < min_val:
            deviation = (min_val - value) / min_val if min_val else 0
            return "low", self._deviation_to_severity(deviation)
        deviation = (value - max_val) / max_val if max_val else 0
        return "high", self._deviation_to_severity(deviation)

    def _deviation_to_severity(self, deviation: float) -> Severity:
        """Convert a fractional deviation to a Severity level."""
        if deviation <= 0.15:
            return Severity.MILD
        elif deviation <= 0.35:
            return Severity.MODERATE
        else:
            return Severity.SEVERE

    # ------------------------------------------------------------------ existing helpers (unchanged)

    def _determine_status_severity(
        self, test_name: str, value: float, age: int, gender: str, ref_range: Dict
    ) -> Tuple[str, Severity]:
        if test_name in ["total_cholesterol", "ldl_cholesterol", "triglycerides", "hba1c"]:
            return self._evaluate_multi_category_test(test_name, value, ref_range)
        if test_name == "hdl_cholesterol":
            return self._evaluate_hdl_cholesterol(value, gender, ref_range)
        if test_name == "esr":
            return self._evaluate_esr(value, age, gender, ref_range)

        if gender in ref_range:
            min_val, max_val = ref_range[gender]
        elif "normal" in ref_range:
            min_val, max_val = ref_range["normal"]
        else:
            return "unknown", Severity.NORMAL

        return self._compare_to_range(value, min_val, max_val)

    def _evaluate_multi_category_test(
        self, test_name: str, value: float, ref_range: Dict
    ) -> Tuple[str, Severity]:
        if test_name == "total_cholesterol":
            if value <= ref_range["desirable"][1]:
                return "normal", Severity.NORMAL
            elif value <= ref_range["borderline"][1]:
                return "high", Severity.MILD
            else:
                return "high", Severity.MODERATE

        elif test_name == "ldl_cholesterol":
            if value <= ref_range["optimal"][1]:
                return "normal", Severity.NORMAL
            elif value <= ref_range["borderline"][1]:
                return "high", Severity.MILD
            elif value <= ref_range["high"][1]:
                return "high", Severity.MODERATE
            else:
                return "high", Severity.SEVERE

        elif test_name == "triglycerides":
            if value <= ref_range["normal"][1]:
                return "normal", Severity.NORMAL
            elif value <= ref_range["borderline"][1]:
                return "high", Severity.MILD
            elif value <= ref_range["high"][1]:
                return "high", Severity.MODERATE
            else:
                return "high", Severity.SEVERE

        elif test_name == "hba1c":
            if value <= ref_range["normal"][1]:
                return "normal", Severity.NORMAL
            elif value <= ref_range["prediabetes"][1]:
                return "high", Severity.MILD
            else:
                return "high", Severity.SEVERE

        return "unknown", Severity.NORMAL

    def _evaluate_hdl_cholesterol(
        self, value: float, gender: str, ref_range: Dict
    ) -> Tuple[str, Severity]:
        threshold = 40 if gender == "male" else 50
        if value >= threshold:
            return "normal", Severity.NORMAL
        deviation = (threshold - value) / threshold
        return "low", self._deviation_to_severity(deviation)

    def _evaluate_esr(
        self, value: float, age: int, gender: str, ref_range: Dict
    ) -> Tuple[str, Severity]:
        if age < 50:
            max_val = ref_range["male_young"][1] if gender == "male" else ref_range["female_young"][1]
        else:
            max_val = ref_range["male_old"][1] if gender == "male" else ref_range["female_old"][1]

        if value <= max_val:
            return "normal", Severity.NORMAL
        deviation = (value - max_val) / max_val
        return "high", self._deviation_to_severity(deviation)

    def _get_interpretation(
        self, test_name: str, status: str, severity: Severity
    ) -> str:
        interpretations = {
            "hemoglobin":     {"low": "May indicate anemia, blood loss, or nutritional deficiency",
                               "high": "May indicate dehydration, lung disease, or blood disorders"},
            "rdw_cv":         {"high": "May indicate early iron deficiency, B12/folate deficiency, or blood disorders",
                               "low": "Generally not clinically significant"},
            "hdl_cholesterol":{"low": "Increased risk of cardiovascular disease"},
            "ldl_cholesterol":{"high": "Increased risk of cardiovascular disease and stroke"},
            "total_cholesterol":{"high": "Increased risk of cardiovascular disease"},
            "triglycerides":  {"high": "Increased risk of heart disease and pancreatitis"},
            "esr":            {"high": "May indicate inflammation, infection, or autoimmune condition"},
            "creatinine":     {"low": "May indicate muscle loss or malnutrition",
                               "high": "May indicate kidney dysfunction"},
            "uric_acid":      {"low": "May indicate malnutrition or liver disease",
                               "high": "May indicate gout risk or kidney problems"},
            "t3":             {"low": "May indicate hypothyroidism, illness, or malnutrition",
                               "high": "May indicate hyperthyroidism or thyroid hormone resistance"},
            "t4":             {"low": "May indicate hypothyroidism or thyroid hormone deficiency",
                               "high": "May indicate hyperthyroidism or excessive thyroid hormone"},
            "tsh":            {"low": "May indicate hyperthyroidism or pituitary dysfunction",
                               "high": "May indicate hypothyroidism or thyroid gland dysfunction"},
            "hba1c":          {"high": "Indicates poor blood sugar control — diabetes risk or poor diabetes management"},
            "glucose_fasting":{"low": "May indicate hypoglycemia",
                               "high": "May indicate diabetes or prediabetes"},
        }

        if test_name in interpretations and status in interpretations[test_name]:
            return interpretations[test_name][status]
        return f"Value is {status} — consult healthcare provider for interpretation"

    def get_abnormal_tests(self, evaluated_tests: List[TestResult]) -> List[Dict]:
        return [
            {
                "test_name":       t.test_name,
                "value":           t.value,
                "unit":            t.unit,
                "reference_range": t.reference_range,
                "status":          t.status,
                "severity":        t.severity.value,
                "interpretation":  t.interpretation,
            }
            for t in evaluated_tests
            if t.status != "normal"
        ]