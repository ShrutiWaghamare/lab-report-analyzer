"""
Hybrid Data Extraction — Rule-Based + LLM Fallback
===================================================

Strategy:
  1. Try RULE-BASED parser (fast, free) — works for ~70% of reports
  2. If low confidence or failure → Use AZURE OPENAI (accurate)
  3. Return structured JSON {patient_info, test_results}

Confidence Scoring:
  - High confidence (>=85%): Rule-based results are reliable
  - Low confidence (<85%): Use Azure LLM for accuracy

Cost Optimization: Only use Azure when needed
  - Typical: 70% rule-based, 30% LLM fallback
  - Cost: ~$0.01 per report (vs $0.02 if all LLM)
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

from ocr.parser import ReportParser
from core.table_aware_extraction import TableAwareExtractor, FormatAwareParser, PDFTypeDetector


# ─────────────────────────────────────────────────────────────────────────────
#  Rule-Based Extractor (Wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedExtractor:
    """Wrapper around ReportParser with confidence scoring."""

    def __init__(self):
        self.parser = ReportParser()
        self.table_extractor = TableAwareExtractor()
        self.format_parser = FormatAwareParser()

    def extract(self, raw_text: str, pdf_path: Optional[str] = None) -> Tuple[Dict, float]:
        """
        Extract structured data using rule-based parser.

        Args:
            raw_text: Raw OCR/extracted text
            pdf_path: Original PDF path (optional, for enhanced table extraction)

        Returns:
            (parsed_data, confidence_score)
            - parsed_data: {patient_info, test_results}
            - confidence: 0.5 to 1.0
        """
        try:
            # Step 1: If we have a file path and it's a PDF, try table-aware extraction
            if pdf_path and pdf_path.lower().endswith('.pdf'):
                logger.info("[RuleBasedExtractor] PDF detected, attempting table-aware extraction")
                text_with_tables, metadata = self.table_extractor.extract_from_pdf(pdf_path)
                
                # Use text with tables if extraction was successful
                if text_with_tables and metadata.get('extraction_method'):
                    raw_text = text_with_tables
                    logger.info(
                        f"[RuleBasedExtractor] Table extraction success: "
                        f"method={metadata['extraction_method']}, "
                        f"tables={metadata.get('table_count', 0)}"
                    )
            
            # Step 2: Parse the text
            parsed = self.parser.parse(raw_text)
            confidence = self._calculate_confidence(parsed, raw_text)
            
            logger.info(
                f"[RuleBasedExtractor] Extracted {len(parsed.get('test_results', []))} tests, "
                f"confidence={confidence:.2f}"
            )
            return parsed, confidence
        except Exception as e:
            logger.warning(f"[RuleBasedExtractor] Failed: {e}")
            return {"patient_info": {}, "test_results": []}, 0.0

    def _calculate_confidence(self, parsed: Dict, raw_text: str) -> float:
        """
        Score confidence 0.0-1.0 based on extraction quality.

        Low confidence signals: use LLM instead.
        """
        score = 0.8  # Start with baseline confidence

        patient_info = parsed.get("patient_info", {})
        test_results = parsed.get("test_results", [])

        # Penalty: Missing patient info
        if patient_info.get("name") == "Unknown":
            score -= 0.15
        if not patient_info.get("age"):
            score -= 0.10
        if not patient_info.get("gender"):
            score -= 0.05

        # Penalty: No tests extracted
        if len(test_results) == 0:
            score -= 0.50  # Critical failure
        elif len(test_results) < 3:
            score -= 0.25  # Very few tests

        # Penalty: Many missing units
        missing_units = sum(1 for t in test_results if not t.get("unit"))
        if missing_units > len(test_results) * 0.3:
            score -= 0.15

        # Penalty: Many blank reference ranges
        missing_refs = sum(1 for t in test_results if not t.get("reference_range"))
        if missing_refs > len(test_results) * 0.5:
            score -= 0.10

        # Bonus: Large raw text (more data to extract from)
        if len(raw_text) > 5000:
            score += 0.05

        return max(0.0, min(1.0, score))  # Clamp to [0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
#  Azure LLM Extractor
# ─────────────────────────────────────────────────────────────────────────────

class AzureLLMExtractor:
    """
    Calls Azure OpenAI gpt-5.2-chat to extract lab data from raw text.
    Returns structured {patient_info, test_results}.
    """

    EXTRACTION_PROMPT = """You are an expert at extracting structured data from laboratory reports.

Extract the following information and return ONLY valid JSON (no markdown, no explanation):

{
  "patient_info": {
    "name": "patient's full name or 'Unknown'",
    "age": "age in years as a number, or empty string if not found",
    "gender": "male, female, other, or empty string",
    "date_of_test": "YYYY-MM-DD format or empty string if not found"
  },
  "test_results": [
    {
      "test_name": "canonical test name (e.g. 'Hemoglobin')",
      "value": 15.5 (numeric value or null if not numeric),
      "unit": "unit of measurement (e.g. 'g/dL')",
      "reference_range": "reference/normal range as shown (e.g. '13-17' or '9.0-11.0')",
      "status": "Normal, High, Low, or Not Specified (based on report if indicated)"
    }
  ]
}

RULES:
1. Extract ALL test results visible in the report
2. For test_name, use the canonical/standard medical name (not abbreviations)
3. For value, use ONLY the actual test result value, not reference ranges
4. For unit, extract the unit of measurement exactly as shown
5. For reference_range, show it exactly as the lab reports it
6. For status, only include if the report explicitly marks it as Normal/High/Low
7. If a field is missing, use null or empty string, never omit the field
8. Return ONLY the JSON object, starting with { and ending with }
9. Do not include markdown code fences or explanations

Lab Report Text:
"""

    def __init__(self):
        self._client_config = self._load_azure_config()

    def _load_azure_config(self) -> Optional[Dict]:
        """Load Azure OpenAI config from environment."""
        key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not key or not endpoint:
            logger.warning(
                "[AzureLLMExtractor] Azure OpenAI not configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to enable LLM extraction."
            )
            return None

        logger.info(f"[AzureLLMExtractor] Azure configured: deployment={deployment}")
        return {
            "key": key,
            "endpoint": endpoint.rstrip("/"),
            "deployment": deployment,
            "api_version": api_version,
        }

    def extract(self, raw_text: str, retry_on_invalid: bool = True) -> Tuple[Dict, float]:
        """
        Extract structured data from raw text using Azure OpenAI.

        Args:
            raw_text: Raw OCR text from lab report
            retry_on_invalid: If JSON parsing fails, retry once with clearer prompt

        Returns:
            (parsed_data, confidence_score)
            - parsed_data: {patient_info, test_results}
            - confidence: 1.0 (trust LLM output when successful)
        """
        if not self._client_config:
            logger.error("[AzureLLMExtractor] Azure not configured, cannot extract.")
            return {"patient_info": {}, "test_results": []}, 0.0

        try:
            # Truncate text to avoid token limits
            text_truncated = raw_text[:10000]

            user_prompt = self.EXTRACTION_PROMPT + "\n" + text_truncated

            response_text = self._call_azure(user_prompt)
            parsed = self._parse_response(response_text)

            if not parsed.get("test_results"):
                if retry_on_invalid:
                    logger.info("[AzureLLMExtractor] Empty response, retrying with clearer prompt...")
                    return self._retry_extraction(raw_text)
                else:
                    logger.warning("[AzureLLMExtractor] No test results in response.")
                    return {"patient_info": {}, "test_results": []}, 0.5

            logger.info(
                f"[AzureLLMExtractor] Extracted {len(parsed.get('test_results', []))} tests from Azure"
            )
            return parsed, 1.0  # Full confidence in LLM output

        except Exception as e:
            logger.error(f"[AzureLLMExtractor] Extraction failed: {e}")
            return {"patient_info": {}, "test_results": []}, 0.0

    def _retry_extraction(self, raw_text: str) -> Tuple[Dict, float]:
        """Retry with a simpler, more explicit prompt."""
        retry_prompt = f"""Extract lab test data. Return ONLY valid JSON with this structure:
{{
  "patient_info": {{"name": "...", "age": 0, "gender": "...", "date_of_test": "..."}},
  "test_results": [
    {{"test_name": "...", "value": 0.0, "unit": "...", "reference_range": "...", "status": "..."}}
  ]
}}

Report:
{raw_text[:8000]}
"""
        try:
            response_text = self._call_azure(retry_prompt)
            parsed = self._parse_response(response_text)
            return parsed, 0.95  # Slightly lower confidence for retry
        except Exception as e:
            logger.error(f"[AzureLLMExtractor] Retry failed: {e}")
            return {"patient_info": {}, "test_results": []}, 0.0

    def _call_azure(self, user_prompt: str) -> str:
        """
        Call Azure OpenAI Chat Completions API.

        Returns:
            JSON string response
        """
        cfg = self._client_config
        url = (
            f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}"
            f"/chat/completions?api-version={cfg['api_version']}"
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at extracting structured data from lab reports. "
                    "Return ONLY valid JSON, no markdown, no explanation.",
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,  # Lower temp for consistency
            "max_completion_tokens": 2000,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "api-key": cfg["key"],
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"[AzureLLMExtractor] HTTP {e.code}: {error_body}")
            raise RuntimeError(f"Azure API error: {error_body}") from e

    def _parse_response(self, text: str) -> Dict:
        """
        Parse Azure response, handling markdown code fences.

        Returns valid JSON dict or empty dict on failure.
        """
        # Remove markdown code fences if present
        clean = re.sub(r"^```(?:json)?\s*", "", text.strip())
        clean = re.sub(r"\s*```$", "", clean)

        try:
            parsed = json.loads(clean)
            # Ensure structure
            if "patient_info" not in parsed:
                parsed["patient_info"] = {}
            if "test_results" not in parsed:
                parsed["test_results"] = []
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"[AzureLLMExtractor] Invalid JSON response: {e}")
            return {"patient_info": {}, "test_results": []}


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid Extractor (Orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

class HybridExtractor:
    """
    Two-stage extraction: try rule-based first, fallback to LLM if needed.

    Configuration:
      CONFIDENCE_THRESHOLD = 0.85 (use LLM if rule-based < this)
    """

    CONFIDENCE_THRESHOLD = 0.85

    def __init__(self):
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = AzureLLMExtractor()

    def extract(self, raw_text: str, pdf_path: Optional[str] = None) -> Tuple[Dict, str]:
        """
        Extract lab data using hybrid approach with table awareness.

        Args:
            raw_text: Raw OCR text from lab report
            pdf_path: Original PDF file path (optional, for enhanced extraction)

        Returns:
            (parsed_data, extraction_method)
            - extraction_method: "RULE-BASED" or "AZURE_LLM"
        """
        # Step 1: Try rule-based extraction (with table awareness if PDF)
        logger.info("[HybridExtractor] Attempting rule-based extraction...")
        parsed, confidence = self.rule_extractor.extract(raw_text, pdf_path=pdf_path)

        if confidence >= self.CONFIDENCE_THRESHOLD:
            logger.info(
                f"[HybridExtractor] ✓ Rule-based successful (confidence={confidence:.2f})"
            )
            return parsed, "RULE-BASED"

        # Step 2: Fallback to LLM
        logger.info(
            f"[HybridExtractor] Rule-based confidence low ({confidence:.2f}), "
            f"falling back to Azure LLM..."
        )
        llm_parsed, llm_confidence = self.llm_extractor.extract(raw_text)

        if len(llm_parsed.get("test_results", [])) > 0:
            logger.info("[HybridExtractor] ✓ Azure LLM extraction successful")
            return llm_parsed, "AZURE_LLM"

        # Last resort: return rule-based result even if low confidence
        logger.warning(
            "[HybridExtractor] Both methods produced poor results, "
            "returning rule-based result"
        )
        return parsed, "RULE-BASED (fallback)"
