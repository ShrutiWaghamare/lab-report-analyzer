"""
Lab Report Crew — Multi-Agent Pipeline
=======================================

Pipeline:
  Agent 1  OCRAgent                   Extract text from PDF / image / file
  Agent 2  HybridExtractionAgent      Rule-based + LLM-fallback extraction → JSON
  Agent 3  RuleAnalysisAgent          Run rule-based MedicalAnalyzer (flags abnormals, conditions)
  Agent 4  AzureAnalysisAgent         Send raw text + rule results to Azure OpenAI for full analysis
  Agent 5  ValidationAgent            Sanity-check the final output

Hybrid Extraction Strategy:
  - Try rule-based parser first (fast, free)
  - If confidence < 85% → Fallback to Azure OpenAI (accurate, ~$0.02 per report)
  - Most reports (70%) use rule-based, only difficult ones use LLM

Environment variables (set in .env or PowerShell):
  AZURE_OPENAI_API_KEY      your Azure key
  AZURE_OPENAI_ENDPOINT     https://<resource>.openai.azure.com
  AZURE_OPENAI_DEPLOYMENT   deployment name  (e.g. gpt-5.2-chat)
  AZURE_OPENAI_API_VERSION  api version      (2024-02-15-preview)
"""

import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── local imports ─────────────────────────────────────────────────────────────
from ocr.extractor import OCRExtractor
from ocr.parser import ReportParser
from core.medical_analyzer import MedicalAnalyzer
from core.hybrid_extractor import HybridExtractor


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline state — passed between agents
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    file_path: Optional[str] = None
    raw_text: Optional[str] = None
    parsed_data: Optional[Dict] = None
    rule_analysis: Optional[Dict] = None   # rule-based MedicalAnalyzer output
    azure_analysis: Optional[Dict] = None  # Azure OpenAI full analysis
    patient_report: Optional[str] = None
    llm_explanation: Optional[str] = None
    validation_notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 1 — OCR
# ─────────────────────────────────────────────────────────────────────────────

class OCRAgent:
    """Extracts text from any file format."""

    def __init__(self):
        self.extractor = OCRExtractor(dpi=300)

    def run(self, state: AgentResult) -> AgentResult:
        logger.info("[OCRAgent] Extracting text.")
        try:
            if state.file_path:
                state.raw_text = self.extractor.extract(state.file_path)
                logger.info(f"[OCRAgent] {len(state.raw_text)} chars from {state.file_path}")
            elif state.raw_text:
                state.raw_text = self.extractor.extract_from_string(state.raw_text)
            else:
                raise ValueError("No file_path or raw_text provided.")
        except Exception as e:
            state.errors.append(f"OCRAgent: {e}")
            logger.error(f"[OCRAgent] {e}", exc_info=True)
        return state


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 2 — Hybrid Extractor (Rule-Based + LLM Fallback)
# ─────────────────────────────────────────────────────────────────────────────

class HybridExtractionAgent:
    """
    Two-stage data extraction with table awareness:
      Stage 1: Try rule-based parser + pdfplumber tables (fast, free)
      Stage 2: If low confidence → Use Azure OpenAI (accurate)
    """

    def __init__(self):
        self.extractor = HybridExtractor()

    def run(self, state: AgentResult) -> AgentResult:
        logger.info("[HybridExtractionAgent] Extracting structured data...")
        if not state.raw_text:
            state.errors.append("HybridExtractionAgent: no text to extract.")
            return state
        try:
            # Pass file_path for enhanced table-aware extraction
            parsed, extraction_method = self.extractor.extract(
                state.raw_text, 
                pdf_path=state.file_path
            )
            state.parsed_data = parsed
            n = len(parsed.get("test_results", []))
            logger.info(
                f"[HybridExtractionAgent] {extraction_method} — "
                f"{n} test results extracted"
            )
        except Exception as e:
            state.errors.append(f"HybridExtractionAgent: {e}")
            logger.error(f"[HybridExtractionAgent] {e}", exc_info=True)
        return state


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 3 — Rule-based analysis
# ─────────────────────────────────────────────────────────────────────────────

class RuleAnalysisAgent:
    """
    Runs the rule-based MedicalAnalyzer pipeline.
    Flags abnormal values, detects conditions, checks emergencies.
    Output is used as structured context for AzureAnalysisAgent.
    """

    def __init__(self):
        self.analyzer = MedicalAnalyzer()

    def run(self, state: AgentResult) -> AgentResult:
        logger.info("[RuleAnalysisAgent] Running rule-based analysis.")
        if not state.parsed_data:
            state.errors.append("RuleAnalysisAgent: no parsed data.")
            return state
        try:
            state.rule_analysis = self.analyzer.analyze(state.parsed_data)
            state.patient_report = self.analyzer.generate_patient_report(state.rule_analysis)
            logger.info("[RuleAnalysisAgent] Complete.")
        except Exception as e:
            state.errors.append(f"RuleAnalysisAgent: {e}")
            logger.error(f"[RuleAnalysisAgent] {e}", exc_info=True)
        return state


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 4 — Azure OpenAI Analysis
# ─────────────────────────────────────────────────────────────────────────────

class AzureAnalysisAgent:
    """
    Sends the raw extracted text AND rule-based findings to Azure OpenAI
    and gets back a comprehensive, structured clinical analysis.
    """

    # FIX: Added explicit "Return ONLY valid JSON" instruction at the top
    # so we don't need response_format=json_object which caused HTTP 400
    SYSTEM_PROMPT = """You are an expert clinical pathologist and medical analyst.
You will receive:
1. Raw extracted text from a lab report
2. Structured rule-based findings (abnormal tests, detected conditions)

Your task is to produce a comprehensive, accurate medical analysis.

IMPORTANT: Return ONLY valid JSON. No markdown, no explanation, no extra text.
Start your response with { and end with }

Use this exact structure:
{
  "patient_summary": "2-3 sentence overall health summary",
  "key_findings": [
    {
      "test": "test name",
      "value": "value with unit",
      "status": "High/Low/Normal/Borderline",
      "clinical_meaning": "plain English explanation of what this means",
      "urgency": "Routine/Monitor/Consult/Urgent"
    }
  ],
  "detected_conditions": [
    {
      "condition": "condition name",
      "confidence": "High/Medium/Low",
      "evidence": "which test values support this",
      "severity": "Mild/Moderate/Severe"
    }
  ],
  "risk_areas": [
    {
      "area": "e.g. Cardiovascular, Liver, Metabolic",
      "risk_level": "Low/Moderate/High",
      "reason": "brief explanation"
    }
  ],
  "recommendations": {
    "immediate": ["actions needed now"],
    "short_term": ["within 1-4 weeks"],
    "lifestyle": ["diet, exercise, habits"],
    "tests_to_repeat": ["which tests and when"]
  },
  "specialists_to_consult": ["list of relevant specialists if any"],
  "positive_findings": ["things that are normal or good"],
  "patient_friendly_explanation": "A warm, clear explanation for the patient in simple language. No jargon.",
  "disclaimer": "This analysis is AI-generated for educational purposes only. Always consult a qualified healthcare professional."
}

Be thorough, accurate, and clinically precise.
Do not fabricate findings not supported by the data.
"""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._client_config = self._load_azure_config() if use_llm else None

    def _load_azure_config(self) -> Optional[Dict]:
        """Load Azure OpenAI config from environment variables."""
        key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")  # FIX: updated default

        if not key or not endpoint:
            logger.warning(
                "[AzureAnalysisAgent] Azure not configured. "
                "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to enable LLM analysis."
            )
            return None

        logger.info(
            f"[AzureAnalysisAgent] Azure configured: "
            f"endpoint={endpoint}, deployment={deployment}, version={api_version}"
        )
        return {
            "key": key,
            "endpoint": endpoint.rstrip("/"),
            "deployment": deployment,
            "api_version": api_version,
        }

    def run(self, state: AgentResult) -> AgentResult:
        logger.info("[AzureAnalysisAgent] Starting analysis.")

        if not self._client_config:
            logger.info("[AzureAnalysisAgent] No Azure config — using rule-based fallback.")
            state.llm_explanation = self._fallback(state)
            state.azure_analysis = None
            return state

        if not state.raw_text:
            state.errors.append("AzureAnalysisAgent: no raw text.")
            return state

        try:
            user_prompt = self._build_prompt(state)
            response_text = self._call_azure(user_prompt)
            state.azure_analysis = self._parse_response(response_text)
            state.llm_explanation = state.azure_analysis.get(
                "patient_friendly_explanation",
                response_text
            )
            logger.info("[AzureAnalysisAgent] Analysis complete.")

        except Exception as e:
            state.errors.append(f"AzureAnalysisAgent: {e}")
            logger.error(f"[AzureAnalysisAgent] {e}", exc_info=True)
            state.llm_explanation = self._fallback(state)

        return state

    def _build_prompt(self, state: AgentResult) -> str:
        """Build the user message combining raw text + rule findings."""
        raw = (state.raw_text or "")[:8000]

        rule_context = {}
        if state.rule_analysis:
            rule_context = {
                "abnormal_tests": state.rule_analysis.get("abnormal_tests", []),
                "detected_conditions": [
                    {"name": c["name"], "severity": c["severity"]}
                    for c in state.rule_analysis.get("detected_conditions", [])
                ],
                "emergency_flags": state.rule_analysis.get("emergency_flags", []),
            }

        return f"""RAW LAB REPORT TEXT:
{raw}

---

RULE-BASED PRE-ANALYSIS (for reference):
{json.dumps(rule_context, indent=2)}

---

Please provide a comprehensive clinical analysis of this lab report in the JSON format specified.
Remember: return ONLY valid JSON, nothing else."""

    def _call_azure(self, user_prompt: str) -> str:
        """
        Call Azure OpenAI chat completions API directly using urllib.

        FIX: Removed "response_format": {"type": "json_object"} — this caused
        HTTP 400 on deployments that don't support JSON mode.
        The system prompt now explicitly instructs the model to return only JSON.

        FIX: Added detailed error logging so the actual Azure error message
        is visible in logs instead of just "HTTP Error 400: Bad Request".
        """
        cfg = self._client_config
        url = (
            f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}"
            f"/chat/completions?api-version={cfg['api_version']}"
        )

        # FIX: Log the URL so you can verify it looks correct
        logger.info(f"[AzureAnalysisAgent] Calling: {url}")

        payload = {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 1,
            "max_completion_tokens": 3000,  # FIX: max_tokens not supported by gpt-5.2-chat and newer models
            # FIX: REMOVED response_format — caused HTTP 400 on many deployments
            # "response_format": {"type": "json_object"},
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
        except urllib.error.HTTPError as e:
            # FIX: Read and log the actual Azure error body
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error(
                f"[AzureAnalysisAgent] HTTP {e.code} from Azure.\n"
                f"URL: {url}\n"
                f"Error body: {error_body}"
            )
            raise RuntimeError(
                f"Azure API returned HTTP {e.code}: {error_body}"
            ) from e

        return result["choices"][0]["message"]["content"]

    def _parse_response(self, text: str) -> Dict:
        """Parse Azure OpenAI response, stripping markdown fences if present."""
        clean = re.sub(r"^```(?:json)?\s*", "", text.strip())
        clean = re.sub(r"\s*```$", "", clean)
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            logger.warning("[AzureAnalysisAgent] Response was not valid JSON, returning raw.")
            return {"patient_friendly_explanation": text, "raw_response": text}

    def _fallback(self, state: AgentResult) -> str:
        """Rule-based summary when no LLM is available."""
        if state.rule_analysis:
            summary = state.rule_analysis.get("summary", "No summary available.")
            return f"{summary}\n\n*Azure OpenAI not configured — showing rule-based summary only.*"
        return "*No analysis available — configure AZURE_OPENAI_API_KEY to enable AI analysis.*"


# ─────────────────────────────────────────────────────────────────────────────
#  Agent 5 — Validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidationAgent:
    """Sanity-checks the pipeline output and surfaces QA notes."""

    def run(self, state: AgentResult) -> AgentResult:
        logger.info("[ValidationAgent] Validating output.")
        notes = []

        if not state.parsed_data:
            notes.append("⚠️ No structured data extracted.")
        else:
            n = len(state.parsed_data.get("test_results", []))
            if n == 0:
                notes.append("⚠️ Zero tests parsed — OCR or parser may have failed.")
            elif n < 3:
                notes.append(f"⚠️ Only {n} test(s) parsed — report may be incomplete.")
            else:
                notes.append(f"✓ {n} test results extracted successfully.")

            pi = state.parsed_data.get("patient_info", {})
            if pi.get("name") == "Unknown":
                notes.append("ℹ️ Patient name not detected.")
            if not pi.get("age"):
                notes.append("ℹ️ Age not detected — age-specific ranges may be less accurate.")
            if not pi.get("gender"):
                notes.append("ℹ️ Gender not detected — gender-specific ranges may be less accurate.")

        if state.azure_analysis:
            notes.append("✓ Azure OpenAI analysis completed.")
        elif state.rule_analysis:
            notes.append("ℹ️ Rule-based analysis only (Azure not configured).")

        for err in state.errors:
            notes.append(f"❌ {err}")

        state.validation_notes = notes
        return state


# ─────────────────────────────────────────────────────────────────────────────
#  Crew Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class LabReportCrew:
    """
    Orchestrates the full 5-agent pipeline.

    Usage:
        crew = LabReportCrew()
        result = crew.run(file_bytes=b"...", file_ext=".pdf")

        result["azure_analysis"]     # full structured clinical analysis from Azure
        result["llm_explanation"]    # patient-friendly plain language explanation
        result["patient_report"]     # markdown report from rule-based analysis
        result["test_results"]       # parsed test values
        result["validation_notes"]   # QA notes
    """

    def __init__(self, use_llm: bool = True):
        self.ocr_agent = OCRAgent()
        self.hybrid_agent = HybridExtractionAgent()
        self.rule_agent = RuleAnalysisAgent()
        self.azure_agent = AzureAnalysisAgent(use_llm=use_llm)
        self.validation_agent = ValidationAgent()

    def run(
        self,
        file_path: Optional[str] = None,
        raw_text: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_ext: Optional[str] = None,
    ) -> Dict[str, Any]:
        state = AgentResult(file_path=file_path, raw_text=raw_text)

        # Extract text from bytes if provided
        if file_bytes and file_ext:
            try:
                state.raw_text = OCRExtractor().extract_from_bytes(file_bytes, file_ext)
            except Exception as e:
                state.errors.append(f"Extraction error: {e}")

        # Run pipeline
        if not state.raw_text:
            state = self.ocr_agent.run(state)

        state = self.hybrid_agent.run(state)
        state = self.rule_agent.run(state)
        state = self.azure_agent.run(state)
        state = self.validation_agent.run(state)

        return {
            "patient_info":     state.parsed_data.get("patient_info", {}) if state.parsed_data else {},
            "test_results":     state.parsed_data.get("test_results", []) if state.parsed_data else [],
            "rule_analysis":    state.rule_analysis,
            "azure_analysis":   state.azure_analysis,
            "analysis_result":  state.azure_analysis or state.rule_analysis,
            "patient_report":   state.patient_report,
            "llm_explanation":  state.llm_explanation,
            "validation_notes": state.validation_notes,
            "errors":           state.errors,
            "raw_text":         state.raw_text,
        }