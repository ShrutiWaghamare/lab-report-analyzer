"""
Medical Analyzer Module — Fixed & Improved
Main orchestrator that coordinates all analysis modules and produces the final report.

FIXES APPLIED:
  1. Removed relative imports (from .module) → absolute (from core.module)
     so the module works both standalone and as a package.
  2. PatientData field default_factory: never pass None for optional list fields;
     omit the kwargs so the dataclass default kicks in correctly.
  3. generate_intelligent_summary returns a dict; key must be 'summary' not direct string.
  4. _determine_follow_up_needed — severity comparison now uses .value consistently.
  5. Added exc_info=True to logger.error for full tracebacks in production.
"""

import json
import datetime
import logging
from typing import Dict, List, Optional
from pathlib import Path

from core.test_evaluator import TestEvaluator, TestResult
from core.condition_analyzer import ConditionAnalyzer, Condition
from core.risk_assessor import AIRiskAssessor, RiskAssessment
from core.recommendation_engine import AIRecommendationEngine, Recommendations
from core.summary_generator import IntelligentSummaryGenerator, SummaryData
from core.emergency_checker import EmergencyChecker, PatientData

logger = logging.getLogger(__name__)


class MedicalAnalyzer:
    """Main orchestrator for the medical analysis pipeline."""

    def __init__(self):
        self.test_evaluator = TestEvaluator()
        self.condition_analyzer = ConditionAnalyzer()
        self.risk_assessor = AIRiskAssessor()
        self.recommendation_engine = AIRecommendationEngine()
        self.summary_generator = IntelligentSummaryGenerator()
        self.emergency_checker = EmergencyChecker()

    # ------------------------------------------------------------------ main

    def analyze(self, json_data: Dict) -> Dict:
        """
        Perform comprehensive medical analysis.

        Args:
            json_data: Dict with 'patient_info' and 'test_results' keys.

        Returns:
            Complete analysis report as a dict.
        """
        try:
            patient_info = json_data.get("patient_info", {})
            test_results = json_data.get("test_results", [])

            age = self._extract_age(patient_info.get("age", ""))
            gender = patient_info.get("gender", "").lower()

            logger.info(
                f"Analysis started — patient={patient_info.get('name', 'Unknown')}, "
                f"age={age}, gender={gender}, tests={len(test_results)}"
            )

            # 1. Evaluate individual test values
            evaluated_tests = self.test_evaluator.evaluate_tests(test_results, age, gender)
            abnormal_tests = self.test_evaluator.get_abnormal_tests(evaluated_tests)
            logger.info(f"Abnormal tests found: {len(abnormal_tests)}")

            # 2. Detect medical conditions
            conditions = self.condition_analyzer.detect_conditions(test_results, age, gender)
            logger.info(f"Conditions detected: {len(conditions)}")

            # 3. Assess health risks
            risks = self.risk_assessor.assess_risks(test_results, age, gender)
            logger.info(f"Risk assessments: {len(risks)}")

            # 4. Emergency checks
            # FIX: do NOT pass medical_history=None or medications=None —
            # the dataclass uses field(default_factory=list), so omitting them
            # gives an empty list correctly; passing None breaks iteration later.
            patient_data = PatientData(age=age, gender=gender, test_results=test_results)
            emergency_flags = self.emergency_checker.check_emergency_flags(patient_data)
            emergency_flags.extend(self.emergency_checker.check_combination_emergencies(patient_data))
            logger.info(f"Emergency flags: {len(emergency_flags)}")

            # 5. Recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                conditions, risks, abnormal_tests
            )

            # 6. Executive summary
            # FIX: generate_intelligent_summary returns {'summary': str, 'metrics': ...}
            summary_data = SummaryData(
                conditions=conditions,
                risks=risks,
                abnormal_tests=abnormal_tests,
                emergency_flags=emergency_flags,
            )
            summary_result = self.summary_generator.generate_intelligent_summary(summary_data, "medical")
            executive_summary = summary_result.get("summary", "Summary unavailable.")

            # 7. Follow-up requirements
            follow_up = self._determine_follow_up_needed(conditions, risks, emergency_flags)

            # 8. Compile final report
            report = {
                "patient_info": patient_info,
                "analysis_date": datetime.datetime.now().isoformat(),
                "abnormal_tests": abnormal_tests,
                "detected_conditions": self.condition_analyzer.conditions_to_dict(conditions),
                "risk_assessments": self.risk_assessor.risk_assessments_to_dict(risks),
                "recommendations": self.recommendation_engine.recommendations_to_dict(recommendations),
                "summary": executive_summary,
                "follow_up_needed": follow_up,
                "emergency_flags": self.emergency_checker.emergency_flags_to_dict(emergency_flags),
            }

            logger.info("Analysis completed successfully.")
            return report

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------ helpers

    def _extract_age(self, age_str: str) -> int:
        """Extract numeric age from strings like '33 years 8 months 20 days'."""
        try:
            if not age_str:
                return 0
            for part in str(age_str).split():
                if part.isdigit():
                    return int(part)
            return 0
        except Exception:
            return 0

    def _determine_follow_up_needed(
        self,
        conditions: List[Condition],
        risks: List[RiskAssessment],
        emergency_flags: List[Dict],
    ) -> Dict:
        """Determine follow-up priority, specialists, tests, and timeline."""

        # FIX: use .value for Enum comparison
        sev_values = [c.severity.value for c in conditions]

        if emergency_flags:
            priority = "URGENT — Within 24-48 hours"
        elif any(s in ("Severe", "Critical") for s in sev_values):
            priority = "HIGH — Within 1 week"
        elif "Moderate" in sev_values:
            priority = "MODERATE — Within 2-4 weeks"
        elif conditions or risks:
            priority = "ROUTINE — Within 3 months"
        else:
            priority = "NORMAL — Continue routine care"

        specialists: set = set()
        tests_to_repeat: set = set()

        condition_map = {
            "anemia": ("Hematologist", ["Complete Blood Count", "Iron Studies", "Vitamin B12"]),
            "diabetes": ("Endocrinologist", ["HbA1c", "Fasting Glucose"]),
            "cardiovascular": ("Cardiologist", ["Lipid Profile", "Blood Pressure"]),
            "inflammation": ("Rheumatologist", ["ESR", "CRP", "Complete Blood Count"]),
            "thyroid": ("Endocrinologist", ["TSH", "T3", "T4"]),
        }

        for cond in conditions:
            name_lower = cond.name.lower()
            for keyword, (specialist, tests) in condition_map.items():
                if keyword in name_lower:
                    specialists.add(specialist)
                    tests_to_repeat.update(tests)

        risk_map = {
            "cardiovascular": "Cardiologist",
            "diabetes": "Endocrinologist",
            "kidney": "Nephrologist",
        }
        for risk in risks:
            cat = risk.category.lower()
            for keyword, specialist in risk_map.items():
                if keyword in cat:
                    specialists.add(specialist)

        timeline: Dict[str, List[str]] = {
            "immediate": [],
            "short_term": [],
            "medium_term": [],
            "long_term": ["Routine health maintenance", "Repeat comprehensive metabolic panel"],
        }

        for cond in conditions:
            sev = cond.severity.value
            if sev in ("Severe", "Critical"):
                timeline["immediate"].append(f"Address {cond.name}")
            elif sev == "Moderate":
                timeline["short_term"].append(f"Follow up on {cond.name}")
            elif sev == "Mild":
                timeline["medium_term"].append(f"Monitor {cond.name}")

        return {
            "priority": priority,
            "specialists_needed": sorted(specialists),
            "tests_to_repeat": sorted(tests_to_repeat),
            "timeline": timeline,
        }

    # ------------------------------------------------------------------ I/O

    def save_analysis(self, analysis_result: Dict, output_path: str) -> None:
        """Save the analysis result dict to a JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Analysis saved to: {output_path}")

    def generate_patient_report(self, analysis_result: Dict) -> str:
        """Generate a patient-friendly markdown report from the analysis dict."""
        patient_info = analysis_result.get("patient_info", {})
        analysis_date = analysis_result.get("analysis_date", "")

        try:
            date_str = datetime.datetime.fromisoformat(analysis_date).strftime("%B %d, %Y")
        except Exception:
            date_str = "N/A"

        lines = [
            "# MEDICAL LAB ANALYSIS REPORT",
            "",
            f"**Patient:** {patient_info.get('name', 'N/A')}",
            f"**Age:** {patient_info.get('age', 'N/A')}",
            f"**Report Date:** {date_str}",
            "",
            "---",
            "",
            "## EXECUTIVE SUMMARY",
            analysis_result.get("summary", "No summary available."),
            "",
            "## EMERGENCY ALERTS",
        ]

        flags = analysis_result.get("emergency_flags", [])
        if flags:
            for flag in flags:
                lines.append(
                    f"**{flag.get('urgency', 'ALERT')}**: "
                    f"{flag.get('condition')} — {flag.get('action')}"
                )
        else:
            lines.append("No emergency conditions detected.")

        lines += ["", "## DETECTED CONDITIONS"]
        conditions = analysis_result.get("detected_conditions", [])
        if conditions:
            for cond in conditions:
                lines += [
                    f"### {cond['name']} ({cond['severity']})",
                    f"**Description:** {cond['description']}",
                    "**Possible Symptoms:**",
                ]
                for s in cond.get("symptoms", []):
                    lines.append(f"- {s}")
                lines.append("**Recommendations:**")
                for r in cond.get("recommendations", []):
                    lines.append(f"- {r}")
                lines.append(f"**Follow-up:** {cond.get('follow_up', 'N/A')}")
                lines.append("")
        else:
            lines.append("No specific conditions detected from lab results.")

        lines += ["", "## RISK ASSESSMENTS"]
        risks = analysis_result.get("risk_assessments", [])
        if risks:
            for risk in risks:
                lines += [f"### {risk['category']} — {risk['risk_level']} Risk", "**Risk Factors:**"]
                for factor in risk.get("factors", []):
                    lines.append(f"- {factor}")
                lines.append("**Recommendations:**")
                for r in risk.get("recommendations", []):
                    lines.append(f"- {r}")
                lines.append("")
        else:
            lines.append("No significant health risks identified.")

        follow_up = analysis_result.get("follow_up_needed", {})
        lines += [
            "",
            "## FOLLOW-UP REQUIREMENTS",
            f"**Priority:** {follow_up.get('priority', 'N/A')}",
            "",
            "**Specialists to Consider:**",
        ]
        for sp in follow_up.get("specialists_needed", []):
            lines.append(f"- {sp}")

        lines.append("\n**Tests to Repeat:**")
        for t in follow_up.get("tests_to_repeat", []):
            lines.append(f"- {t}")

        recs = analysis_result.get("recommendations", {})
        lines += ["", "## IMMEDIATE ACTIONS REQUIRED"]
        for action in recs.get("immediate_actions", []):
            lines.append(f"- {action}")

        lines += ["", "## LIFESTYLE RECOMMENDATIONS"]
        for change in recs.get("lifestyle_changes", []):
            lines.append(f"- {change}")

        lines += [
            "",
            "---",
            "## IMPORTANT DISCLAIMER",
            "This analysis is for educational purposes only and does not replace professional medical advice.",
            "Always consult a qualified healthcare professional for diagnosis and treatment.",
            "",
            f"**Generated by:** Medical Lab Analyzer v3.0  |  **Date:** {analysis_date}",
        ]

        return "\n".join(lines)
