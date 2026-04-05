"""
Enhanced Summary Generator Module with AI/ML Capabilities
Generates intelligent plain text/markdown summaries for HR or medical professionals.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class Severity(Enum):
    NORMAL = "Normal"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class Condition:
    name: str
    severity: Severity
    description: str
    follow_up: str
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    affected_systems: List[str] = field(default_factory=list)

@dataclass
class RiskAssessment:
    category: str
    risk_level: RiskLevel
    score: float
    factors: List[str]
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 1.0
    time_horizon: str = "short-term"

@dataclass
class SummaryData:
    conditions: List[Condition] = field(default_factory=list)
    risks: List[RiskAssessment] = field(default_factory=list)
    abnormal_tests: List[Dict] = field(default_factory=list)
    emergency_flags: List[Dict] = field(default_factory=list)
    patient_demographics: Dict = field(default_factory=dict)
    analysis_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SummaryMetrics:
    """Metrics for summary quality assessment"""
    readability_score: float
    complexity_grade: float
    word_count: int
    sentence_count: int
    key_topics: List[str]
    confidence_score: float

class IntelligentSummaryGenerator:
    """Enhanced summary generator with AI/ML capabilities."""
    
    def __init__(self, enable_ml: bool = True):
        self.enable_ml = enable_ml
        self.summary_templates = self._load_summary_templates()
        self.medical_terminology = self._load_medical_terminology()
        self.severity_weights = self._load_severity_weights()
        
        # Initialize ML components
        if self.enable_ml:
            self._initialize_ml_components()
    
    def _load_summary_templates(self) -> Dict[str, Dict]:
        """Load enhanced summary templates for different scenarios and audiences"""
        return {
            "medical": {
                "normal": "Laboratory analysis indicates parameters within established reference ranges. Recommend continued routine monitoring per standard protocols.",
                "mild_abnormalities": "Minor deviations from reference ranges detected. Consider lifestyle interventions and reassessment in 3-6 months.",
                "moderate_concerns": "Moderate abnormalities identified requiring clinical correlation and targeted intervention. Follow-up recommended within 2-4 weeks.",
                "severe_issues": "Significant pathological findings detected. Immediate clinical evaluation and intervention required.",
                "critical": "Critical laboratory values identified. Emergency medical assessment and immediate intervention indicated."
            },
            "patient": {
                "normal": "Your test results look good and fall within normal ranges. Keep up your healthy habits!",
                "mild_abnormalities": "Your results show some minor changes that can often be improved with lifestyle adjustments.",
                "moderate_concerns": "Some of your test results need attention. Please schedule a follow-up with your doctor to discuss next steps.",
                "severe_issues": "Your test results show some concerning changes that need prompt medical attention.",
                "critical": "Your test results require immediate medical attention. Please contact your healthcare provider right away."
            },
            "hr": {
                "normal": "Employee health screening results indicate no significant health risks that would impact work performance.",
                "mild_abnormalities": "Minor health indicators noted. Wellness program participation recommended.",
                "moderate_concerns": "Moderate health concerns identified. Consider occupational health consultation.",
                "severe_issues": "Significant health issues requiring medical clearance before return to work.",
                "critical": "Critical health findings requiring immediate medical intervention and work restrictions."
            }
        }
    
    def _load_medical_terminology(self) -> Dict[str, str]:
        """Load medical terminology mappings for patient-friendly language"""
        return {
            "hypertension": "high blood pressure",
            "hyperlipidemia": "high cholesterol",
            "diabetes mellitus": "diabetes",
            "myocardial infarction": "heart attack",
            "cerebrovascular accident": "stroke",
            "renal insufficiency": "kidney problems",
            "hepatic dysfunction": "liver problems",
            "anemia": "low red blood cell count",
            "leukocytosis": "high white blood cell count",
            "thrombocytopenia": "low platelet count"
        }
    
    def _load_severity_weights(self) -> Dict[Severity, float]:
        """Load severity weighting for prioritization"""
        return {
            Severity.NORMAL: 0.0,
            Severity.MILD: 1.0,
            Severity.MODERATE: 3.0,
            Severity.SEVERE: 7.0,
            Severity.CRITICAL: 10.0
        }
    
    def _initialize_ml_components(self):
        """Initialize machine learning components"""
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            # Don't initialize topic_model here - will be created dynamically
            self.topic_model = None
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ML components: {e}")
            self.enable_ml = False
    
    def generate_intelligent_summary(self, summary_data: SummaryData, 
                                   target_audience: str = "medical",
                                   max_length: Optional[int] = None) -> Dict[str, Union[str, SummaryMetrics]]:
        """
        Generate intelligent summary using AI/ML techniques.
        
        Args:
            summary_data: SummaryData object containing all analysis results
            target_audience: Target audience ('medical', 'patient', 'hr')
            max_length: Maximum length of summary in characters
            
        Returns:
            Dictionary containing summary and metrics
        """
        try:
            # Validate input
            if not self._validate_summary_data(summary_data):
                raise ValueError("Invalid summary data provided")
            
            # Determine priority and complexity
            priority_score = self._calculate_priority_score(summary_data)
            complexity_level = self._assess_complexity(summary_data)
            
            # Generate context-aware summary
            summary_text = self._generate_context_aware_summary(
                summary_data, target_audience, priority_score, complexity_level
            )
            
            # Apply length constraints if specified
            if max_length:
                summary_text = self._truncate_intelligently(summary_text, max_length)
            
            # Generate summary metrics
            metrics = self._calculate_summary_metrics(summary_text, summary_data)
            
            # Apply ML enhancements if enabled
            if self.enable_ml:
                summary_text = self._enhance_with_ml(summary_text, summary_data, target_audience)
                metrics = self._update_metrics_with_ml(metrics, summary_text)
            
            return {
                "summary": summary_text,
                "metrics": metrics,
                "audience": target_audience,
                "priority_score": priority_score,
                "complexity_level": complexity_level
            }
            
        except Exception as e:
            logger.error(f"Error generating intelligent summary: {e}")
            return {
                "summary": self._generate_fallback_summary(summary_data),
                "metrics": None,
                "error": str(e)
            }
    
    def _validate_summary_data(self, summary_data: SummaryData) -> bool:
        """Validate summary data structure"""
        if not isinstance(summary_data, SummaryData):
            return False
        
        # Check if at least some data is present
        has_data = (
            len(summary_data.conditions) > 0 or
            len(summary_data.risks) > 0 or
            len(summary_data.abnormal_tests) > 0 or
            len(summary_data.emergency_flags) > 0
        )
        
        return has_data
    
    def _calculate_priority_score(self, summary_data: SummaryData) -> float:
        """Calculate priority score based on severity and urgency"""
        score = 0.0
        
        # Emergency flags have highest priority
        score += len(summary_data.emergency_flags) * 10.0
        
        # Condition severity scoring
        for condition in summary_data.conditions:
            weight = self.severity_weights.get(condition.severity, 0)
            confidence_factor = condition.confidence_score
            score += weight * confidence_factor
        
        # Risk level scoring
        risk_weights = {
            RiskLevel.LOW: 0.5,
            RiskLevel.MODERATE: 2.0,
            RiskLevel.HIGH: 5.0,
            RiskLevel.CRITICAL: 8.0
        }
        
        for risk in summary_data.risks:
            weight = risk_weights.get(risk.risk_level, 0)
            confidence_factor = risk.confidence
            score += weight * confidence_factor
        
        # Critical test results
        critical_tests = [t for t in summary_data.abnormal_tests 
                         if t.get("severity") in ["Severe", "Critical"]]
        score += len(critical_tests) * 3.0
        
        return min(score, 100.0)  # Cap at 100
    
    def _assess_complexity(self, summary_data: SummaryData) -> str:
        """Assess the complexity of the medical case"""
        complexity_factors = 0
        
        # Multiple conditions increase complexity
        complexity_factors += len(summary_data.conditions)
        
        # Multiple affected systems
        all_systems = []
        for condition in summary_data.conditions:
            # Note: condition.affected_systems not available in current Condition class
            # all_systems.extend(condition.affected_systems)
            pass
        unique_systems = len(set(all_systems))
        complexity_factors += unique_systems
        
        # Multiple risk categories
        complexity_factors += len(set(risk.category for risk in summary_data.risks))
        
        if complexity_factors <= 2:
            return "simple"
        elif complexity_factors <= 5:
            return "moderate"
        else:
            return "complex"
    
    def _generate_context_aware_summary(self, summary_data: SummaryData, 
                                      target_audience: str, 
                                      priority_score: float,
                                      complexity_level: str) -> str:
        """Generate context-aware summary based on audience and complexity"""
        
        # Determine summary style based on priority and audience
        if priority_score >= 70:
            urgency_level = "critical"
        elif priority_score >= 40:
            urgency_level = "severe_issues"
        elif priority_score >= 15:
            urgency_level = "moderate_concerns"
        elif priority_score > 0:
            urgency_level = "mild_abnormalities"
        else:
            urgency_level = "normal"
        
        # Get base template
        base_template = self.summary_templates.get(target_audience, {}).get(urgency_level, "")
        
        # Build comprehensive summary
        summary_parts = [base_template] if base_template else []
        
        # Add emergency alerts
        if summary_data.emergency_flags:
            emergency_text = self._format_emergency_alerts(summary_data.emergency_flags, target_audience)
            summary_parts.insert(0, emergency_text)
        
        # Add key findings
        key_findings = self._extract_prioritized_findings(summary_data, target_audience)
        if key_findings:
            summary_parts.append(key_findings)
        
        # Add recommendations based on complexity
        recommendations = self._generate_smart_recommendations(summary_data, complexity_level, target_audience)
        if recommendations:
            summary_parts.append(recommendations)
        
        # Add follow-up timeline
        follow_up = self._determine_follow_up_timeline(summary_data, priority_score)
        if follow_up:
            summary_parts.append(follow_up)
        
        return " ".join(summary_parts)
    
    def _format_emergency_alerts(self, emergency_flags: List[Dict], audience: str) -> str:
        """Format emergency alerts for specific audience"""
        if audience == "patient":
            return f"⚠️ URGENT: Please seek immediate medical attention for {len(emergency_flags)} critical finding(s)."
        elif audience == "hr":
            return f"⚠️ CRITICAL: Employee requires immediate medical intervention - {len(emergency_flags)} emergency finding(s) detected."
        else:
            conditions = [flag.condition for flag in emergency_flags]
            return f"🚨 EMERGENCY: {len(emergency_flags)} critical findings require immediate intervention: {', '.join(conditions)}"
    
    def _extract_prioritized_findings(self, summary_data: SummaryData, audience: str) -> str:
        """Extract and prioritize key findings based on audience with specific test values"""
        findings = []
        
        # Extract specific abnormal test values for detailed summary
        if summary_data.abnormal_tests:
            key_tests = []
            
            # Look for specific important tests
            test_priorities = [
                ("pcv", "PCV"),
                ("rdw", "RDW"), 
                ("rdw_cv", "RDW (CV)"),
                ("ldl_cholesterol", "LDL Cholesterol"),
                ("lymphocytes", "Lymphocytes"),
                ("hemoglobin", "Hemoglobin"),
                ("t3", "T3"),
                ("t4", "T4"),
                ("tsh", "TSH")
            ]
            
            for test in summary_data.abnormal_tests:
                test_name_lower = test.get("test_name", "").lower().replace(" ", "_")
                
                for priority_test, display_name in test_priorities:
                    if priority_test in test_name_lower:
                        value = test.get("value")
                        unit = test.get("unit", "")
                        status = test.get("status", "")
                        ref_range = test.get("reference_range", "")
                        
                        if value is not None and status != "normal":
                            if audience == "patient":
                                # Patient-friendly format
                                if status == "high":
                                    key_tests.append(f"high {display_name} ({value} {unit})")
                                elif status == "low":
                                    key_tests.append(f"low {display_name} ({value} {unit})")
                            else:
                                # Medical format
                                key_tests.append(f"{display_name}: {value} {unit} ({status})")
                            break
            
            # Add the most important findings
            if key_tests:
                if audience == "patient":
                    findings.append(f"Laboratory analysis shows mostly normal parameters with some mild abnormalities: {', '.join(key_tests[:4])}.")
                else:
                    findings.append(f"Key abnormal findings: {', '.join(key_tests[:4])}.")
        
        # Add condition-based findings
        if summary_data.conditions:
            condition_descriptions = []
            for condition in summary_data.conditions:
                if audience == "patient":
                    condition_name = self.medical_terminology.get(condition.name.lower(), condition.name)
                    condition_descriptions.append(f"mild {condition_name.lower()}")
                else:
                    condition_descriptions.append(f"{condition.name} ({condition.severity.value})")
            
            if condition_descriptions:
                if audience == "patient":
                    findings.append(f"These findings suggest {', '.join(condition_descriptions)}.")
                else:
                    findings.append(f"Detected conditions: {', '.join(condition_descriptions)}.")
        
        # Add risk assessment
        if summary_data.risks:
            risk_levels = [risk.risk_level.value for risk in summary_data.risks]
            if audience == "patient":
                findings.append(f"Risk assessment shows: {', '.join(set(risk_levels))}.")
            else:
                findings.append(f"Risk assessment: {', '.join(set(risk_levels))}.")
        
        # Add emergency status
        if not summary_data.emergency_flags:
            if audience == "patient":
                findings.append("No urgent or critical values detected.")
            else:
                findings.append("No emergency flags detected.")
        
        return " ".join(findings)
    
    def _generate_smart_recommendations(self, summary_data: SummaryData, 
                                      complexity_level: str, audience: str) -> str:
        """Generate intelligent recommendations based on complexity and audience"""
        all_recommendations = []
        
        # Collect recommendations from conditions and risks
        for condition in summary_data.conditions:
            all_recommendations.extend(condition.recommendations)
        
        for risk in summary_data.risks:
            all_recommendations.extend(risk.recommendations)
        
        # Add specific recommendations based on abnormal tests
        if summary_data.abnormal_tests:
            # Check for specific conditions and add targeted recommendations
            has_anemia = any("pcv" in test.get("test_name", "").lower() and test.get("status") == "low" 
                            for test in summary_data.abnormal_tests)
            has_rdw = any("rdw" in test.get("test_name", "").lower() and test.get("status") == "high" 
                         for test in summary_data.abnormal_tests)
            has_ldl = any("ldl" in test.get("test_name", "").lower() and test.get("status") == "high" 
                         for test in summary_data.abnormal_tests)
            has_lymphocytosis = any("lymphocytes" in test.get("test_name", "").lower() and test.get("status") == "high" 
                                   for test in summary_data.abnormal_tests)
            
            if has_anemia:
                all_recommendations.append("repeat CBC in 2-4 weeks")
            if has_ldl:
                all_recommendations.append("follow lifestyle modifications for cardiovascular health")
            if has_rdw:
                all_recommendations.append("consider iron studies")
            if has_lymphocytosis:
                all_recommendations.append("monitor for infection or inflammation")
        
        if not all_recommendations:
            all_recommendations.append("Consult physician if persistent or symptomatic")
        
        # Prioritize and deduplicate
        unique_recommendations = list(set(all_recommendations))
        
        # Limit based on complexity
        max_recs = {"simple": 2, "moderate": 3, "complex": 5}.get(complexity_level, 3)
        top_recommendations = unique_recommendations[:max_recs]
        
        if audience == "patient":
            prefix = "Recommended actions:"
        else:
            prefix = "Recommended actions:"
        
        return f"{prefix} {', '.join(top_recommendations)}"
    
    def _determine_follow_up_timeline(self, summary_data: SummaryData, priority_score: float) -> str:
        """Determine appropriate follow-up timeline"""
        if priority_score >= 70:
            return "Follow-up: Immediate medical attention required."
        elif priority_score >= 40:
            return "Follow-up: Within 24-48 hours."
        elif priority_score >= 15:
            return "Follow-up: Within 1-2 weeks."
        elif priority_score > 0:
            return "Follow-up: Routine monitoring and repeat relevant tests as per recommendations."
        else:
            return "Follow-up: Routine monitoring as per standard protocols."
    
    def _enhance_with_ml(self, summary_text: str, summary_data: SummaryData, audience: str) -> str:
        """Enhance summary using ML techniques"""
        try:
            # Extract key topics using ML
            topics = self._extract_topics_ml(summary_text, summary_data)
            
            # Optimize readability
            optimized_text = self._optimize_readability(summary_text, audience)
            
            # Add ML-generated insights
            insights = self._generate_ml_insights(summary_data)
            
            if insights:
                optimized_text += f" {insights}"
            
            return optimized_text
            
        except Exception as e:
            logger.warning(f"ML enhancement failed: {e}")
            return summary_text
    
    def _extract_topics_ml(self, text: str, summary_data: SummaryData) -> List[str]:
        """Extract key topics using machine learning"""
        return self._extract_key_topics(text, summary_data, n_topics=3)
    
    def _extract_key_topics(self, text, summary_data, n_topics=3):
        try:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Adjust n_topics based on available sentences
            actual_n_topics = min(n_topics, max(1, len(sentences) - 1))
            
            if len(sentences) < 2:
                return sentences[:1] if sentences else ["general medical findings"]
                
            # Combine text with condition descriptions
            all_text = [text]
            for condition in summary_data.conditions:
                all_text.append(condition.description)
            
            # Ensure we have enough samples for clustering
            if len(all_text) < 2:
                return ["medical findings", "patient care", "follow-up required"]
            
            # Dynamically set n_clusters based on available samples
            n_clusters = min(actual_n_topics, len(all_text), 3)  # Max 3 clusters
            
            # Vectorize
            vectors = self.vectorizer.fit_transform(all_text)
            
            # Create KMeans model with appropriate n_clusters
            topic_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = topic_model.fit_predict(vectors)
            
            # Extract representative terms for each cluster
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for cluster_id in set(clusters):
                cluster_center = topic_model.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-3:][::-1]
                cluster_topics = [feature_names[i] for i in top_indices]
                topics.extend(cluster_topics)
            
            return list(set(topics))[:n_clusters]  # Return top n_clusters unique topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return ["medical findings", "patient care", "follow-up required"]
    
    def _optimize_readability(self, text: str, audience: str) -> str:
        """Optimize text readability for target audience"""
        if audience == "patient":
            # Add actual text simplification logic
            simplified = text.replace("exhibits", "shows")
            simplified = simplified.replace("necessitating", "needing")
            simplified = simplified.replace("cardiovascular", "heart")
            simplified = simplified.replace("abnormalities", "problems")
            simplified = simplified.replace("elevated", "high")
            simplified = simplified.replace("decreased", "low")
            simplified = simplified.replace("concentration", "level")
            simplified = simplified.replace("hemoglobin", "blood oxygen")
            simplified = simplified.replace("cholesterol", "fat in blood")
            simplified = simplified.replace("triglycerides", "blood fats")
            return simplified
        return text  # Return original for medical audience
    
    def _generate_ml_insights(self, summary_data: SummaryData) -> str:
        """Generate ML-based insights"""
        try:
            insights = []
            if summary_data.conditions:
                insights.append(f"Detected {len(summary_data.conditions)} medical conditions")
            if summary_data.risks:
                risk_levels = [risk.risk_level.value for risk in summary_data.risks]
                insights.append(f"Risk assessment shows: {', '.join(set(risk_levels))}")
            return ". ".join(insights) + "." if insights else "Standard medical evaluation completed."
        except Exception:
            return "Medical analysis completed with standard protocols."
    
    def _calculate_summary_metrics(self, summary_text: str, summary_data: SummaryData) -> SummaryMetrics:
        """Calculate comprehensive summary metrics"""
        try:
            # Basic text metrics
            word_count = len(summary_text.split())
            sentences = nltk.sent_tokenize(summary_text)
            sentence_count = len(sentences)
            
            # Readability metrics
            readability = flesch_reading_ease(summary_text)
            complexity = flesch_kincaid_grade(summary_text)
            
            # Extract key topics (simplified)
            words = summary_text.lower().split()
            key_topics = [word for word in set(words) if len(word) > 4 and word.isalpha()][:5]
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(summary_data)
            
            return SummaryMetrics(
                readability_score=readability,
                complexity_grade=complexity,
                word_count=word_count,
                sentence_count=sentence_count,
                key_topics=key_topics,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
            return SummaryMetrics(0, 0, 0, 0, [], 0.0)
    
    def _calculate_confidence_score(self, summary_data: SummaryData) -> float:
        """Calculate overall confidence score for the summary"""
        if not summary_data.conditions and not summary_data.risks:
            return 1.0
        
        # Average confidence from conditions and risks
        all_scores = []
        
        for condition in summary_data.conditions:
            all_scores.append(condition.confidence_score)
        
        for risk in summary_data.risks:
            all_scores.append(risk.confidence)
        
        return sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    def _update_metrics_with_ml(self, metrics: SummaryMetrics, text: str) -> SummaryMetrics:
        """Update metrics with ML-enhanced analysis"""
        try:
            # Enhanced topic extraction if ML is available
            if self.enable_ml and hasattr(self, 'vectorizer'):
                vectors = self.vectorizer.fit_transform([text])
                feature_names = self.vectorizer.get_feature_names_out()
                scores = vectors.toarray()[0]
                top_indices = scores.argsort()[-5:][::-1]
                ml_topics = [feature_names[i] for i in top_indices if scores[i] > 0]
                metrics.key_topics = ml_topics
            
            return metrics
            
        except Exception as e:
            logger.warning(f"ML metrics update failed: {e}")
            return metrics
    
    def _truncate_intelligently(self, text: str, max_length: int) -> str:
        """Intelligently truncate text while preserving meaning"""
        if len(text) <= max_length:
            return text
        
        # Find last complete sentence within limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.7:  # If we can keep most of the text
            return truncated[:last_period + 1]
        else:
            # Find last complete word
            last_space = truncated.rfind(' ')
            if last_space > 0:
                return truncated[:last_space] + "..."
            else:
                return text[:max_length-3] + "..."
    
    def _generate_fallback_summary(self, summary_data: SummaryData) -> str:
        """Generate basic fallback summary when AI/ML fails"""
        if not summary_data.conditions and not summary_data.abnormal_tests:
            return "Analysis completed. No significant abnormalities detected."
        
        findings = []
        if summary_data.emergency_flags:
            findings.append(f"{len(summary_data.emergency_flags)} emergency finding(s) require immediate attention")
        
        if summary_data.conditions:
            condition_names = [c.name for c in summary_data.conditions[:3]]
            findings.append(f"Conditions detected: {', '.join(condition_names)}")
        
        return ". ".join(findings) + ". Detailed analysis recommended."

    def generate_comparative_summary(self, current_data: SummaryData, 
                                   historical_data: List[SummaryData]) -> str:
        """Generate comparative summary showing trends over time"""
        if not historical_data:
            return self.generate_intelligent_summary(current_data)["summary"]
        
        # Compare current with most recent historical data
        prev_data = historical_data[-1]
        
        improvements = []
        deteriorations = []
        new_conditions = []
        
        # Compare conditions
        prev_condition_names = {c.name for c in prev_data.conditions}
        current_condition_names = {c.name for c in current_data.conditions}
        
        new_conditions = list(current_condition_names - prev_condition_names)
        resolved_conditions = list(prev_condition_names - current_condition_names)
        
        summary_parts = []
        
        if new_conditions:
            summary_parts.append(f"New findings: {', '.join(new_conditions)}")
        
        if resolved_conditions:
            summary_parts.append(f"Resolved: {', '.join(resolved_conditions)}")
        
        # Add current status
        current_summary = self.generate_intelligent_summary(current_data)["summary"]
        summary_parts.append(f"Current status: {current_summary}")
        
        return " | ".join(summary_parts)

    def export_summary_report(self, summary_data: SummaryData, 
                            format_type: str = "markdown") -> str:
        """Export comprehensive summary report in specified format"""
        
        medical_summary = self.generate_intelligent_summary(summary_data, "medical")
        patient_summary = self.generate_intelligent_summary(summary_data, "patient")
        
        if format_type.lower() == "markdown":
            return self._export_markdown_report(summary_data, medical_summary, patient_summary)
        elif format_type.lower() == "html":
            return self._export_html_report(summary_data, medical_summary, patient_summary)
        else:
            return self._export_text_report(summary_data, medical_summary, patient_summary)
    
    def _export_markdown_report(self, summary_data: SummaryData, 
                              medical_summary: Dict, patient_summary: Dict) -> str:
        """Export report in markdown format"""
        report = f"""# Medical Analysis Report

**Generated:** {summary_data.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

**Priority Score:** {medical_summary.get('priority_score', 0):.1f}/100  
**Complexity:** {medical_summary.get('complexity_level', 'Unknown')}

### Medical Professional Summary
{medical_summary['summary']}

### Patient-Friendly Summary  
{patient_summary['summary']}

## Detailed Findings

"""
        
        if summary_data.emergency_flags:
            report += "### 🚨 Emergency Flags\n"
            for flag in summary_data.emergency_flags:
                report += f"- **{flag.condition}**: {flag.action}\n"
            report += "\n"
        
        if summary_data.conditions:
            report += "### Detected Conditions\n"
            for condition in summary_data.conditions:
                report += f"- **{condition.name}** ({condition.severity.value})\n"
                report += f"  - Description: {condition.description}\n"
                report += f"  - Confidence: {condition.confidence_score:.2f}\n"
                if condition.recommendations:
                    report += f"  - Recommendations: {', '.join(condition.recommendations[:2])}\n"
                report += "\n"
        
        if summary_data.risks:
            report += "### Risk Assessments\n"
            for risk in summary_data.risks:
                report += f"- **{risk.category}** - {risk.risk_level.value} Risk\n"
                report += f"  - Score: {risk.score:.2f}\n"
                report += f"  - Factors: {', '.join(risk.factors)}\n"
                report += f"  - Time Horizon: {risk.time_horizon}\n"
                report += "\n"
        
        # Add metrics if available
        if medical_summary.get('metrics'):
            metrics = medical_summary['metrics']
            report += f"""## Summary Quality Metrics

- **Readability Score:** {metrics.readability_score:.1f}
- **Complexity Grade:** {metrics.complexity_grade:.1f}  
- **Word Count:** {metrics.word_count}
- **Confidence:** {metrics.confidence_score:.2f}
- **Key Topics:** {', '.join(metrics.key_topics)}

"""
        
        return report


class SummaryAnalytics:
    """Advanced analytics for summary generation and quality assessment"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.usage_statistics = {}
    
    def analyze_summary_effectiveness(self, summary_data: SummaryData, 
                                    generated_summaries: Dict[str, str]) -> Dict:
        """Analyze the effectiveness of generated summaries"""
        
        analytics = {
            "coverage_analysis": self._analyze_coverage(summary_data, generated_summaries),
            "consistency_score": self._calculate_consistency_score(generated_summaries),
            "completeness_score": self._calculate_completeness_score(summary_data, generated_summaries),
            "audience_appropriateness": self._assess_audience_appropriateness(generated_summaries),
            "actionability_score": self._calculate_actionability_score(summary_data, generated_summaries)
        }
        
        return analytics
    
    def _analyze_coverage(self, summary_data: SummaryData, summaries: Dict[str, str]) -> Dict:
        """Analyze how well summaries cover all important findings"""
        
        # Extract key elements that should be covered
        important_elements = set()
        
        # Critical conditions
        for condition in summary_data.conditions:
            if condition.severity in [Severity.SEVERE, Severity.CRITICAL]:
                important_elements.add(condition.name.lower())
        
        # Emergency flags
        for flag in summary_data.emergency_flags:
            important_elements.add(flag.condition.lower() if flag.condition else "")
        
        # High risks
        for risk in summary_data.risks:
            if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                important_elements.add(risk.category.lower())
        
        coverage_scores = {}
        
        for audience, summary_text in summaries.items():
            summary_lower = summary_text.lower()
            covered_elements = sum(1 for element in important_elements 
                                 if element in summary_lower)
            
            coverage_score = (covered_elements / len(important_elements) 
                            if important_elements else 1.0)
            coverage_scores[audience] = coverage_score
        
        return {
            "total_important_elements": len(important_elements),
            "coverage_by_audience": coverage_scores,
            "average_coverage": sum(coverage_scores.values()) / len(coverage_scores)
        }
    
    def _calculate_consistency_score(self, summaries: Dict[str, str]) -> float:
        """Calculate consistency between different audience summaries"""
        
        if len(summaries) < 2:
            return 1.0
        
        # Simple consistency check based on key medical terms
        medical_terms = []
        for summary in summaries.values():
            words = summary.lower().split()
            # Extract potential medical terms (longer words)
            terms = [word for word in words if len(word) > 6 and word.isalpha()]
            medical_terms.append(set(terms))
        
        # Calculate overlap between term sets
        if not medical_terms:
            return 1.0
        
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(medical_terms)):
            for j in range(i + 1, len(medical_terms)):
                set1, set2 = medical_terms[i], medical_terms[j]
                if set1 or set2:
                    overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                    total_overlap += overlap
                    comparisons += 1
        
        return total_overlap / comparisons if comparisons > 0 else 1.0
    
    def _calculate_completeness_score(self, summary_data: SummaryData, 
                                    summaries: Dict[str, str]) -> Dict:
        """Calculate completeness score for each summary type"""
        
        completeness_scores = {}
        
        for audience, summary_text in summaries.items():
            score = 0.0
            max_score = 0.0
            
            # Check for emergency flags coverage
            if summary_data.emergency_flags:
                max_score += 3.0
                if any(flag.condition.lower() in summary_text.lower() 
                      for flag in summary_data.emergency_flags):
                    score += 3.0
            
            # Check for severe conditions coverage
            severe_conditions = [c for c in summary_data.conditions 
                               if c.severity in [Severity.SEVERE, Severity.CRITICAL]]
            if severe_conditions:
                max_score += 2.0
                covered_severe = sum(1 for condition in severe_conditions
                                   if condition.name.lower() in summary_text.lower())
                score += 2.0 * (covered_severe / len(severe_conditions))
            
            # Check for high risks coverage
            high_risks = [r for r in summary_data.risks 
                         if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if high_risks:
                max_score += 1.0
                covered_risks = sum(1 for risk in high_risks
                                  if risk.category.lower() in summary_text.lower())
                score += 1.0 * (covered_risks / len(high_risks))
            
            # Check for recommendations presence
            has_recommendations = any(condition.recommendations for condition in summary_data.conditions)
            if has_recommendations:
                max_score += 1.0
                if any(word in summary_text.lower() 
                      for word in ["recommend", "suggest", "action", "follow", "next"]):
                    score += 1.0
            
            completeness_scores[audience] = score / max_score if max_score > 0 else 1.0
        
        return completeness_scores
    
    def _assess_audience_appropriateness(self, summaries: Dict[str, str]) -> Dict:
        """Assess how appropriate each summary is for its target audience"""
        
        appropriateness_scores = {}
        
        for audience, summary_text in summaries.items():
            score = 0.0
            
            if audience == "patient":
                # Patient summaries should be less technical
                technical_terms = ["pathological", "etiology", "differential", "protocol"]
                friendly_terms = ["condition", "health", "doctor", "treatment"]
                
                technical_count = sum(1 for term in technical_terms 
                                    if term in summary_text.lower())
                friendly_count = sum(1 for term in friendly_terms 
                                   if term in summary_text.lower())
                
                # Lower technical terms and higher friendly terms = better for patients
                score = max(0, 1.0 - (technical_count * 0.2) + (friendly_count * 0.1))
                
            elif audience == "medical":
                # Medical summaries should be more technical and precise
                medical_terms = ["clinical", "assessment", "intervention", "protocol", 
                               "evaluation", "diagnosis", "prognosis"]
                medical_count = sum(1 for term in medical_terms 
                                  if term in summary_text.lower())
                
                score = min(1.0, medical_count * 0.15 + 0.4)
                
            elif audience == "hr":
                # HR summaries should focus on work-related implications
                hr_terms = ["work", "employment", "clearance", "restrictions", 
                          "fitness", "occupational"]
                hr_count = sum(1 for term in hr_terms 
                             if term in summary_text.lower())
                
                score = min(1.0, hr_count * 0.2 + 0.5)
            
            appropriateness_scores[audience] = max(0.0, min(1.0, score))
        
        return appropriateness_scores
    
    def _calculate_actionability_score(self, summary_data: SummaryData, 
                                     summaries: Dict[str, str]) -> Dict:
        """Calculate how actionable each summary is"""
        
        actionability_scores = {}
        
        action_words = ["contact", "schedule", "follow", "seek", "continue", 
                       "monitor", "discuss", "consider", "recommend", "urgent"]
        
        for audience, summary_text in summaries.items():
            summary_lower = summary_text.lower()
            
            # Count action words
            action_count = sum(1 for word in action_words if word in summary_lower)
            
            # Check for specific timelines
            has_timeline = any(timeline in summary_lower 
                             for timeline in ["immediate", "24", "48", "week", "month"])
            
            # Check for emergency indicators
            has_urgency = any(urgent in summary_lower 
                            for urgent in ["urgent", "emergency", "critical", "immediate"])
            
            base_score = min(1.0, action_count * 0.1)
            
            if has_timeline:
                base_score += 0.2
            
            if has_urgency and (summary_data.emergency_flags or 
                               any(c.severity == Severity.CRITICAL for c in summary_data.conditions)):
                base_score += 0.3
            
            actionability_scores[audience] = min(1.0, base_score)
        
        return actionability_scores


class SummaryQualityAssurance:
    """Quality assurance system for generated summaries"""
    
    def __init__(self):
        self.quality_thresholds = {
            "readability_min": 30.0,  # Flesch reading ease minimum
            "readability_max": 90.0,  # Flesch reading ease maximum
            "max_sentence_length": 25,  # Maximum words per sentence
            "max_paragraph_length": 150,  # Maximum words per paragraph
            "min_confidence": 0.7,  # Minimum confidence score
            "max_complexity_grade": 14  # Maximum Flesch-Kincaid grade
        }
    
    def validate_summary_quality(self, summary_text: str, 
                               target_audience: str,
                               summary_metrics: SummaryMetrics) -> Dict:
        """Comprehensive quality validation of generated summary"""
        
        validation_results = {
            "overall_quality": "PASS",
            "issues": [],
            "warnings": [],
            "suggestions": [],
            "quality_score": 0.0
        }
        
        quality_checks = [
            self._check_readability(summary_text, target_audience, summary_metrics),
            self._check_completeness(summary_text),
            self._check_accuracy_indicators(summary_text, summary_metrics),
            self._check_structure_quality(summary_text),
            self._check_audience_appropriateness(summary_text, target_audience),
            self._check_medical_terminology_usage(summary_text, target_audience)
        ]
        
        # Aggregate results
        total_score = 0
        issue_count = 0
        
        for check_result in quality_checks:
            total_score += check_result["score"]
            validation_results["issues"].extend(check_result.get("issues", []))
            validation_results["warnings"].extend(check_result.get("warnings", []))
            validation_results["suggestions"].extend(check_result.get("suggestions", []))
            
            if check_result.get("critical_failure"):
                validation_results["overall_quality"] = "FAIL"
                issue_count += 1
        
        validation_results["quality_score"] = total_score / len(quality_checks)
        
        # Determine overall quality
        if validation_results["quality_score"] < 0.6 or issue_count > 0:
            validation_results["overall_quality"] = "FAIL"
        elif validation_results["quality_score"] < 0.8 or len(validation_results["warnings"]) > 2:
            validation_results["overall_quality"] = "WARNING"
        
        return validation_results
    
    def _check_readability(self, text: str, audience: str, metrics: SummaryMetrics) -> Dict:
        """Check readability appropriateness for target audience"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        target_grades = {"patient": 8, "hr": 10, "medical": 12}
        target_grade = target_grades.get(audience, 10)
        
        if metrics.complexity_grade > target_grade + 3:
            result["issues"].append(f"Text too complex for {audience} audience")
            result["score"] = 0.0
            result["critical_failure"] = True
        elif metrics.complexity_grade > target_grade + 1:
            result["warnings"].append(f"Text complexity borderline high for {audience}")
            result["score"] = 0.7
        
        if metrics.readability_score < 30:
            result["warnings"].append("Text may be difficult to read")
            result["score"] *= 0.8
        
        # Check sentence length
        sentences = nltk.sent_tokenize(text)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        
        if len(long_sentences) > len(sentences) * 0.3:  # More than 30% long sentences
            result["suggestions"].append("Consider breaking up long sentences")
            result["score"] *= 0.9
        
        return result
    
    def _check_completeness(self, text: str) -> Dict:
        """Check if summary appears complete and comprehensive"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        # Check minimum length
        if len(text) < 50:
            result["issues"].append("Summary too short to be comprehensive")
            result["score"] = 0.3
            result["critical_failure"] = True
        elif len(text) < 100:
            result["warnings"].append("Summary may be too brief")
            result["score"] = 0.7
        
        # Check for key components
        has_findings = any(word in text.lower() for word in 
                          ["condition", "finding", "result", "detected", "identified"])
        has_action = any(word in text.lower() for word in 
                        ["recommend", "follow", "contact", "seek", "schedule"])
        
        if not has_findings:
            result["warnings"].append("Summary lacks clear findings")
            result["score"] *= 0.8
        
        if not has_action:
            result["suggestions"].append("Consider adding actionable recommendations")
            result["score"] *= 0.9
        
        return result
    
    def _check_accuracy_indicators(self, text: str, metrics: SummaryMetrics) -> Dict:
        """Check indicators of accuracy and reliability"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        # Check confidence score
        if metrics.confidence_score < 0.5:
            result["issues"].append("Low confidence in analysis results")
            result["score"] = 0.4
            result["critical_failure"] = True
        elif metrics.confidence_score < 0.7:
            result["warnings"].append("Moderate confidence in results")
            result["score"] = 0.8
        
        # Check for vague language
        vague_terms = ["may", "might", "possibly", "perhaps", "unclear"]
        vague_count = sum(1 for term in vague_terms if term in text.lower())
        
        if vague_count > 3:
            result["warnings"].append("Summary contains significant uncertainty")
            result["score"] *= 0.8
        
        return result
    
    def _check_structure_quality(self, text: str) -> Dict:
        """Check structural quality of the summary"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        # Check for proper sentence structure
        sentences = nltk.sent_tokenize(text)
        
        # Check for very short sentences (may indicate fragmentation)
        short_sentences = [s for s in sentences if len(s.split()) < 3]
        if len(short_sentences) > len(sentences) * 0.2:
            result["suggestions"].append("Some sentences may be too short")
            result["score"] *= 0.9
        
        # Check for repetition
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only check longer words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_freq.items() if count > 3]
        if repeated_words:
            result["suggestions"].append("Consider reducing word repetition")
            result["score"] *= 0.95
        
        return result
    
    def _check_audience_appropriateness(self, text: str, audience: str) -> Dict:
        """Check if language is appropriate for target audience"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        technical_terms = [
            "pathophysiology", "etiology", "differential", "prognosis",
            "comorbidity", "contraindication", "therapeutic", "pharmacological"
        ]
        
        if audience == "patient":
            technical_count = sum(1 for term in technical_terms if term in text.lower())
            if technical_count > 2:
                result["warnings"].append("May contain too much medical jargon for patients")
                result["score"] = 0.7
        elif audience == "medical":
            # Medical summaries should have some technical content
            if not any(term in text.lower() for term in technical_terms[:4]):
                result["suggestions"].append("Consider adding more clinical terminology")
                result["score"] = 0.9
        
        return result
    
    def _check_medical_terminology_usage(self, text: str, audience: str) -> Dict:
        """Check appropriate usage of medical terminology"""
        
        result = {"score": 1.0, "issues": [], "warnings": [], "suggestions": []}
        
        # Common medical abbreviations that should be spelled out for patients
        abbreviations = ["MI", "CVA", "COPD", "DM", "HTN", "CHF"]
        
        if audience == "patient":
            found_abbrevs = [abbrev for abbrev in abbreviations if abbrev in text]
            if found_abbrevs:
                result["suggestions"].append(
                    f"Consider spelling out abbreviations for patients: {', '.join(found_abbrevs)}"
                )
                result["score"] *= 0.9
        
        # Check for medication names without explanation
        potential_meds = re.findall(r'\b[A-Z][a-z]+[A-Z][a-z]+\b', text)  # CamelCase words
        if audience == "patient" and potential_meds:
            result["suggestions"].append("Consider explaining medication names for patients")
            result["score"] *= 0.95
        
        return result
    
    def generate_improvement_recommendations(self, validation_results: Dict,
                                           original_text: str,
                                           target_audience: str) -> List[str]:
        """Generate specific recommendations for improving summary quality"""
        
        recommendations = []
        
        if validation_results["overall_quality"] == "FAIL":
            recommendations.append("Summary requires significant revision before use")
        
        # Address specific issues
        for issue in validation_results["issues"]:
            if "too complex" in issue:
                recommendations.append("Simplify language and use shorter sentences")
            elif "too short" in issue:
                recommendations.append("Expand summary to include more comprehensive information")
            elif "Low confidence" in issue:
                recommendations.append("Review analysis parameters and data quality")
        
        # Address warnings
        for warning in validation_results["warnings"]:
            if "complexity borderline high" in warning:
                recommendations.append("Consider simplifying technical language")
            elif "difficult to read" in warning:
                recommendations.append("Break up complex sentences and paragraphs")
            elif "too brief" in warning:
                recommendations.append("Add more detail about findings and recommendations")
        
        # Add audience-specific recommendations
        if target_audience == "patient":
            recommendations.append("Ensure all medical terms are explained in plain language")
            recommendations.append("Include clear next steps and contact information")
        elif target_audience == "medical":
            recommendations.append("Include relevant clinical details and reference ranges")
            recommendations.append("Specify follow-up protocols and monitoring requirements")
        elif target_audience == "hr":
            recommendations.append("Focus on work-related implications and fitness for duty")
            recommendations.append("Include specific workplace accommodations if needed")
        
        return list(set(recommendations))  # Remove duplicates


# Usage Example and Testing Functions
def create_sample_summary_data() -> SummaryData:
    """Create sample data for testing the enhanced summary generator"""
    
    sample_conditions = [
        Condition(
            name="Hypertension",
            severity=Severity.MODERATE,
            description="Elevated blood pressure requiring management",
            follow_up="Monitor blood pressure regularly",
            recommendations=["Lifestyle modifications", "Consider medication"],
            confidence_score=0.85,
            affected_systems=["cardiovascular"]
        ),
        Condition(
            name="Type 2 Diabetes",
            severity=Severity.SEVERE,
            description="Elevated glucose levels indicating diabetes",
            follow_up="Endocrinology consultation needed",
            recommendations=["Dietary counseling", "Medication management", "Regular monitoring"],
            confidence_score=0.92,
            affected_systems=["endocrine", "cardiovascular"]
        )
    ]
    
    sample_risks = [
        RiskAssessment(
            category="Cardiovascular Disease",
            risk_level=RiskLevel.HIGH,
            score=7.8,
            factors=["Hypertension", "Diabetes", "Family history"],
            recommendations=["Cardiac screening", "Lifestyle intervention"],
            confidence_score=0.88,
            time_horizon="5-year"
        )
    ]
    
    sample_abnormal_tests = [
        {
            "test_name": "HbA1c",
            "value": 8.5,
            "unit": "%",
            "status": "High",
            "severity": "Severe",
            "interpretation": "Poor glycemic control"
        }
    ]
    
    sample_emergency_flags = [
        {
            "condition": "Severe Hyperglycemia",
            "value": "Glucose > 400 mg/dL",
            "action": "Immediate medical evaluation required"
        }
    ]
    
    return SummaryData(
        conditions=sample_conditions,
        risks=sample_risks,
        abnormal_tests=sample_abnormal_tests,
        emergency_flags=sample_emergency_flags,
        patient_demographics={"age": 55, "gender": "M"},
        analysis_timestamp=datetime.now()
    )


def demonstrate_enhanced_functionality():
    """Demonstrate the enhanced AI/ML functionality"""
    
    # Initialize the enhanced generator
    generator = IntelligentSummaryGenerator(enable_ml=True)
    
    # Create sample data
    sample_data = create_sample_summary_data()
    
    # Generate intelligent summaries for different audiences
    print("=== ENHANCED SUMMARY GENERATION DEMO ===\n")
    
    audiences = ["medical", "patient", "hr"]
    
    for audience in audiences:
        print(f"--- {audience.upper()} SUMMARY ---")
        result = generator.generate_intelligent_summary(sample_data, audience)
        
        print(f"Summary: {result['summary']}")
        print(f"Priority Score: {result.get('priority_score', 0):.1f}/100")
        print(f"Complexity: {result.get('complexity_level', 'Unknown')}")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"Readability: {metrics.readability_score:.1f}")
            print(f"Confidence: {metrics.confidence_score:.2f}")
            print(f"Key Topics: {', '.join(metrics.key_topics[:3])}")
        
        print()
    
    # Demonstrate quality assurance
    print("=== QUALITY ASSURANCE DEMO ===\n")
    qa_system = SummaryQualityAssurance()
    
    medical_result = generator.generate_intelligent_summary(sample_data, "medical")
    validation = qa_system.validate_summary_quality(
        medical_result['summary'], 
        "medical", 
        medical_result.get('metrics')
    )
    
    print(f"Quality Assessment: {validation['overall_quality']}")
    print(f"Quality Score: {validation['quality_score']:.2f}")
    print(f"Issues: {len(validation['issues'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    
    # Demonstrate analytics
    print("=== ANALYTICS DEMO ===\n")
    analytics = SummaryAnalytics()
    
    all_summaries = {}
    for audience in audiences:
        result = generator.generate_intelligent_summary(sample_data, audience)
        all_summaries[audience] = result['summary']
    
    effectiveness = analytics.analyze_summary_effectiveness(sample_data, all_summaries)
    print(f"Average Coverage: {effectiveness['coverage_analysis']['average_coverage']:.2f}")
    print(f"Consistency Score: {effectiveness['consistency_score']:.2f}")
    
    # Demonstrate report export
    print("=== REPORT EXPORT DEMO ===\n")
    markdown_report = generator.export_summary_report(sample_data, "markdown")
    print("Generated comprehensive markdown report")
    print(f"Report length: {len(markdown_report)} characters")


if __name__ == "__main__":
    demonstrate_enhanced_functionality()
    
    def _export_html_report(self, summary_data: SummaryData, 
                           medical_summary: Dict, patient_summary: Dict) -> str:
        """Export report in HTML format"""
        # Simplified HTML export
        html = f"""
        <html>
        <head><title>Medical Analysis Report</title></head>
        <body>
        <h1>Medical Analysis Report</h1>
        <p><strong>Generated:</strong> {summary_data.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h2>Summary</h2>
        <p>{medical_summary['summary']}</p>
        </body>
        </html>
        """
        return html
    
    def _export_text_report(self, summary_data: SummaryData, 
                           medical_summary: Dict, patient_summary: Dict) -> str:
        """Export report in plain text format"""
        report = f"""MEDICAL ANALYSIS REPORT
{'-' * 50}
Generated: {summary_data.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
Priority Score: {medical_summary.get('priority_score', 0):.1f}/100
Complexity: {medical_summary.get('complexity_level', 'Unknown')}

MEDICAL SUMMARY
{medical_summary['summary']}

PATIENT SUMMARY  
{patient_summary['summary']}
"""
        return report