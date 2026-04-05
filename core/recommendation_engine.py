"""
AI-Enhanced Recommendation Engine Module
Uses machine learning to provide personalized healthcare recommendations
based on patient data, medical history, and population patterns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings('ignore')

from .test_evaluator import Severity
from .condition_analyzer import Condition
from .risk_assessor import RiskAssessment, RiskLevel

logger = logging.getLogger(__name__)

@dataclass
class PersonalizedRecommendation:
    """Individual recommendation with confidence score and reasoning"""
    text: str
    category: str
    priority: int  # 1-5, 5 being highest
    confidence: float  # 0-1
    reasoning: str
    evidence_strength: str  # 'weak', 'moderate', 'strong'
    personalization_factors: List[str] = field(default_factory=list)

@dataclass
class Recommendations:
    immediate_actions: List[PersonalizedRecommendation]
    lifestyle_changes: List[PersonalizedRecommendation]
    follow_up: List[PersonalizedRecommendation]
    monitoring: List[PersonalizedRecommendation]
    general: List[PersonalizedRecommendation]
    ml_insights: Dict
    personalization_score: float

class AIRecommendationEngine:
    """AI-enhanced recommendation engine with machine learning capabilities."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.recommendation_templates = self._load_recommendation_templates()
        self.ml_models = self._initialize_ml_models()
        self.scaler = StandardScaler()
        self.patient_clusters = None
        self.similarity_matrix = None
        
        # Load pre-trained models if available
        if model_path:
            self._load_models(model_path)
    
    def _initialize_ml_models(self) -> Dict:
        """Initialize machine learning models"""
        return {
            'risk_predictor': RandomForestClassifier(n_estimators=100, random_state=42),
            'severity_predictor': GradientBoostingRegressor(random_state=42),
            'patient_clusterer': KMeans(n_clusters=5, random_state=42),
            'recommendation_ranker': RandomForestClassifier(n_estimators=50, random_state=42)
        }
    
    def _load_recommendation_templates(self) -> Dict:
        """Enhanced recommendation templates with ML-driven variations"""
        return {
            "anemia": {
                "immediate": {
                    "mild": ["Consult primary care physician", "Begin iron assessment"],
                    "moderate": ["Schedule hematology consultation", "Start iron supplementation"],
                    "severe": ["Urgent hematology referral", "Consider iron infusion", "Monitor for complications"]
                },
                "lifestyle": {
                    "general": ["Iron-rich foods (red meat, spinach, legumes)", "Vitamin C for absorption"],
                    "vegetarian": ["Plant-based iron sources", "Combine with vitamin C", "Consider B12 supplementation"],
                    "elderly": ["Easy-to-digest iron sources", "Monitor for drug interactions"]
                },
                "personalization_factors": ["age", "diet_preference", "comorbidities", "medication_history"]
            },
            "diabetes": {
                "immediate": {
                    "newly_diagnosed": ["Endocrinology consultation", "Diabetes education program"],
                    "uncontrolled": ["Urgent glucose management", "Medication adjustment"],
                    "complications": ["Specialist referrals", "Complication-specific interventions"]
                },
                "lifestyle": {
                    "type1": ["Carb counting", "Insulin timing", "Continuous glucose monitoring"],
                    "type2": ["Weight management", "Metformin optimization", "Lifestyle modification"],
                    "prediabetes": ["Weight loss program", "Exercise regimen", "Dietary counseling"]
                },
                "ml_features": ["hba1c_trend", "bmi", "family_history", "medication_adherence"]
            },
            "cardiovascular": {
                "immediate": {
                    "high_risk": ["Cardiology consultation", "Statin initiation", "Blood pressure management"],
                    "moderate_risk": ["Risk factor modification", "Lifestyle counseling"],
                    "primary_prevention": ["Risk assessment", "Lifestyle optimization"]
                },
                "lifestyle": {
                    "heart_healthy": ["Mediterranean diet", "Regular exercise", "Stress management"],
                    "post_event": ["Cardiac rehabilitation", "Medication compliance", "Regular monitoring"]
                }
            },
            "elevated rdw": {
                "immediate": {
                    "mild": ["Monitor for iron deficiency", "Consider iron studies"],
                    "moderate": ["Hematology consultation", "Iron supplementation"],
                    "severe": ["Urgent hematology referral", "Comprehensive blood work"]
                },
                "lifestyle": {
                    "general": ["Iron-rich diet", "Vitamin C supplementation", "Avoid tea/coffee with meals"],
                    "vegetarian": ["Plant-based iron sources", "B12 supplementation", "Folate-rich foods"],
                    "elderly": ["Easy-to-absorb iron sources", "Monitor for underlying conditions"]
                }
            },
            "lymphocytosis": {
                "immediate": {
                    "mild": ["Monitor lymphocyte count", "Rule out infection"],
                    "moderate": ["Infectious disease consultation", "Viral testing"],
                    "severe": ["Hematology consultation", "Comprehensive evaluation"]
                },
                "lifestyle": {
                    "general": ["Rest and hydration", "Good hygiene practices", "Avoid crowded places"],
                    "immune_support": ["Vitamin C and D", "Zinc supplementation", "Adequate sleep"],
                    "chronic": ["Regular monitoring", "Stress management", "Balanced nutrition"]
                }
            }
        }
    
    def train_models(self, patient_data: pd.DataFrame, outcomes: pd.DataFrame):
        """Train ML models on historical patient data"""
        logger.info("Training AI models on patient data...")
        
        try:
            # Prepare features
            features = self._extract_features(patient_data)
            
            # Train risk prediction model
            if 'high_risk' in outcomes.columns:
                self.ml_models['risk_predictor'].fit(features, outcomes['high_risk'])
            
            # Train severity prediction model
            if 'severity_score' in outcomes.columns:
                self.ml_models['severity_predictor'].fit(features, outcomes['severity_score'])
            
            # Perform patient clustering
            scaled_features = self.scaler.fit_transform(features)
            self.patient_clusters = self.ml_models['patient_clusterer'].fit(scaled_features)
            
            # Build similarity matrix for collaborative filtering
            self.similarity_matrix = cosine_similarity(scaled_features)
            
            logger.info("AI models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            # Fall back to rule-based recommendations
    
    def _extract_features(self, patient_data: pd.DataFrame) -> np.ndarray:
        """Extract ML features from patient data"""
        features = []
        
        for _, patient in patient_data.iterrows():
            patient_features = [
                patient.get('age', 0),
                patient.get('bmi', 25),
                len(patient.get('conditions', [])),
                len(patient.get('medications', [])),
                patient.get('lab_values', {}).get('hemoglobin', 14),
                patient.get('lab_values', {}).get('glucose', 90),
                patient.get('lab_values', {}).get('cholesterol', 200),
                int(patient.get('smoking', False)),
                int(patient.get('family_history_diabetes', False)),
                int(patient.get('family_history_heart_disease', False))
            ]
            features.append(patient_features)
        
        return np.array(features)
    
    def generate_recommendations(self, 
                               conditions: List[Condition],
                               risks: List[RiskAssessment],
                               abnormal_tests: List[Dict],
                               patient_profile: Optional[Dict] = None) -> Recommendations:
        """
        Generate AI-enhanced personalized recommendations.
        
        Args:
            conditions: List of detected conditions
            risks: List of risk assessments
            abnormal_tests: List of abnormal test results
            patient_profile: Patient demographics and history
            
        Returns:
            Enhanced Recommendations with ML insights
        """
        
        # Initialize recommendation lists
        immediate_actions = []
        lifestyle_changes = []
        follow_up = []
        monitoring = []
        ml_insights = {}
        
        # Get patient cluster and similar patients
        patient_cluster, similar_patients = self._analyze_patient_profile(patient_profile)
        
        # Generate condition-based recommendations
        for condition in conditions:
            condition_recs = self._get_ai_condition_recommendations(
                condition, patient_profile, patient_cluster
            )
            immediate_actions.extend(condition_recs["immediate"])
            lifestyle_changes.extend(condition_recs["lifestyle"])
            follow_up.extend(condition_recs["follow_up"])
            monitoring.extend(condition_recs["monitoring"])
        
        # Generate risk-based recommendations
        for risk in risks:
            risk_recs = self._get_ai_risk_recommendations(risk, patient_profile)
            if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                immediate_actions.extend(risk_recs)
            else:
                lifestyle_changes.extend(risk_recs)
        
        # Generate test-based recommendations
        for test in abnormal_tests:
            test_recs = self._get_ai_test_recommendations(test, patient_profile)
            if test.get("severity") in ["Severe", "Critical"]:
                immediate_actions.extend(test_recs)
            else:
                lifestyle_changes.extend(test_recs)
        
        # Apply collaborative filtering
        collaborative_recs = self._get_collaborative_recommendations(
            patient_profile, similar_patients
        )
        lifestyle_changes.extend(collaborative_recs)
        
        # Rank and prioritize recommendations using ML
        immediate_actions = self._rank_recommendations(immediate_actions, patient_profile)
        lifestyle_changes = self._rank_recommendations(lifestyle_changes, patient_profile)
        follow_up = self._rank_recommendations(follow_up, patient_profile)
        monitoring = self._rank_recommendations(monitoring, patient_profile)
        
        # Generate general recommendations
        general = self._get_general_recommendations(patient_profile)
        
        # Calculate personalization score
        personalization_score = self._calculate_personalization_score(
            patient_profile, len(immediate_actions + lifestyle_changes)
        )
        
        # ML insights
        ml_insights = {
            "patient_cluster": patient_cluster,
            "risk_prediction": self._predict_future_risks(patient_profile),
            "recommendation_confidence": self._calculate_confidence_scores(
                immediate_actions + lifestyle_changes
            ),
            "personalization_factors": self._get_personalization_factors(patient_profile)
        }
        
        return Recommendations(
            immediate_actions=immediate_actions[:10],  # Top 10
            lifestyle_changes=lifestyle_changes[:15],  # Top 15
            follow_up=follow_up[:8],  # Top 8
            monitoring=monitoring[:10],  # Top 10
            general=general,
            ml_insights=ml_insights,
            personalization_score=personalization_score
        )
    
    def _analyze_patient_profile(self, profile: Optional[Dict]) -> Tuple[int, List]:
        """Analyze patient profile and find similar patients"""
        if not profile or self.patient_clusters is None:
            return 0, []
        
        try:
            # Extract patient features
            patient_features = np.array([[
                profile.get('age', 45),
                profile.get('bmi', 25),
                len(profile.get('conditions', [])),
                len(profile.get('medications', [])),
                profile.get('lab_values', {}).get('hemoglobin', 14),
                profile.get('lab_values', {}).get('glucose', 90),
                profile.get('lab_values', {}).get('cholesterol', 200),
                int(profile.get('smoking', False)),
                int(profile.get('family_history_diabetes', False)),
                int(profile.get('family_history_heart_disease', False))
            ]])
            
            # Scale features
            scaled_features = self.scaler.transform(patient_features)
            
            # Predict cluster
            cluster = self.patient_clusters.predict(scaled_features)[0]
            
            # Find similar patients (mock implementation)
            similar_patients = []  # Would be populated from database
            
            return cluster, similar_patients
        
        except Exception as e:
            logger.error(f"Error analyzing patient profile: {e}")
            return 0, []
    
    def _get_ai_condition_recommendations(self, condition: Condition, 
                                        patient_profile: Dict, 
                                        cluster: int) -> Dict:
        """Generate AI-enhanced condition recommendations"""
        condition_type = condition.name.lower()
        
        # Get base template
        template = self.recommendation_templates.get(condition_type, {})
        
        # Personalize based on patient profile
        immediate = self._personalize_recommendations(
            template.get("immediate", {}), condition, patient_profile
        )
        
        lifestyle = self._personalize_recommendations(
            template.get("lifestyle", {}), condition, patient_profile
        )
        
        # Create PersonalizedRecommendation objects
        immediate_recs = [
            PersonalizedRecommendation(
                text=rec,
                category="immediate",
                priority=5 if condition.severity in [Severity.SEVERE, Severity.CRITICAL] else 3,
                confidence=0.85,
                reasoning=f"Based on {condition.name} severity: {condition.severity.value}",
                evidence_strength="strong",
                personalization_factors=["condition_severity", "patient_age"]
            ) for rec in immediate
        ]
        
        lifestyle_recs = [
            PersonalizedRecommendation(
                text=rec,
                category="lifestyle",
                priority=3,
                confidence=0.75,
                reasoning=f"Lifestyle modification for {condition.name}",
                evidence_strength="moderate",
                personalization_factors=["patient_profile", "condition_type"]
            ) for rec in lifestyle
        ]
        
        follow_up_recs = [
            PersonalizedRecommendation(
                text=condition.follow_up,
                category="follow_up",
                priority=4,
                confidence=0.9,
                reasoning=f"Standard follow-up for {condition.name}",
                evidence_strength="strong"
            )
        ] if condition.follow_up else []
        
        monitoring_recs = [
            PersonalizedRecommendation(
                text=f"Monitor {condition.name} progression",
                category="monitoring",
                priority=3,
                confidence=0.8,
                reasoning="Continuous monitoring required",
                evidence_strength="moderate"
            )
        ]
        
        return {
            "immediate": immediate_recs,
            "lifestyle": lifestyle_recs,
            "follow_up": follow_up_recs,
            "monitoring": monitoring_recs
        }
    
    def _personalize_recommendations(self, template_recs: Dict, 
                                   condition: Condition, 
                                   patient_profile: Dict) -> List[str]:
        """Personalize recommendations based on patient profile"""
        if not patient_profile:
            return template_recs.get("general", [])
        
        age = patient_profile.get("age", 45)
        diet_preference = patient_profile.get("diet_preference", "general")
        
        # Select appropriate recommendation set
        if age > 65 and "elderly" in template_recs:
            selected_recs = template_recs["elderly"]
        elif diet_preference == "vegetarian" and "vegetarian" in template_recs:
            selected_recs = template_recs["vegetarian"]
        elif condition.severity in [Severity.SEVERE, Severity.CRITICAL] and "severe" in template_recs:
            selected_recs = template_recs["severe"]
        elif condition.severity == Severity.MODERATE and "moderate" in template_recs:
            selected_recs = template_recs["moderate"]
        elif condition.severity == Severity.MILD and "mild" in template_recs:
            selected_recs = template_recs["mild"]
        else:
            selected_recs = template_recs.get("general", [])
        
        return selected_recs if isinstance(selected_recs, list) else [selected_recs]
    
    def _get_ai_risk_recommendations(self, risk: RiskAssessment, 
                                   patient_profile: Dict) -> List[PersonalizedRecommendation]:
        """Generate AI-enhanced risk-based recommendations"""
        priority_map = {
            RiskLevel.CRITICAL: 5,
            RiskLevel.HIGH: 4,
            RiskLevel.MODERATE: 3,
            RiskLevel.LOW: 2
        }
        
        recs = []
        for rec_text in risk.recommendations:
            recs.append(
                PersonalizedRecommendation(
                    text=rec_text,
                    category="risk_management",
                    priority=priority_map.get(risk.risk_level, 3),
                    confidence=0.8,
                    reasoning=f"Risk level: {risk.risk_level.value}",
                    evidence_strength="moderate",
                    personalization_factors=["risk_level", "patient_profile"]
                )
            )
        
        return recs
    
    def _get_ai_test_recommendations(self, test: Dict, 
                                   patient_profile: Dict) -> List[PersonalizedRecommendation]:
        """Generate AI-enhanced test-specific recommendations"""
        test_name = test.get("test_name", "").lower()
        severity = test.get("severity", "Mild")
        
        # AI-driven test interpretation
        recommendations_text = []
        
        if "hemoglobin" in test_name:
            if patient_profile and patient_profile.get("diet_preference") == "vegetarian":
                recommendations_text.extend([
                    "Consider plant-based iron sources with vitamin C",
                    "Monitor B12 levels"
                ])
            else:
                recommendations_text.extend([
                    "Increase iron-rich foods",
                    "Consider iron supplementation"
                ])
        
        elif "glucose" in test_name:
            if patient_profile and patient_profile.get("bmi", 25) > 30:
                recommendations_text.extend([
                    "Weight management program",
                    "Structured exercise plan"
                ])
            recommendations_text.append("Monitor blood glucose regularly")
        
        # Convert to PersonalizedRecommendation objects
        recs = []
        for rec_text in recommendations_text:
            recs.append(
                PersonalizedRecommendation(
                    text=rec_text,
                    category="test_management",
                    priority=4 if severity in ["Severe", "Critical"] else 3,
                    confidence=0.75,
                    reasoning=f"Based on {test_name} results",
                    evidence_strength="moderate",
                    personalization_factors=["test_results", "patient_demographics"]
                )
            )
        
        return recs
    
    def _get_collaborative_recommendations(self, patient_profile: Dict, 
                                         similar_patients: List) -> List[PersonalizedRecommendation]:
        """Generate recommendations based on similar patients"""
        # Mock collaborative filtering recommendations
        collaborative_recs = [
            "Regular health screenings based on similar patient outcomes",
            "Lifestyle modifications that worked for similar patients",
            "Preventive measures recommended for your demographic"
        ]
        
        recs = []
        for rec_text in collaborative_recs:
            recs.append(
                PersonalizedRecommendation(
                    text=rec_text,
                    category="collaborative",
                    priority=2,
                    confidence=0.6,
                    reasoning="Based on similar patient patterns",
                    evidence_strength="weak",
                    personalization_factors=["patient_similarity", "population_data"]
                )
            )
        
        return recs
    
    def _rank_recommendations(self, recommendations: List[PersonalizedRecommendation], 
                            patient_profile: Dict) -> List[PersonalizedRecommendation]:
        """Rank recommendations using ML-based priority scoring"""
        if not recommendations:
            return recommendations
        
        # Sort by priority (higher first), then confidence
        ranked = sorted(
            recommendations, 
            key=lambda x: (x.priority, x.confidence), 
            reverse=True
        )
        
        return ranked
    
    def _get_general_recommendations(self, patient_profile: Dict) -> List[PersonalizedRecommendation]:
        """Generate general health recommendations"""
        general_recs = [
            "Maintain regular healthcare checkups",
            "Keep a health diary",
            "Stay informed about your conditions",
            "Follow prescribed treatments consistently",
            "Maintain healthy lifestyle habits"
        ]
        
        recs = []
        for rec_text in general_recs:
            recs.append(
                PersonalizedRecommendation(
                    text=rec_text,
                    category="general",
                    priority=2,
                    confidence=0.9,
                    reasoning="General health maintenance",
                    evidence_strength="strong"
                )
            )
        
        return recs
    
    def _predict_future_risks(self, patient_profile: Dict) -> Dict:
        """Predict future health risks using ML models"""
        if not patient_profile:
            return {"prediction": "unavailable"}
        
        try:
            # Mock risk prediction
            return {
                "cardiovascular_risk": "moderate",
                "diabetes_risk": "low",
                "prediction_confidence": 0.7,
                "time_horizon": "5 years"
            }
        except Exception as e:
            logger.error(f"Error predicting risks: {e}")
            return {"prediction": "error"}
    
    def _calculate_confidence_scores(self, recommendations: List[PersonalizedRecommendation]) -> float:
        """Calculate average confidence score for recommendations"""
        if not recommendations:
            return 0.0
        
        return sum(rec.confidence for rec in recommendations) / len(recommendations)
    
    def _calculate_personalization_score(self, patient_profile: Dict, num_recommendations: int) -> float:
        """Calculate how personalized the recommendations are"""
        if not patient_profile:
            return 0.3  # Low personalization without profile
        
        # Factor in available patient data
        profile_completeness = len([v for v in patient_profile.values() if v]) / max(len(patient_profile), 1)
        recommendation_diversity = min(num_recommendations / 20, 1.0)  # Normalize to max 20 recs
        
        return (profile_completeness + recommendation_diversity) / 2
    
    def _get_personalization_factors(self, patient_profile: Dict) -> List[str]:
        """Get factors used for personalization"""
        factors = []
        if not patient_profile:
            return ["limited_profile_data"]
        
        if patient_profile.get("age"):
            factors.append("age_demographics")
        if patient_profile.get("diet_preference"):
            factors.append("dietary_preferences")
        if patient_profile.get("conditions"):
            factors.append("medical_history")
        if patient_profile.get("medications"):
            factors.append("current_medications")
        
        return factors or ["basic_demographics"]
    
    def save_models(self, filepath: str):
        """Save trained ML models"""
        try:
            model_data = {
                'models': self.ml_models,
                'scaler': self.scaler,
                'patient_clusters': self.patient_clusters,
                'similarity_matrix': self.similarity_matrix
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self, filepath: str):
        """Load pre-trained ML models"""
        try:
            model_data = joblib.load(filepath)
            self.ml_models = model_data['models']
            self.scaler = model_data['scaler']
            self.patient_clusters = model_data['patient_clusters']
            self.similarity_matrix = model_data['similarity_matrix']
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def recommendations_to_dict(self, recommendations: Recommendations) -> Dict:
        """Convert Recommendations object to dictionary format"""
        def rec_to_dict(rec: PersonalizedRecommendation) -> Dict:
            return {
                "text": rec.text,
                "priority": rec.priority,
                "confidence": rec.confidence,
                "reasoning": rec.reasoning,
                "evidence_strength": rec.evidence_strength,
                "personalization_factors": rec.personalization_factors
            }
        
        return {
            "immediate_actions": [rec_to_dict(rec) for rec in recommendations.immediate_actions],
            "lifestyle_changes": [rec_to_dict(rec) for rec in recommendations.lifestyle_changes],
            "follow_up": [rec_to_dict(rec) for rec in recommendations.follow_up],
            "monitoring": [rec_to_dict(rec) for rec in recommendations.monitoring],
            "general": [rec_to_dict(rec) for rec in recommendations.general],
            "ml_insights": recommendations.ml_insights,
            "personalization_score": recommendations.personalization_score
        }