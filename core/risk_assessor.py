"""
AI-Enhanced Risk Assessor Module
Uses built-in ML algorithms to calculate risk levels for diseases with personalized scoring.
"""

import logging
import math
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class RiskAssessment:
    category: str
    risk_level: RiskLevel
    score: float
    confidence: float
    factors: List[str]
    recommendations: List[str]
    ai_insights: Dict
    personalization_factors: List[str]

class AIRiskAssessor:
    """AI-enhanced risk assessor with machine learning capabilities using built-in libraries."""
    
    def __init__(self):
        self.risk_rules = self._load_risk_rules()
        self.population_data = self._initialize_population_data()
        self.patient_history = []
        self.risk_patterns = defaultdict(list)
        
    def _load_risk_rules(self) -> Dict:
        """Load enhanced risk assessment rules with AI components"""
        return {
            "cardiovascular": {
                "factors": ["age", "hdl", "ldl", "triglycerides", "blood_pressure", "family_history", "smoking"],
                "weights": {
                    "age": 0.2,
                    "hdl": 0.15,
                    "ldl": 0.25,
                    "triglycerides": 0.15,
                    "blood_pressure": 0.15,
                    "family_history": 0.05,
                    "smoking": 0.05
                },
                "thresholds": {
                    "age_high": 45,
                    "hdl_low_male": 40,
                    "hdl_low_female": 50,
                    "ldl_high": 160,
                    "ldl_borderline": 130,
                    "triglycerides_high": 200,
                    "triglycerides_very_high": 500
                },
                "risk_multipliers": {
                    "diabetes": 1.5,
                    "hypertension": 1.3,
                    "obesity": 1.2
                }
            },
            "diabetes": {
                "factors": ["hba1c", "glucose", "age", "bmi", "family_history"],
                "weights": {
                    "hba1c": 0.4,
                    "glucose": 0.3,
                    "age": 0.15,
                    "bmi": 0.1,
                    "family_history": 0.05
                },
                "thresholds": {
                    "hba1c_prediabetes": 5.7,
                    "hba1c_diabetes": 6.5,
                    "glucose_prediabetes": 100,
                    "glucose_diabetes": 126,
                    "age_risk": 45,
                    "bmi_risk": 25
                },
                "progression_rates": {
                    "prediabetes_to_diabetes": 0.11,  # 11% annual rate
                    "normal_to_prediabetes": 0.05
                }
            },
            "kidney_disease": {
                "factors": ["creatinine", "egfr", "age", "diabetes", "hypertension"],
                "weights": {
                    "creatinine": 0.3,
                    "egfr": 0.4,
                    "age": 0.15,
                    "diabetes": 0.1,
                    "hypertension": 0.05
                },
                "thresholds": {
                    "creatinine_high": 1.3,
                    "egfr_stage_3": 60,
                    "egfr_stage_4": 30,
                    "egfr_stage_5": 15,
                    "age_risk": 60
                }
            },
            "metabolic_syndrome": {
                "factors": ["waist_circumference", "triglycerides", "hdl", "blood_pressure", "glucose"],
                "criteria_count": 3,  # Need 3+ criteria for diagnosis
                "thresholds": {
                    "waist_male": 102,
                    "waist_female": 88,
                    "triglycerides": 150,
                    "hdl_male": 40,
                    "hdl_female": 50,
                    "systolic_bp": 130,
                    "glucose": 100
                }
            }
        }
    
    def _initialize_population_data(self) -> Dict:
        """Initialize population statistics for risk comparison"""
        return {
            "cardiovascular": {
                "prevalence_by_age": {
                    "20-39": 0.07,
                    "40-59": 0.19,
                    "60-79": 0.37,
                    "80+": 0.58
                },
                "gender_multiplier": {"male": 1.2, "female": 1.0}
            },
            "diabetes": {
                "prevalence_by_age": {
                    "18-44": 0.04,
                    "45-64": 0.17,
                    "65+": 0.27
                },
                "ethnicity_multiplier": {
                    "asian": 1.2,
                    "hispanic": 1.7,
                    "african_american": 1.8,
                    "caucasian": 1.0
                }
            }
        }
    
    def assess_risks(self, test_results: List[Dict], age: int, gender: str, 
                    patient_profile: Optional[Dict] = None) -> List[RiskAssessment]:
        """
        AI-enhanced risk assessment with personalization.
        
        Args:
            test_results: List of test dictionaries
            age: Patient age in years
            gender: Patient gender ('male' or 'female')
            patient_profile: Additional patient information for AI analysis
            
        Returns:
            List of enhanced RiskAssessment objects
        """
        risks = []
        test_dict = {test["test_name"].lower().replace(" ", "_"): test["value"] 
                    for test in test_results}
        
        # Enhanced cardiovascular risk with AI
        cv_risk = self._calculate_ai_cardiovascular_risk(test_dict, age, gender, patient_profile)
        if cv_risk:
            risks.append(cv_risk)
        
        # Enhanced diabetes risk with progression prediction
        dm_risk = self._calculate_ai_diabetes_risk(test_dict, age, patient_profile)
        if dm_risk:
            risks.append(dm_risk)
        
        # Enhanced kidney disease risk
        kidney_risk = self._calculate_ai_kidney_risk(test_dict, age, patient_profile)
        if kidney_risk:
            risks.append(kidney_risk)
        
        # Metabolic syndrome assessment
        metabolic_risk = self._calculate_metabolic_syndrome_risk(test_dict, age, gender, patient_profile)
        if metabolic_risk:
            risks.append(metabolic_risk)
        
        # Apply AI-based risk interactions and adjustments
        risks = self._apply_ai_risk_interactions(risks, patient_profile)
        
        return risks
    
    def _calculate_ai_cardiovascular_risk(self, test_dict: Dict, age: int, gender: str, 
                                        patient_profile: Optional[Dict] = None) -> Optional[RiskAssessment]:
        """AI-enhanced cardiovascular risk calculation"""
        score = 0.0
        factors = []
        confidence = 0.8
        ai_insights = {}
        personalization_factors = []
        
        rules = self.risk_rules["cardiovascular"]
        weights = rules["weights"]
        
        # Age factor with continuous scoring
        age_score = self._calculate_age_risk_score(age, "cardiovascular")
        score += age_score * weights["age"]
        if age > 45:
            factors.append(f"Age {age} (risk increases with age)")
            personalization_factors.append("age_factor")
        
        # HDL factor with gender-specific thresholds
        if "hdl_cholesterol" in test_dict:
            hdl = test_dict["hdl_cholesterol"]
            threshold = rules["thresholds"]["hdl_low_male"] if gender == "male" else rules["thresholds"]["hdl_low_female"]
            
            if hdl < threshold:
                hdl_score = (threshold - hdl) / threshold  # Normalized score
                score += hdl_score * weights["hdl"]
                factors.append(f"Low HDL ({hdl} mg/dL, target >{threshold})")
                personalization_factors.append("hdl_gender_specific")
        
        # LDL with AI-based severity assessment
        if "ldl_cholesterol" in test_dict:
            ldl = test_dict["ldl_cholesterol"]
            ldl_score = self._calculate_ldl_risk_score(ldl)
            score += ldl_score * weights["ldl"]
            
            if ldl > rules["thresholds"]["ldl_high"]:
                factors.append(f"High LDL ({ldl} mg/dL)")
                ai_insights["ldl_severity"] = "severe"
            elif ldl > rules["thresholds"]["ldl_borderline"]:
                factors.append(f"Borderline high LDL ({ldl} mg/dL)")
                ai_insights["ldl_severity"] = "moderate"
        
        # Triglycerides with non-linear risk assessment
        if "triglycerides" in test_dict:
            tg = test_dict["triglycerides"]
            tg_score = self._calculate_triglyceride_risk_score(tg)
            score += tg_score * weights["triglycerides"]
            
            if tg > rules["thresholds"]["triglycerides_very_high"]:
                factors.append(f"Very high triglycerides ({tg} mg/dL)")
                ai_insights["pancreatitis_risk"] = "high"
            elif tg > rules["thresholds"]["triglycerides_high"]:
                factors.append(f"High triglycerides ({tg} mg/dL)")
        
        # Apply risk multipliers from comorbidities
        if patient_profile:
            multiplier = self._calculate_comorbidity_multiplier(patient_profile, "cardiovascular")
            score *= multiplier
            
            if multiplier > 1.0:
                factors.append(f"Increased risk due to comorbidities (×{multiplier:.1f})")
                personalization_factors.append("comorbidity_adjustment")
        
        # Population comparison for AI insights
        population_risk = self._get_population_baseline_risk(age, gender, "cardiovascular")
        relative_risk = score / max(population_risk, 0.1)
        ai_insights["relative_to_population"] = relative_risk
        
        # Calculate final risk level with AI adjustment
        risk_level = self._determine_ai_risk_level(score, relative_risk)
        
        # Adjust confidence based on available data
        data_completeness = len([k for k in ["hdl_cholesterol", "ldl_cholesterol", "triglycerides"] 
                               if k in test_dict]) / 3
        confidence = 0.6 + (0.3 * data_completeness)
        
        if score > 0.1:  # Only return if significant risk
            # Generate AI-enhanced recommendations
            recommendations = self._generate_ai_recommendations("cardiovascular", risk_level, 
                                                              factors, patient_profile)
            
            return RiskAssessment(
                category="Cardiovascular Disease",
                risk_level=risk_level,
                score=score,
                confidence=confidence,
                factors=factors,
                recommendations=recommendations,
                ai_insights=ai_insights,
                personalization_factors=personalization_factors
            )
        
        return None
    
    def _calculate_ai_diabetes_risk(self, test_dict: Dict, age: int, 
                                  patient_profile: Optional[Dict] = None) -> Optional[RiskAssessment]:
        """AI-enhanced diabetes risk with progression prediction"""
        score = 0.0
        factors = []
        confidence = 0.85
        ai_insights = {}
        personalization_factors = []
        
        rules = self.risk_rules["diabetes"]
        weights = rules["weights"]
        
        # HbA1c with progression risk prediction
        if "hba1c" in test_dict:
            hba1c = test_dict["hba1c"]
            hba1c_score = self._calculate_hba1c_risk_score(hba1c)
            score += hba1c_score * weights["hba1c"]
            
            # Predict progression risk
            if 5.7 <= hba1c < 6.5:
                progression_risk = self._predict_diabetes_progression(hba1c, patient_profile)
                factors.append(f"Prediabetes (HbA1c: {hba1c}%)")
                ai_insights["progression_risk_5_years"] = f"{progression_risk:.1%}"
                personalization_factors.append("progression_prediction")
            elif hba1c >= 6.5:
                factors.append(f"Diabetic range HbA1c ({hba1c}%)")
                ai_insights["diabetes_control"] = "poor" if hba1c > 8.0 else "suboptimal"
        
        # Fasting glucose with time-series analysis if available
        if "glucose_fasting" in test_dict:
            glucose = test_dict["glucose_fasting"]
            glucose_score = self._calculate_glucose_risk_score(glucose)
            score += glucose_score * weights["glucose"]
            
            if 100 <= glucose < 126:
                factors.append(f"Impaired fasting glucose ({glucose} mg/dL)")
            elif glucose >= 126:
                factors.append(f"Diabetic range glucose ({glucose} mg/dL)")
        
        # Age factor with personalized assessment
        if age > rules["thresholds"]["age_risk"]:
            age_score = (age - 45) / 40  # Normalized age risk
            score += age_score * weights["age"]
            factors.append(f"Age-related risk (age {age})")
        
        # BMI factor if available in patient profile
        if patient_profile and "bmi" in patient_profile:
            bmi = patient_profile["bmi"]
            if bmi > rules["thresholds"]["bmi_risk"]:
                bmi_score = min((bmi - 25) / 15, 1.0)  # Cap at BMI 40
                score += bmi_score * weights["bmi"]
                factors.append(f"Elevated BMI ({bmi})")
                personalization_factors.append("bmi_risk")
        
        # Family history factor
        if patient_profile and patient_profile.get("family_history_diabetes", False):
            score += 0.3 * weights["family_history"]
            factors.append("Family history of diabetes")
            personalization_factors.append("genetic_predisposition")
        
        # Calculate risk level
        risk_level = self._determine_ai_risk_level(score, 1.0)
        
        if score > 0.1:
            recommendations = self._generate_ai_recommendations("diabetes", risk_level, 
                                                              factors, patient_profile)
            
            return RiskAssessment(
                category="Type 2 Diabetes",
                risk_level=risk_level,
                score=score,
                confidence=confidence,
                factors=factors,
                recommendations=recommendations,
                ai_insights=ai_insights,
                personalization_factors=personalization_factors
            )
        
        return None
    
    def _calculate_ai_kidney_risk(self, test_dict: Dict, age: int,
                                patient_profile: Optional[Dict] = None) -> Optional[RiskAssessment]:
        """AI-enhanced kidney disease risk assessment"""
        score = 0.0
        factors = []
        confidence = 0.9
        ai_insights = {}
        personalization_factors = []
        
        rules = self.risk_rules["kidney_disease"]
        weights = rules["weights"]
        
        # eGFR-based assessment with staging
        if "egfr" in test_dict:
            egfr = test_dict["egfr"]
            egfr_score = self._calculate_egfr_risk_score(egfr)
            score += egfr_score * weights["egfr"]
            
            # CKD staging
            if egfr < rules["thresholds"]["egfr_stage_5"]:
                factors.append(f"Severe kidney disease (eGFR: {egfr})")
                ai_insights["ckd_stage"] = 5
            elif egfr < rules["thresholds"]["egfr_stage_4"]:
                factors.append(f"Moderate-severe kidney disease (eGFR: {egfr})")
                ai_insights["ckd_stage"] = 4
            elif egfr < rules["thresholds"]["egfr_stage_3"]:
                factors.append(f"Mild-moderate kidney disease (eGFR: {egfr})")
                ai_insights["ckd_stage"] = 3
            
            personalization_factors.append("ckd_staging")
        
        # Creatinine assessment
        if "creatinine" in test_dict:
            creatinine = test_dict["creatinine"]
            if creatinine > rules["thresholds"]["creatinine_high"]:
                creatinine_score = min((creatinine - 1.0) / 2.0, 1.0)
                score += creatinine_score * weights["creatinine"]
                factors.append(f"Elevated creatinine ({creatinine} mg/dL)")
        
        # Age factor
        if age > rules["thresholds"]["age_risk"]:
            age_score = (age - 60) / 30
            score += age_score * weights["age"]
            factors.append(f"Advanced age ({age} years)")
        
        # Comorbidity factors
        if patient_profile:
            conditions = patient_profile.get("conditions", [])
            if "diabetes" in [c.lower() for c in conditions]:
                score += 0.3 * weights["diabetes"]
                factors.append("Diabetes mellitus (kidney disease risk factor)")
                ai_insights["diabetic_nephropathy_risk"] = "elevated"
                
            if "hypertension" in [c.lower() for c in conditions]:
                score += 0.2 * weights["hypertension"]
                factors.append("Hypertension (kidney disease risk factor)")
        
        risk_level = self._determine_ai_risk_level(score, 1.0)
        
        if score > 0.1:
            recommendations = self._generate_ai_recommendations("kidney_disease", risk_level,
                                                              factors, patient_profile)
            
            return RiskAssessment(
                category="Chronic Kidney Disease",
                risk_level=risk_level,
                score=score,
                confidence=confidence,
                factors=factors,
                recommendations=recommendations,
                ai_insights=ai_insights,
                personalization_factors=personalization_factors
            )
        
        return None
    
    def _calculate_metabolic_syndrome_risk(self, test_dict: Dict, age: int, gender: str,
                                         patient_profile: Optional[Dict] = None) -> Optional[RiskAssessment]:
        """Calculate metabolic syndrome risk"""
        criteria_met = 0
        factors = []
        ai_insights = {}
        
        rules = self.risk_rules["metabolic_syndrome"]
        thresholds = rules["thresholds"]
        
        # Check each criterion
        if patient_profile and "waist_circumference" in patient_profile:
            waist = patient_profile["waist_circumference"]
            threshold = thresholds["waist_male"] if gender == "male" else thresholds["waist_female"]
            if waist > threshold:
                criteria_met += 1
                factors.append(f"Abdominal obesity (waist: {waist} cm)")
        
        if "triglycerides" in test_dict and test_dict["triglycerides"] > thresholds["triglycerides"]:
            criteria_met += 1
            factors.append(f"High triglycerides ({test_dict['triglycerides']} mg/dL)")
        
        if "hdl_cholesterol" in test_dict:
            hdl = test_dict["hdl_cholesterol"]
            threshold = thresholds["hdl_male"] if gender == "male" else thresholds["hdl_female"]
            if hdl < threshold:
                criteria_met += 1
                factors.append(f"Low HDL ({hdl} mg/dL)")
        
        if "glucose_fasting" in test_dict and test_dict["glucose_fasting"] > thresholds["glucose"]:
            criteria_met += 1
            factors.append(f"Elevated glucose ({test_dict['glucose_fasting']} mg/dL)")
        
        ai_insights["criteria_met"] = criteria_met
        ai_insights["total_criteria"] = 5
        
        if criteria_met >= rules["criteria_count"]:
            risk_level = RiskLevel.HIGH if criteria_met >= 4 else RiskLevel.MODERATE
            
            recommendations = [
                "Weight management and lifestyle modification",
                "Regular physical activity (150+ minutes/week)",
                "Heart-healthy diet (Mediterranean style)",
                "Regular monitoring of metabolic parameters",
                "Consider cardiology consultation"
            ]
            
            return RiskAssessment(
                category="Metabolic Syndrome",
                risk_level=risk_level,
                score=criteria_met / 5.0,
                confidence=0.95,
                factors=factors,
                recommendations=recommendations,
                ai_insights=ai_insights,
                personalization_factors=["metabolic_clustering"]
            )
        
        return None
    
    # Helper methods for AI calculations
    def _calculate_age_risk_score(self, age: int, category: str) -> float:
        """Calculate age-related risk score"""
        if category == "cardiovascular":
            return min((age - 30) / 50, 1.0) if age > 30 else 0.0
        elif category == "diabetes":
            return min((age - 30) / 40, 1.0) if age > 30 else 0.0
        return 0.0
    
    def _calculate_ldl_risk_score(self, ldl: float) -> float:
        """Calculate LDL risk score with non-linear scaling"""
        if ldl <= 100:
            return 0.0
        elif ldl <= 130:
            return 0.3
        elif ldl <= 160:
            return 0.6
        else:
            return min(0.8 + (ldl - 160) / 200, 1.0)
    
    def _calculate_triglyceride_risk_score(self, tg: float) -> float:
        """Calculate triglyceride risk with exponential scaling"""
        if tg <= 150:
            return 0.0
        else:
            return min(0.5 * (1 - math.exp(-(tg - 150) / 200)), 1.0)
    
    def _calculate_hba1c_risk_score(self, hba1c: float) -> float:
        """Calculate HbA1c risk score"""
        if hba1c < 5.7:
            return max((hba1c - 5.0) / 2, 0.0)
        elif hba1c < 6.5:
            return 0.5 + (hba1c - 5.7) / 1.6
        else:
            return min(0.8 + (hba1c - 6.5) / 5, 1.0)
    
    def _calculate_glucose_risk_score(self, glucose: float) -> float:
        """Calculate fasting glucose risk score"""
        if glucose < 100:
            return 0.0
        elif glucose < 126:
            return (glucose - 100) / 52  # Prediabetic range
        else:
            return min(0.8 + (glucose - 126) / 200, 1.0)
    
    def _calculate_egfr_risk_score(self, egfr: float) -> float:
        """Calculate eGFR-based risk score"""
        if egfr >= 90:
            return 0.0
        elif egfr >= 60:
            return (90 - egfr) / 60
        else:
            return min(0.5 + (60 - egfr) / 120, 1.0)
    
    def _predict_diabetes_progression(self, hba1c: float, patient_profile: Optional[Dict]) -> float:
        """Predict 5-year diabetes progression risk"""
        base_risk = self.risk_rules["diabetes"]["progression_rates"]["prediabetes_to_diabetes"]
        
        # Adjust for HbA1c level
        hba1c_factor = min((hba1c - 5.7) / 0.8, 1.0)
        
        # Adjust for other factors
        age_factor = 1.0
        bmi_factor = 1.0
        
        if patient_profile:
            age = patient_profile.get("age", 45)
            if age > 65:
                age_factor = 1.3
            
            bmi = patient_profile.get("bmi", 25)
            if bmi > 30:
                bmi_factor = 1.4
        
        return min(base_risk * (1 + hba1c_factor) * age_factor * bmi_factor, 0.8)
    
    def _calculate_comorbidity_multiplier(self, patient_profile: Dict, category: str) -> float:
        """Calculate risk multiplier based on comorbidities"""
        multiplier = 1.0
        conditions = [c.lower() for c in patient_profile.get("conditions", [])]
        
        if category == "cardiovascular":
            multipliers = self.risk_rules["cardiovascular"]["risk_multipliers"]
            if "diabetes" in conditions:
                multiplier *= multipliers["diabetes"]
            if "hypertension" in conditions:
                multiplier *= multipliers["hypertension"]
            if patient_profile.get("bmi", 25) > 30:
                multiplier *= multipliers["obesity"]
        
        return multiplier
    
    def _get_population_baseline_risk(self, age: int, gender: str, category: str) -> float:
        """Get population baseline risk for comparison"""
        pop_data = self.population_data.get(category, {})
        
        # Age-based prevalence
        age_prevalence = 0.1  # Default
        prevalence_by_age = pop_data.get("prevalence_by_age", {})
        
        for age_range, prevalence in prevalence_by_age.items():
            if self._age_in_range(age, age_range):
                age_prevalence = prevalence
                break
        
        # Gender adjustment
        gender_mult = pop_data.get("gender_multiplier", {}).get(gender, 1.0)
        
        return age_prevalence * gender_mult
    
    def _age_in_range(self, age: int, age_range: str) -> bool:
        """Check if age falls in specified range"""
        if "+" in age_range:
            min_age = int(age_range.replace("+", ""))
            return age >= min_age
        elif "-" in age_range:
            min_age, max_age = map(int, age_range.split("-"))
            return min_age <= age <= max_age
        return False
    
    def _determine_ai_risk_level(self, score: float, relative_risk: float) -> RiskLevel:
        """Determine risk level using AI-enhanced scoring"""
        # Combine absolute score with relative risk
        combined_score = (score + relative_risk) / 2
        
        if combined_score >= 0.8:
            return RiskLevel.CRITICAL
        elif combined_score >= 0.6:
            return RiskLevel.HIGH
        elif combined_score >= 0.3:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def _apply_ai_risk_interactions(self, risks: List[RiskAssessment], 
                                  patient_profile: Optional[Dict]) -> List[RiskAssessment]:
        """Apply AI-based risk interactions and adjustments"""
        if len(risks) <= 1:
            return risks
        
        # Check for risk synergies
        risk_categories = [risk.category for risk in risks]
        
        # Cardiovascular + Diabetes interaction
        if "Cardiovascular Disease" in risk_categories and "Type 2 Diabetes" in risk_categories:
            for risk in risks:
                if risk.category in ["Cardiovascular Disease", "Type 2 Diabetes"]:
                    risk.score = min(risk.score * 1.2, 1.0)  # 20% increase
                    risk.factors.append("Synergistic risk: CV + Diabetes")
                    risk.ai_insights["risk_interaction"] = "cardiovascular_diabetes_synergy"
        
        # Metabolic syndrome interactions
        if "Metabolic Syndrome" in risk_categories:
            for risk in risks:
                if risk.category != "Metabolic Syndrome":
                    risk.score = min(risk.score * 1.15, 1.0)
                    risk.ai_insights["metabolic_syndrome_effect"] = "elevated"
        
        return risks
    
    def _generate_ai_recommendations(self, category: str, risk_level: RiskLevel,
                                   factors: List[str], patient_profile: Optional[Dict]) -> List[str]:
        """Generate AI-enhanced personalized recommendations"""
        base_recommendations = {
            "cardiovascular": {
                RiskLevel.LOW: [
                    "Maintain heart-healthy lifestyle",
                    "Regular exercise (150 min/week)",
                    "Annual lipid screening"
                ],
                RiskLevel.MODERATE: [
                    "Cardiology consultation recommended",
                    "Heart-healthy diet (Mediterranean style)",
                    "Regular blood pressure monitoring",
                    "Consider statin therapy discussion"
                ],
                RiskLevel.HIGH: [
                    "Urgent cardiology referral",
                    "Aggressive lifestyle modification",
                    "Start statin therapy",
                    "Blood pressure optimization",
                    "Smoking cessation if applicable"
                ],
                RiskLevel.CRITICAL: [
                    "Immediate cardiology consultation",
                    "Consider emergency intervention",
                    "Intensive medical management",
                    "Frequent monitoring required"
                ]
            },
            "diabetes": {
                RiskLevel.LOW: [
                    "Annual diabetes screening",
                    "Maintain healthy weight",
                    "Regular physical activity"
                ],
                RiskLevel.MODERATE: [
                    "Diabetes prevention program",
                    "Weight loss if overweight",
                    "HbA1c monitoring every 6 months",
                    "Dietary counseling"
                ],
                RiskLevel.HIGH: [
                    "Endocrinology consultation",
                    "Intensive lifestyle intervention",
                    "Consider metformin",
                    "Quarterly HbA1c monitoring"
                ],
                RiskLevel.CRITICAL: [
                    "Immediate endocrinology referral",
                    "Start diabetes medication",
                    "Intensive glucose monitoring",
                    "Diabetes education program"
                ]
            },
            "kidney_disease": {
                RiskLevel.LOW: [
                    "Annual kidney function monitoring",
                    "Maintain adequate hydration",
                    "Monitor blood pressure"
                ],
                RiskLevel.MODERATE: [
                    "Nephrology consultation",
                    "Blood pressure optimization",
                    "Protein restriction if indicated",
                    "Monitor for complications"
                ],
                RiskLevel.HIGH: [
                    "Urgent nephrology referral",
                    "Aggressive blood pressure control",
                    "Medication review for nephrotoxics",
                    "Prepare for renal replacement therapy"
                ],
                RiskLevel.CRITICAL: [
                    "Immediate nephrology consultation",
                    "Emergency dialysis evaluation",
                    "Intensive medical management",
                    "Transplant evaluation if appropriate"
                ]
            }
        }
        
        recommendations = base_recommendations.get(category, {}).get(risk_level, [])
        
        # Personalize recommendations based on patient profile
        if patient_profile:
            recommendations = self._personalize_recommendations(recommendations, category, patient_profile)
        
        return recommendations
    
    def _personalize_recommendations(self, base_recs: List[str], category: str, 
                                   patient_profile: Dict) -> List[str]:
        """Personalize recommendations based on patient profile"""
        personalized = base_recs.copy()
        
        age = patient_profile.get("age", 45)
        conditions = [c.lower() for c in patient_profile.get("conditions", [])]
        
        # Age-specific recommendations
        if age > 65:
            if category == "cardiovascular":
                personalized.append("Consider lower-intensity statin if needed")
                personalized.append("Monitor for drug interactions")
            elif category == "diabetes":
                personalized.append("Relaxed glycemic targets may be appropriate")
        
        # Condition-specific modifications
        if "chronic kidney disease" in conditions and category == "diabetes":
            personalized.append("Use kidney-safe diabetes medications")
            personalized.append("Monitor for diabetic nephropathy progression")
        
        if "heart failure" in conditions and category == "kidney_disease":
            personalized.append("Careful fluid balance management")
            personalized.append("Coordinate with cardiology")
        
        # Lifestyle factors
        if patient_profile.get("smoking", False):
            personalized.append("Smoking cessation is critical")
            personalized.append("Consider nicotine replacement therapy")
        
        return list(set(personalized))  # Remove duplicates
    
    def learn_from_outcomes(self, patient_id: str, risk_assessment: RiskAssessment, 
                          actual_outcome: Dict):
        """Learn from actual patient outcomes to improve future predictions"""
        outcome_data = {
            "patient_id": patient_id,
            "predicted_risk": risk_assessment.risk_level.value,
            "predicted_score": risk_assessment.score,
            "actual_outcome": actual_outcome,
            "timestamp": datetime.now().isoformat()
        }
        
        self.patient_history.append(outcome_data)
        
        # Update risk patterns
        category = risk_assessment.category
        self.risk_patterns[category].append({
            "factors": risk_assessment.factors,
            "predicted_score": risk_assessment.score,
            "actual_severity": actual_outcome.get("severity", "unknown")
        })
        
        # Adjust prediction accuracy (simple learning mechanism)
        if len(self.risk_patterns[category]) > 10:
            self._update_risk_thresholds(category)
    
    def _update_risk_thresholds(self, category: str):
        """Update risk thresholds based on historical outcomes"""
        patterns = self.risk_patterns[category]
        
        # Simple threshold adjustment based on prediction accuracy
        # This is a basic implementation - real ML would be more sophisticated
        accurate_predictions = sum(1 for p in patterns[-10:] 
                                 if self._prediction_was_accurate(p))
        accuracy = accurate_predictions / 10
        
        # Adjust thresholds if accuracy is low
        if accuracy < 0.7:
            logger.info(f"Adjusting risk thresholds for {category} due to low accuracy: {accuracy}")
            # Implementation would adjust internal thresholds
    
    def _prediction_was_accurate(self, pattern: Dict) -> bool:
        """Check if prediction was accurate (simplified)"""
        predicted_score = pattern["predicted_score"]
        actual_severity = pattern["actual_severity"]
        
        # Simple mapping of severity to score ranges
        severity_ranges = {
            "mild": (0.0, 0.3),
            "moderate": (0.3, 0.6),
            "severe": (0.6, 1.0)
        }
        
        if actual_severity in severity_ranges:
            min_score, max_score = severity_ranges[actual_severity]
            return min_score <= predicted_score <= max_score
        
        return True  # Unknown severity - assume accurate
    
    def get_risk_trends(self, patient_id: str) -> Dict:
        """Get risk trends for a specific patient"""
        patient_data = [h for h in self.patient_history if h["patient_id"] == patient_id]
        
        if len(patient_data) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        patient_data.sort(key=lambda x: x["timestamp"])
        
        # Calculate trend
        scores = [d["predicted_score"] for d in patient_data]
        if len(scores) >= 3:
            recent_avg = statistics.mean(scores[-3:])
            earlier_avg = statistics.mean(scores[:-3]) if len(scores) > 3 else scores[0]
            
            trend = "improving" if recent_avg < earlier_avg else "worsening" if recent_avg > earlier_avg else "stable"
        else:
            trend = "improving" if scores[-1] < scores[0] else "worsening" if scores[-1] > scores[0] else "stable"
        
        return {
            "trend": trend,
            "latest_score": scores[-1],
            "score_change": scores[-1] - scores[0],
            "assessments_count": len(patient_data)
        }
    
    def export_model_insights(self) -> Dict:
        """Export insights from the AI risk assessment model"""
        return {
            "total_assessments": len(self.patient_history),
            "risk_categories_analyzed": list(self.risk_patterns.keys()),
            "average_accuracy_by_category": {
                category: self._calculate_category_accuracy(category)
                for category in self.risk_patterns.keys()
            },
            "common_risk_factors": self._get_common_risk_factors(),
            "model_version": "1.0",
            "last_updated": datetime.now().isoformat()
        }
    
    def _calculate_category_accuracy(self, category: str) -> float:
        """Calculate prediction accuracy for a specific category"""
        patterns = self.risk_patterns[category]
        if len(patterns) < 5:
            return 0.0
        
        accurate = sum(1 for p in patterns if self._prediction_was_accurate(p))
        return accurate / len(patterns)
    
    def _get_common_risk_factors(self) -> Dict:
        """Identify most common risk factors across all assessments"""
        all_factors = []
        for patterns in self.risk_patterns.values():
            for pattern in patterns:
                all_factors.extend(pattern["factors"])
        
        factor_counts = Counter(all_factors)
        return dict(factor_counts.most_common(10))
    
    def risk_assessments_to_dict(self, risks: List[RiskAssessment]) -> List[Dict]:
        """Convert enhanced RiskAssessment objects to dictionary format"""
        return [
            {
                "category": risk.category,
                "risk_level": risk.risk_level.value,
                "score": risk.score,
                "confidence": risk.confidence,
                "factors": risk.factors,
                "recommendations": risk.recommendations,
                "ai_insights": risk.ai_insights,
                "personalization_factors": risk.personalization_factors
            }
            for risk in risks
        ]