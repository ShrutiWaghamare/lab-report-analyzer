"""
Enhanced Condition Analyzer Module with AI/ML capabilities
Detects health conditions using rule-based logic, pattern recognition, and machine learning.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

from .test_evaluator import Severity

logger = logging.getLogger(__name__)

@dataclass
class Condition:
    name: str
    severity: Severity
    description: str
    symptoms: List[str]
    recommendations: List[str]
    follow_up: str
    risk_factors: List[str]
    confidence_score: float = 0.0  # AI confidence score
    detection_method: str = "rule_based"  # rule_based, ml_model, pattern_matching

class MLConditionPredictor:
    """Machine Learning component for condition prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, test_dict: Dict, age: int, gender: str) -> np.array:
        """Convert test results to ML features"""
        # Standard feature set for ML model
        feature_map = {
            'age': age,
            'gender_male': 1 if gender.lower() == 'male' else 0,
            'hemoglobin': test_dict.get('hemoglobin', 0),
            'rbc_count': test_dict.get('rbc_count', 0),
            'pcv': test_dict.get('pcv', 0),
            'mcv': test_dict.get('mcv', 0),
            'mch': test_dict.get('mch', 0),
            'mchc': test_dict.get('mchc', 0),
            'rdw_cv': test_dict.get('rdw_cv', 0),
            'wbc_count': test_dict.get('wbc_count', 0),
            'neutrophils': test_dict.get('neutrophils', 0),
            'lymphocytes': test_dict.get('lymphocytes', 0),
            'platelets': test_dict.get('platelets', 0),
            'glucose_fasting': test_dict.get('glucose_fasting', 0),
            'hba1c': test_dict.get('hba1c', 0),
            'total_cholesterol': test_dict.get('total_cholesterol', 0),
            'hdl_cholesterol': test_dict.get('hdl_cholesterol', 0),
            'ldl_cholesterol': test_dict.get('ldl_cholesterol', 0),
            'triglycerides': test_dict.get('triglycerides', 0),
            't3': test_dict.get('t3', 0),
            't4': test_dict.get('t4', 0),
            'tsh': test_dict.get('tsh', 0),
            'creatinine': test_dict.get('creatinine', 0),
            'uric_acid': test_dict.get('uric_acid', 0),
            'esr': test_dict.get('esr', 0),
        }
        
        self.feature_names = list(feature_map.keys())
        return np.array(list(feature_map.values())).reshape(1, -1)
    
    def train_models(self, training_data: Optional[pd.DataFrame] = None):
        """Train ML models for condition detection"""
        if training_data is None:
            # Generate synthetic training data for demonstration
            training_data = self._generate_synthetic_training_data()
        
        # Prepare features and labels
        feature_columns = [col for col in training_data.columns if col not in ['condition', 'severity']]
        X = training_data[feature_columns].values
        
        # Train models for different conditions
        conditions = ['anemia', 'diabetes', 'cardiovascular_risk', 'thyroid_dysfunction']
        
        for condition in conditions:
            y = (training_data['condition'] == condition).astype(int)
            
            # Skip if no positive cases
            if y.sum() == 0:
                continue
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble model
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Store model and scaler
            self.models[condition] = model
            self.scalers[condition] = scaler
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model for {condition} trained with accuracy: {accuracy:.3f}")
        
        self.is_trained = True
        
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for model training"""
        np.random.seed(42)
        n_samples = 1000
        
        data = []
        for i in range(n_samples):
            # Generate base patient data
            age = np.random.randint(18, 80)
            gender_male = np.random.randint(0, 2)
            
            # Generate test values with realistic correlations
            sample = {
                'age': age,
                'gender_male': gender_male,
                'hemoglobin': np.random.normal(13.5, 2),
                'rbc_count': np.random.normal(4.5, 0.5),
                'pcv': np.random.normal(42, 5),
                'mcv': np.random.normal(88, 8),
                'mch': np.random.normal(29, 3),
                'mchc': np.random.normal(33, 2),
                'rdw_cv': np.random.normal(13, 2),
                'wbc_count': np.random.normal(7, 2),
                'neutrophils': np.random.normal(60, 10),
                'lymphocytes': np.random.normal(30, 8),
                'platelets': np.random.normal(250, 50),
                'glucose_fasting': np.random.normal(95, 15),
                'hba1c': np.random.normal(5.4, 0.8),
                'total_cholesterol': np.random.normal(200, 40),
                'hdl_cholesterol': np.random.normal(50, 15),
                'ldl_cholesterol': np.random.normal(120, 30),
                'triglycerides': np.random.normal(150, 60),
                't3': np.random.normal(100, 30),
                't4': np.random.normal(8, 2),
                'tsh': np.random.normal(2.5, 1.5),
                'creatinine': np.random.normal(1.0, 0.3),
                'uric_acid': np.random.normal(5.5, 1.5),
                'esr': np.random.normal(10, 10),
            }
            
            # Determine conditions based on realistic thresholds
            condition = 'normal'
            if sample['hemoglobin'] < 12 and sample['mcv'] < 80:
                condition = 'anemia'
            elif sample['hba1c'] > 6.5 or sample['glucose_fasting'] > 126:
                condition = 'diabetes'
            elif sample['ldl_cholesterol'] > 160 or sample['hdl_cholesterol'] < 40:
                condition = 'cardiovascular_risk'
            elif sample['tsh'] > 4.94 or sample['t3'] < 80:
                condition = 'thyroid_dysfunction'
            
            sample['condition'] = condition
            sample['severity'] = np.random.choice(['mild', 'moderate', 'severe'], p=[0.6, 0.3, 0.1])
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def predict_condition(self, features: np.array, condition_type: str) -> Tuple[bool, float]:
        """Predict if a condition is present using ML model"""
        if not self.is_trained or condition_type not in self.models:
            return False, 0.0
        
        try:
            model = self.models[condition_type]
            scaler = self.scalers[condition_type]
            
            features_scaled = scaler.transform(features)
            probability = model.predict_proba(features_scaled)[0][1]  # Probability of positive class
            prediction = probability > 0.5
            
            return prediction, probability
        except Exception as e:
            logger.error(f"ML prediction failed for {condition_type}: {str(e)}")
            return False, 0.0

class PatternMatcher:
    """Advanced pattern matching for condition detection"""
    
    def __init__(self):
        self.pattern_rules = self._load_advanced_patterns()
    
    def _load_advanced_patterns(self) -> Dict:
        """Load advanced pattern matching rules"""
        return {
            "metabolic_syndrome": {
                "criteria": [
                    ("waist_circumference", ">", 102, "male"),  # Would need additional data
                    ("triglycerides", ">=", 150),
                    ("hdl_cholesterol", "<", 40, "male"),
                    ("hdl_cholesterol", "<", 50, "female"),
                    ("blood_pressure", ">=", "130/85"),  # Would need BP data
                    ("glucose_fasting", ">=", 100)
                ],
                "required_count": 3
            },
            "insulin_resistance": {
                "patterns": [
                    {"hba1c": (5.7, 6.4), "triglycerides": (">", 150), "hdl_cholesterol": ("<", 50)},
                    {"glucose_fasting": (100, 125), "triglycerides": (">", 200)}
                ]
            },
            "chronic_kidney_disease": {
                "stages": {
                    1: {"egfr": ">=90", "additional": "kidney_damage"},
                    2: {"egfr": (60, 89), "additional": "kidney_damage"},
                    3: {"egfr": (30, 59)},
                    4: {"egfr": (15, 29)},
                    5: {"egfr": "<15"}
                }
            }
        }
    
    def detect_patterns(self, test_dict: Dict, age: int, gender: str) -> List[Dict]:
        """Detect complex patterns in test results"""
        detected_patterns = []
        
        # Check for metabolic syndrome pattern
        metabolic_pattern = self._check_metabolic_syndrome(test_dict, gender)
        if metabolic_pattern:
            detected_patterns.append(metabolic_pattern)
        
        # Check for insulin resistance pattern
        insulin_pattern = self._check_insulin_resistance(test_dict)
        if insulin_pattern:
            detected_patterns.append(insulin_pattern)
        
        return detected_patterns
    
    def _check_metabolic_syndrome(self, test_dict: Dict, gender: str) -> Optional[Dict]:
        """Check for metabolic syndrome pattern"""
        criteria_met = 0
        met_criteria = []
        
        # Triglycerides >= 150
        if test_dict.get('triglycerides', 0) >= 150:
            criteria_met += 1
            met_criteria.append("High triglycerides")
        
        # HDL cholesterol
        hdl_threshold = 40 if gender.lower() == 'male' else 50
        if test_dict.get('hdl_cholesterol', 100) < hdl_threshold:
            criteria_met += 1
            met_criteria.append("Low HDL cholesterol")
        
        # Fasting glucose >= 100
        if test_dict.get('glucose_fasting', 0) >= 100:
            criteria_met += 1
            met_criteria.append("Elevated fasting glucose")
        
        if criteria_met >= 2:  # Modified threshold since we don't have all criteria
            return {
                "pattern": "metabolic_syndrome_risk",
                "criteria_met": criteria_met,
                "details": met_criteria,
                "confidence": min(criteria_met / 3.0, 1.0)
            }
        
        return None
    
    def _check_insulin_resistance(self, test_dict: Dict) -> Optional[Dict]:
        """Check for insulin resistance pattern"""
        hba1c = test_dict.get('hba1c', 0)
        triglycerides = test_dict.get('triglycerides', 0)
        hdl = test_dict.get('hdl_cholesterol', 100)
        glucose = test_dict.get('glucose_fasting', 0)
        
        # Pattern 1: Prediabetic HbA1c + high TG + low HDL
        if 5.7 <= hba1c <= 6.4 and triglycerides > 150 and hdl < 50:
            return {
                "pattern": "insulin_resistance",
                "confidence": 0.8,
                "indicators": ["Prediabetic HbA1c", "High triglycerides", "Low HDL"]
            }
        
        # Pattern 2: IFG + high triglycerides
        if 100 <= glucose <= 125 and triglycerides > 200:
            return {
                "pattern": "insulin_resistance",
                "confidence": 0.7,
                "indicators": ["Impaired fasting glucose", "Very high triglycerides"]
            }
        
        return None

class EnhancedConditionAnalyzer:
    """Enhanced condition analyzer with AI/ML capabilities"""
    
    def __init__(self):
        self.ml_predictor = MLConditionPredictor()
        self.pattern_matcher = PatternMatcher()
        self.condition_rules = self._load_condition_rules()
        
        # Initialize and train ML models
        try:
            self.ml_predictor.train_models()
            logger.info("ML models trained successfully")
        except Exception as e:
            logger.warning(f"ML model training failed: {str(e)}. Using rule-based detection only.")
    
    def _load_condition_rules(self) -> Dict:
        """Load improved condition detection rules with correct thresholds"""
        return {
            "anemia": {
                "tests": ["hemoglobin", "rbc_count", "pcv"],
                "thresholds": {
                    "male": {"hemoglobin": 13.0, "pcv": 40},
                    "female": {"hemoglobin": 12.0, "pcv": 36}
                },
                "symptoms": ["Fatigue", "Weakness", "Pale skin", "Shortness of breath", "Cold hands/feet"],
                "types": {
                    "iron_deficiency": {"mcv": "<80", "rdw": ">15"},
                    "chronic_disease": {"mcv": "80-100", "esr": ">20"},
                    "b12_deficiency": {"mcv": ">100", "rdw": ">15"}
                }
            },
            "diabetes": {
                "tests": ["hba1c", "glucose_fasting"],
                "thresholds": {
                    "diabetes": {"hba1c": 6.5, "glucose_fasting": 126},
                    "prediabetes": {"hba1c": 5.7, "glucose_fasting": 100}
                },
                "symptoms": ["Excessive thirst", "Frequent urination", "Unexplained weight loss", "Fatigue", "Blurred vision"]
            },
            "thyroid_dysfunction": {
                "tests": ["t3", "t4", "tsh"],
                "normal_ranges": {
                    "t3": (35, 193),
                    "t4": (4.87, 11.2),
                    "tsh": (0.35, 4.94)
                },
                "symptoms": ["Weight changes", "Temperature sensitivity", "Heart rate changes", "Mood changes"]
            }
        }
    
    def detect_conditions(self, test_results: List[Dict], age: int, gender: str) -> List[Condition]:
        """Enhanced condition detection using multiple approaches"""
        conditions = []
        
        # Create test dictionary with proper name mapping (same as test evaluator)
        test_name_mapping = {
            "rdw": "rdw_cv",  # Map RDW to RDW-CV
            "rdw_sd": "rdw_sd",
            "ldl_cholesterol": "ldl_cholesterol",
            "hdl_cholesterol": "hdl_cholesterol",
            "total_cholesterol": "total_cholesterol",
            "triglycerides": "triglycerides",
            "t3": "t3",
            "t4": "t4", 
            "tsh": "tsh",
            "hemoglobin": "hemoglobin",
            "rbc_count": "rbc_count",
            "pcv": "pcv",
            "mcv": "mcv",
            "mch": "mch",
            "mchc": "mchc",
            "wbc_count": "wbc_count",
            "platelets": "platelets",
            "neutrophils": "neutrophils",
            "lymphocytes": "lymphocytes",
            "hba1c": "hba1c",
            "glucose_fasting": "glucose_fasting",
            "creatinine": "creatinine",
            "egfr": "egfr",
            "uric_acid": "uric_acid",
            "sodium": "sodium",
            "potassium": "potassium",
            "chloride": "chloride",
            "esr": "esr",
            "sgpt": "sgpt",  # ALT
            "alt": "sgpt",   # ALT alias
            "sgot": "sgot",  # AST
            "ast": "sgot",   # AST alias
            "alp": "alp",    # Alkaline Phosphatase
            "total_bilirubin": "total_bilirubin",
            "direct_bilirubin": "direct_bilirubin",
            "albumin": "albumin"
        }
        
        test_dict = {}
        for test in test_results:
            test_name = test["test_name"].lower().replace(" ", "_")
            mapped_name = test_name_mapping.get(test_name, test_name)
            test_dict[mapped_name] = test["value"]
        
        # Method 1: Rule-based detection (corrected)
        rule_based_conditions = self._rule_based_detection(test_dict, age, gender)
        conditions.extend(rule_based_conditions)
        
        # Method 2: ML-based detection
        if self.ml_predictor.is_trained:
            ml_conditions = self._ml_based_detection(test_dict, age, gender)
            conditions.extend(ml_conditions)
        
        # Method 3: Pattern matching
        pattern_conditions = self._pattern_based_detection(test_dict, age, gender)
        conditions.extend(pattern_conditions)
        
        # Remove duplicates and merge similar conditions
        conditions = self._merge_similar_conditions(conditions)
        
        return conditions
    
    def _rule_based_detection(self, test_dict: Dict, age: int, gender: str) -> List[Condition]:
        """Improved rule-based condition detection"""
        conditions = []
        
        # Check for anemia (corrected)
        anemia_condition = self._detect_anemia_corrected(test_dict, age, gender)
        if anemia_condition:
            conditions.append(anemia_condition)
        
        # Check for diabetes/prediabetes (corrected)
        diabetes_condition = self._detect_diabetes_corrected(test_dict, age)
        if diabetes_condition:
            conditions.append(diabetes_condition)
        
        # Check cardiovascular risk (improved)
        cardio_condition = self._detect_cardiovascular_risk_improved(test_dict, age, gender)
        if cardio_condition:
            conditions.append(cardio_condition)
        
        # Check thyroid dysfunction (corrected)
        thyroid_condition = self._detect_thyroid_dysfunction_corrected(test_dict, age)
        if thyroid_condition:
            conditions.append(thyroid_condition)
        
        # Check for elevated RDW (iron deficiency indicator)
        rdw_condition = self._detect_elevated_rdw(test_dict)
        if rdw_condition:
            conditions.append(rdw_condition)
        
        # Check for lymphocytosis
        lymphocytosis_condition = self._detect_lymphocytosis(test_dict)
        if lymphocytosis_condition:
            conditions.append(lymphocytosis_condition)
        
        # Check for liver dysfunction (NEW)
        liver_condition = self._detect_liver_dysfunction(test_dict)
        if liver_condition:
            conditions.append(liver_condition)
        
        return conditions
    
    def _detect_anemia_corrected(self, test_dict: Dict, age: int, gender: str) -> Optional[Condition]:
        """Corrected anemia detection"""
        hb_value = test_dict.get("hemoglobin")
        pcv_value = test_dict.get("pcv")
        rdw_value = test_dict.get("rdw_cv")  # Corrected key
        
        if hb_value is None and pcv_value is None:
            return None
        
        # Use corrected thresholds
        if gender.lower() == "male":
            hb_threshold = 13.0
            pcv_threshold = 40.0
        else:
            hb_threshold = 12.0
            pcv_threshold = 36.0
        
        # Check for anemia
        anemia_detected = False
        primary_indicator = ""
        
        if hb_value is not None and hb_value < hb_threshold:
            anemia_detected = True
            primary_indicator = f"Low hemoglobin ({hb_value} g/dL)"
        elif pcv_value is not None and pcv_value < pcv_threshold:
            anemia_detected = True
            primary_indicator = f"Low PCV ({pcv_value}%)"
        
        if anemia_detected:
            severity = self._calculate_anemia_severity(hb_value or (pcv_value * 0.33), hb_threshold)
            anemia_type = self._determine_anemia_type_corrected(test_dict)
            
            return Condition(
                name=f"Anemia ({anemia_type})",
                severity=severity,
                description=f"{primary_indicator} indicates anemia",
                symptoms=["Fatigue", "Weakness", "Pale skin", "Shortness of breath"],
                recommendations=[
                    "Iron studies (serum iron, ferritin, TIBC)",
                    "Increase iron-rich foods",
                    "Consider iron supplements if iron deficient",
                    "Consult hematologist if severe",
                    "Rule out bleeding sources"
                ],
                follow_up="Repeat CBC with iron studies in 4-6 weeks",
                risk_factors=["Iron deficiency", "Chronic disease", "Blood loss"],
                confidence_score=0.85,
                detection_method="rule_based"
            )
        
        return None
    
    def _detect_diabetes_corrected(self, test_dict: Dict, age: int) -> Optional[Condition]:
        """Corrected diabetes detection"""
        hba1c = test_dict.get("hba1c")
        glucose = test_dict.get("glucose_fasting")
        
        if hba1c is None and glucose is None:
            return None
        
        # Check HbA1c first (more reliable)
        if hba1c is not None:
            if hba1c >= 6.5:
                return self._create_diabetes_condition_corrected(hba1c, glucose, "Type 2 Diabetes")
            elif hba1c >= 5.7:
                return self._create_diabetes_condition_corrected(hba1c, glucose, "Prediabetes")
        
        # Check fasting glucose
        if glucose is not None:
            if glucose >= 126:
                return self._create_diabetes_condition_corrected(hba1c, glucose, "Type 2 Diabetes")
            elif glucose >= 100:
                return self._create_diabetes_condition_corrected(hba1c, glucose, "Prediabetes")
        
        return None
    
    def _detect_thyroid_dysfunction_corrected(self, test_dict: Dict, age: int) -> Optional[Condition]:
        """Corrected thyroid dysfunction detection using proper ranges"""
        t3_value = test_dict.get("t3")
        t4_value = test_dict.get("t4")
        tsh_value = test_dict.get("tsh")
        
        # Normal ranges: T3: 35-193, T4: 4.87-11.2, TSH: 0.35-4.94
        conditions_found = []
        
        # Only flag if clearly outside normal ranges
        if t3_value is not None and (t3_value < 35 or t3_value > 193):
            if t3_value < 35:
                conditions_found.append(("Low T3", t3_value, "ng/dL"))
            else:
                conditions_found.append(("High T3", t3_value, "ng/dL"))
        
        if t4_value is not None and (t4_value < 4.87 or t4_value > 11.2):
            if t4_value < 4.87:
                conditions_found.append(("Low T4", t4_value, "μg/dL"))
            else:
                conditions_found.append(("High T4", t4_value, "μg/dL"))
        
        if tsh_value is not None and (tsh_value < 0.35 or tsh_value > 4.94):
            if tsh_value < 0.35:
                conditions_found.append(("Low TSH", tsh_value, "μIU/mL"))
            else:
                conditions_found.append(("High TSH", tsh_value, "μIU/mL"))
        
        # Only return condition if we have clear abnormalities
        if conditions_found:
            # Determine primary dysfunction type
            dysfunction_type = "Thyroid Dysfunction"
            severity = Severity.MILD
            
            # Classic patterns
            if tsh_value and t4_value:
                if tsh_value > 4.94 and t4_value < 4.87:
                    dysfunction_type = "Primary Hypothyroidism"
                    severity = Severity.MODERATE if tsh_value > 10 else Severity.MODERATE
                elif tsh_value < 0.35 and t4_value > 11.2:
                    dysfunction_type = "Primary Hyperthyroidism"
                    severity = Severity.MODERATE
            
            description = f"Abnormal thyroid function: {', '.join([f'{name} ({value} {unit})' for name, value, unit in conditions_found])}"
            
            return Condition(
                name=dysfunction_type,
                severity=severity,
                description=description,
                symptoms=["Fatigue", "Weight changes", "Temperature sensitivity", "Heart rate changes"],
                recommendations=[
                    "Complete thyroid panel (TSH, Free T4, Free T3)",
                    "Consult endocrinologist",
                    "Monitor symptoms",
                    "Consider thyroid antibodies if indicated"
                ],
                follow_up="Repeat thyroid function tests in 6-8 weeks",
                risk_factors=["Family history", "Age", "Gender", "Autoimmune conditions"],
                confidence_score=0.8,
                detection_method="rule_based"
            )
        
        return None
    
    def _detect_elevated_rdw(self, test_dict: Dict) -> Optional[Condition]:
        """Detect elevated RDW which may indicate iron deficiency or blood disorders"""
        rdw_value = test_dict.get("rdw_cv")
        
        if rdw_value is not None and rdw_value > 14.0:  # Reference range: 11.6-14.0%
            severity = Severity.MILD if rdw_value <= 16.0 else Severity.MODERATE
            
            return Condition(
                name="Elevated RDW (Iron Deficiency Indicator)",
                severity=severity,
                description=f"Elevated RDW ({rdw_value}%) suggests possible iron deficiency or early blood disorders",
                symptoms=["Fatigue", "Weakness", "Pale skin", "Shortness of breath"],
                recommendations=[
                    "Iron studies (serum iron, ferritin, TIBC)",
                    "Increase iron-rich foods",
                    "Consider iron supplements if iron deficient",
                    "Rule out blood disorders",
                    "Consult hematologist if persistent"
                ],
                follow_up="Repeat CBC with iron studies in 4-6 weeks",
                risk_factors=["Iron deficiency", "B12/folate deficiency", "Blood disorders"],
                confidence_score=0.8,
                detection_method="rule_based"
            )
        
        return None
    
    def _detect_lymphocytosis(self, test_dict: Dict) -> Optional[Condition]:
        """Detect lymphocytosis (elevated lymphocyte count)"""
        lymphocytes_value = test_dict.get("lymphocytes")
        
        if lymphocytes_value is not None and lymphocytes_value > 40:  # Reference range: 20-40%
            severity = Severity.MILD if lymphocytes_value <= 50 else Severity.MODERATE
            
            return Condition(
                name="Lymphocytosis",
                severity=severity,
                description=f"Elevated lymphocyte count ({lymphocytes_value}%) - may indicate infection or immune response",
                symptoms=["Often asymptomatic", "Possible fatigue", "Mild fever"],
                recommendations=[
                    "Rule out viral infections",
                    "Monitor for symptoms",
                    "Consider repeat CBC in 2-4 weeks",
                    "Consult physician if persistent or symptomatic"
                ],
                follow_up="Repeat CBC in 2-4 weeks to monitor trend",
                risk_factors=["Viral infections", "Immune response", "Chronic inflammation"],
                confidence_score=0.75,
                detection_method="rule_based"
            )
        
        return None
    
    def _detect_liver_dysfunction(self, test_dict: Dict) -> Optional[Condition]:
        """Detect liver dysfunction based on liver enzyme levels"""
        alt_value = test_dict.get("sgpt")  # ALT/SGPT
        ast_value = test_dict.get("sgot")  # AST/SGOT
        alp_value = test_dict.get("alp")   # Alkaline Phosphatase
        bilirubin_value = test_dict.get("total_bilirubin")
        albumin_value = test_dict.get("albumin")
        
        # Reference ranges
        alt_threshold = 50  # U/L
        ast_threshold = 40  # U/L (AST typically lower than ALT)
        alp_threshold = 130  # U/L
        bilirubin_threshold = 1.2  # mg/dL
        albumin_threshold = 3.5  # g/dL (low = liver dysfunction)
        
        abnormal_markers = []
        severity = Severity.MILD
        
        # Check ALT (SGPT)
        if alt_value is not None and alt_value > alt_threshold:
            abnormal_markers.append(f"ALT (SGPT): {alt_value} U/L (normal < {alt_threshold})")
            if alt_value > alt_threshold * 3:  # >150
                severity = Severity.MODERATE if severity == Severity.MILD else Severity.SEVERE
            elif alt_value > alt_threshold * 2:  # >100
                severity = max(severity, Severity.MODERATE)
        
        # Check AST (SGOT)
        if ast_value is not None and ast_value > ast_threshold:
            abnormal_markers.append(f"AST (SGOT): {ast_value} U/L (normal < {ast_threshold})")
            if ast_value > ast_threshold * 3:
                severity = Severity.MODERATE if severity == Severity.MILD else Severity.SEVERE
        
        # Check Alkaline Phosphatase
        if alp_value is not None and alp_value > alp_threshold:
            abnormal_markers.append(f"Alkaline Phosphatase: {alp_value} U/L (normal < {alp_threshold})")
        
        # Check Bilirubin
        if bilirubin_value is not None and bilirubin_value > bilirubin_threshold:
            abnormal_markers.append(f"Total Bilirubin: {bilirubin_value} mg/dL (normal < {bilirubin_threshold})")
            severity = max(severity, Severity.MODERATE)
        
        # Check Albumin (low = concerning)
        if albumin_value is not None and albumin_value < albumin_threshold:
            abnormal_markers.append(f"Albumin: {albumin_value} g/dL (low, normal > {albumin_threshold})")
            severity = Severity.MODERATE if severity == Severity.MILD else Severity.SEVERE
        
        if abnormal_markers:
            return Condition(
                name="Liver Dysfunction",
                severity=severity,
                description=f"Abnormal liver function markers detected: {'; '.join(abnormal_markers)}",
                symptoms=["Jaundice (yellowing)", "Fatigue", "Abdominal pain", "Dark urine", "Pale stools"],
                recommendations=[
                    "Consult hepatologist or gastroenterologist",
                    "Avoid alcohol and hepatotoxic medications",
                    "Complete hepatitis screening (A, B, C)",
                    "Liver ultrasound or imaging if indicated",
                    "Monitor liver enzyme trends",
                    "Dietary modifications (limit fat, increase fruits/vegetables)"
                ],
                follow_up="Repeat liver function tests in 2-4 weeks",
                risk_factors=["Alcohol use", "Hepatitis", "Fatty liver disease", "Drug toxicity", "Autoimmune liver disease"],
                confidence_score=0.85,
                detection_method="rule_based"
            )
        
        return None
    
    def _ml_based_detection(self, test_dict: Dict, age: int, gender: str) -> List[Condition]:

        """ML-based condition detection"""
        conditions = []
        
        try:
            features = self.ml_predictor.prepare_features(test_dict, age, gender)
            
            # Check each condition type
            ml_conditions = ['anemia', 'diabetes', 'cardiovascular_risk', 'thyroid_dysfunction']
            
            for condition_type in ml_conditions:
                prediction, confidence = self.ml_predictor.predict_condition(features, condition_type)
                
                if prediction and confidence > 0.7:  # High confidence threshold
                    # Additional validation for thyroid dysfunction to prevent false positives
                    if condition_type == 'thyroid_dysfunction':
                        # Check if thyroid values are actually normal
                        t3_value = test_dict.get("t3")
                        t4_value = test_dict.get("t4")
                        tsh_value = test_dict.get("tsh")
                        
                        # If all thyroid values are within normal range, skip ML prediction
                        if (t3_value is not None and 35 <= t3_value <= 193 and
                            t4_value is not None and 4.87 <= t4_value <= 11.2 and
                            tsh_value is not None and 0.35 <= tsh_value <= 4.94):
                            continue  # Skip this prediction as thyroid values are normal
                    
                    severity = Severity.MODERATE if confidence > 0.8 else Severity.MILD
                    
                    condition = Condition(
                        name=f"AI-Detected {condition_type.replace('_', ' ').title()}",
                        severity=severity,
                        description=f"ML model detected {condition_type} with {confidence:.1%} confidence",
                        symptoms=self._get_symptoms_for_condition(condition_type),
                        recommendations=self._get_ml_recommendations(condition_type),
                        follow_up="Confirm with additional testing",
                        risk_factors=["Pattern detected by AI analysis"],
                        confidence_score=confidence,
                        detection_method="ml_model"
                    )
                    conditions.append(condition)
        
        except Exception as e:
            logger.error(f"ML detection failed: {str(e)}")
        
        return conditions
    
    def _pattern_based_detection(self, test_dict: Dict, age: int, gender: str) -> List[Condition]:
        """Pattern-based condition detection"""
        conditions = []
        
        patterns = self.pattern_matcher.detect_patterns(test_dict, age, gender)
        
        for pattern in patterns:
            if pattern["pattern"] == "metabolic_syndrome_risk":
                condition = Condition(
                    name="Metabolic Syndrome Risk",
                    severity=Severity.MODERATE,
                    description=f"Pattern analysis shows {pattern['criteria_met']} metabolic syndrome criteria met",
                    symptoms=["Abdominal obesity", "Insulin resistance", "High blood pressure"],
                    recommendations=[
                        "Lifestyle modification program",
                        "Weight management",
                        "Regular exercise",
                        "Dietary counseling",
                        "Monitor blood pressure"
                    ],
                    follow_up="Comprehensive metabolic evaluation in 3 months",
                    risk_factors=pattern["details"],
                    confidence_score=pattern["confidence"],
                    detection_method="pattern_matching"
                )
                conditions.append(condition)
            
            elif pattern["pattern"] == "insulin_resistance":
                condition = Condition(
                    name="Insulin Resistance Pattern",
                    severity=Severity.MILD,
                    description="Pattern suggests insulin resistance",
                    symptoms=["Fatigue after meals", "Cravings for sweets", "Difficulty losing weight"],
                    recommendations=[
                        "Low glycemic index diet",
                        "Regular physical activity",
                        "Weight management",
                        "Consider insulin resistance testing"
                    ],
                    follow_up="OGTT and insulin levels in 6 weeks",
                    risk_factors=pattern["indicators"],
                    confidence_score=pattern["confidence"],
                    detection_method="pattern_matching"
                )
                conditions.append(condition)
        
        return conditions
    
    def _merge_similar_conditions(self, conditions: List[Condition]) -> List[Condition]:
        """Merge similar conditions and remove duplicates"""
        merged = []
        condition_groups = {}
        
        # Group similar conditions
        for condition in conditions:
            base_name = condition.name.lower().replace("ai-detected", "").strip()
            if base_name not in condition_groups:
                condition_groups[base_name] = []
            condition_groups[base_name].append(condition)
        
        # Merge each group
        for group_name, group_conditions in condition_groups.items():
            if len(group_conditions) == 1:
                merged.append(group_conditions[0])
            else:
                # Create merged condition with highest confidence
                best_condition = max(group_conditions, key=lambda c: c.confidence_score)
                
                # Combine detection methods
                methods = list(set([c.detection_method for c in group_conditions]))
                best_condition.detection_method = "+".join(methods)
                
                # Average confidence scores
                avg_confidence = sum([c.confidence_score for c in group_conditions]) / len(group_conditions)
                best_condition.confidence_score = avg_confidence
                
                merged.append(best_condition)
        
        return merged
    
    # Helper methods
    def _calculate_anemia_severity(self, hb_value: float, threshold: float) -> Severity:
        """Calculate anemia severity"""
        if hb_value is None:
            return Severity.MILD
        
        deficit = threshold - hb_value
        if deficit < 1.0:
            return Severity.MILD
        elif deficit < 3.0:
            return Severity.MODERATE
        else:
            return Severity.SEVERE
    
    def _determine_anemia_type_corrected(self, test_dict: Dict) -> str:
        """Determine anemia type with corrected logic"""
        mcv = test_dict.get("mcv")
        rdw = test_dict.get("rdw_cv")  # Corrected key
        esr = test_dict.get("esr")
        
        if mcv is not None:
            if mcv < 80:
                if rdw is not None and rdw > 15:
                    return "Iron Deficiency"
                else:
                    return "Microcytic Anemia"
            elif mcv > 100:
                return "Megaloblastic Anemia"
            else:
                if esr is not None and esr > 20:
                    return "Chronic Disease"
                else:
                    return "Normocytic Anemia"
        
        return "Unclassified Anemia"
    
    def _create_diabetes_condition_corrected(self, hba1c: Optional[float], glucose: Optional[float], condition_type: str) -> Condition:
        """Create corrected diabetes condition"""
        primary_indicator = ""
        if hba1c is not None:
            primary_indicator = f"HbA1c: {hba1c}%"
        if glucose is not None:
            if primary_indicator:
                primary_indicator += f", Fasting glucose: {glucose} mg/dL"
            else:
                primary_indicator = f"Fasting glucose: {glucose} mg/dL"
        
        if condition_type == "Type 2 Diabetes":
            severity = Severity.SEVERE if (hba1c and hba1c > 8.0) or (glucose and glucose > 200) else Severity.MODERATE
            recommendations = [
                "Immediate endocrinologist consultation",
                "Diabetes education program",
                "Blood glucose monitoring",
                "Medication management",
                "Diabetic diet plan",
                "Regular exercise program",
                "Eye and foot examinations"
            ]
            follow_up = "Urgent follow-up within 1 week"
        else:  # Prediabetes
            severity = Severity.MILD
            recommendations = [
                "Lifestyle modification program",
                "Weight loss if overweight (5-10% body weight)",
                "150 minutes moderate exercise weekly",
                "Mediterranean or DASH diet",
                "Regular HbA1c monitoring every 6 months",
                "Annual diabetes screening"
            ]
            follow_up = "Lifestyle counseling and recheck in 3 months"
        
        return Condition(
            name=condition_type,
            severity=severity,
            description=f"{primary_indicator} indicates {condition_type.lower()}",
            symptoms=["Increased thirst", "Frequent urination", "Fatigue", "Blurred vision", "Slow wound healing"],
            recommendations=recommendations,
            follow_up=follow_up,
            risk_factors=["Family history", "Obesity", "Sedentary lifestyle", "Age > 45"],
            confidence_score=0.9,
            detection_method="rule_based"
        )
    
    def _detect_cardiovascular_risk_improved(self, test_dict: Dict, age: int, gender: str) -> Optional[Condition]:
        """Improved cardiovascular risk detection"""
        risk_factors = []
        risk_score = 0
        
        # HDL cholesterol
        hdl = test_dict.get("hdl_cholesterol")
        if hdl is not None:
            threshold = 40 if gender.lower() == "male" else 50
            if hdl < threshold:
                risk_factors.append(f"Low HDL cholesterol ({hdl} mg/dL)")
                risk_score += 2
        
        # LDL cholesterol - CORRECTED THRESHOLD
        ldl = test_dict.get("ldl_cholesterol")
        if ldl is not None:
            if ldl > 160:
                risk_factors.append(f"High LDL cholesterol ({ldl} mg/dL)")
                risk_score += 3
            elif ldl > 100:  # CORRECTED: 100 is the actual threshold for optimal
                risk_factors.append(f"Above optimal LDL cholesterol ({ldl} mg/dL)")
                risk_score += 1
        
        # Triglycerides
        tg = test_dict.get("triglycerides")
        if tg is not None:
            if tg > 200:
                risk_factors.append(f"High triglycerides ({tg} mg/dL)")
                risk_score += 2
            elif tg > 150:
                risk_factors.append(f"Borderline high triglycerides ({tg} mg/dL)")
                risk_score += 1
        
        # Total cholesterol
        total_chol = test_dict.get("total_cholesterol")
        if total_chol is not None and total_chol > 240:
            risk_factors.append(f"High total cholesterol ({total_chol} mg/dL)")
            risk_score += 2
        
        # Age factor
        age_threshold = 45 if gender.lower() == "male" else 55
        if age > age_threshold:
            risk_score += 1
        
        if risk_factors:
            # Determine severity based on risk score
            if risk_score >= 5:
                severity = Severity.SEVERE
                priority = "High priority"
            elif risk_score >= 3:
                severity = Severity.MODERATE
                priority = "Moderate priority"
            else:
                severity = Severity.MILD
                priority = "Low priority"
            
            return Condition(
                name="Cardiovascular Risk Factors",
                severity=severity,
                description=f"Multiple cardiovascular risk factors detected ({priority})",
                symptoms=["Often asymptomatic", "Possible chest discomfort", "Shortness of breath with exertion"],
                recommendations=[
                    "Cardiology consultation",
                    "Heart-healthy diet (Mediterranean or DASH)",
                    "Regular aerobic exercise (150 min/week)",
                    "Weight management if overweight", 
                    "Consider statin therapy",
                    "Blood pressure monitoring",
                    "Smoking cessation if applicable"
                ],
                follow_up="Repeat lipid panel in 6-12 weeks after lifestyle changes",
                risk_factors=risk_factors,
                confidence_score=0.85,
                detection_method="rule_based"
            )
        
        return None
    
    def _get_symptoms_for_condition(self, condition_type: str) -> List[str]:
        """Get symptoms for ML-detected conditions"""
        symptom_map = {
            "anemia": ["Fatigue", "Weakness", "Pale skin", "Shortness of breath"],
            "diabetes": ["Excessive thirst", "Frequent urination", "Fatigue", "Blurred vision"],
            "cardiovascular_risk": ["Chest discomfort", "Shortness of breath", "Fatigue"],
            "thyroid_dysfunction": ["Fatigue", "Weight changes", "Temperature sensitivity"]
        }
        return symptom_map.get(condition_type, ["Consult healthcare provider for symptoms"])
    
    def _get_ml_recommendations(self, condition_type: str) -> List[str]:
        """Get recommendations for ML-detected conditions"""
        rec_map = {
            "anemia": ["Confirm with iron studies", "Increase iron-rich foods", "Medical evaluation"],
            "diabetes": ["Confirm with additional glucose testing", "Lifestyle modifications", "Medical consultation"],
            "cardiovascular_risk": ["Confirm with comprehensive lipid panel", "Heart-healthy lifestyle", "Cardiology evaluation"],
            "thyroid_dysfunction": ["Confirm with complete thyroid panel", "Endocrinology consultation"]
        }
        return rec_map.get(condition_type, ["Confirm findings with healthcare provider"])
    
    def conditions_to_dict(self, conditions: List[Condition]) -> List[Dict]:
        """Convert Condition objects to dictionary format with AI/ML enhancements"""
        return [
            {
                "name": condition.name,
                "severity": condition.severity.value,
                "description": condition.description,
                "symptoms": condition.symptoms,
                "recommendations": condition.recommendations,
                "follow_up": condition.follow_up,
                "risk_factors": condition.risk_factors,
                "confidence_score": condition.confidence_score,
                "detection_method": condition.detection_method,
                "ai_enhanced": condition.detection_method in ["ml_model", "pattern_matching", "rule_based+ml_model"]
            }
            for condition in conditions
        ]

    def get_model_info(self) -> Dict:
        """Get information about trained models"""
        return {
            "ml_models_available": list(self.ml_predictor.models.keys()) if self.ml_predictor.is_trained else [],
            "model_trained": self.ml_predictor.is_trained,
            "pattern_rules_loaded": len(self.pattern_matcher.pattern_rules),
            "condition_rules_loaded": len(self.condition_rules),
            "ai_capabilities": [
                "Machine Learning Prediction",
                "Pattern Recognition", 
                "Advanced Rule-Based Analysis",
                "Multi-Method Validation",
                "Confidence Scoring"
            ]
        }
    
    def save_models(self, model_path: str = "models/"):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs(model_path, exist_ok=True)
            
            for condition, model in self.ml_predictor.models.items():
                model_file = f"{model_path}/model_{condition}.joblib"
                scaler_file = f"{model_path}/scaler_{condition}.joblib"
                
                joblib.dump(model, model_file)
                joblib.dump(self.ml_predictor.scalers[condition], scaler_file)
            
            logger.info(f"Models saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save models: {str(e)}")
    
    def load_models(self, model_path: str = "models/"):
        """Load pre-trained models from disk"""
        try:
            import os
            if not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist")
                return
            
            conditions = ['anemia', 'diabetes', 'cardiovascular_risk', 'thyroid_dysfunction']
            for condition in conditions:
                model_file = f"{model_path}/model_{condition}.joblib"
                scaler_file = f"{model_path}/scaler_{condition}.joblib"
                
                if os.path.exists(model_file) and os.path.exists(scaler_file):
                    self.ml_predictor.models[condition] = joblib.load(model_file)
                    self.ml_predictor.scalers[condition] = joblib.load(scaler_file)
            
            self.ml_predictor.is_trained = len(self.ml_predictor.models) > 0
            logger.info(f"Loaded {len(self.ml_predictor.models)} models from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")

# Alias for backward compatibility
ConditionAnalyzer = EnhancedConditionAnalyzer