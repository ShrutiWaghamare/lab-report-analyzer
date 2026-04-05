"""
Enhanced Emergency Checker Module with AI/ML capabilities
Flags life-threatening values and uses ML for pattern recognition.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EmergencyFlag:
    condition: str
    value: str
    action: str
    urgency: str
    threshold: Optional[float] = None
    test_name: Optional[str] = None
    confidence: Optional[float] = None
    ml_prediction: bool = False

@dataclass
class PatientData:
    age: int
    gender: str
    test_results: List[Dict]
    medical_history: Optional[List[str]] = field(default_factory=list)
    medications: Optional[List[str]] = field(default_factory=list)

class EmergencyChecker:
    """Enhanced emergency checker with AI/ML capabilities."""
    
    def __init__(self, enable_ml: bool = True):
        self.emergency_thresholds = self._load_emergency_thresholds()
        self.enable_ml = enable_ml
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.emergency_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        if enable_ml:
            self._initialize_ml_models()
    
    def _load_emergency_thresholds(self) -> Dict:
        """Load emergency thresholds for critical values with corrections"""
        return {
            "hemoglobin": {
                "critical_low": 7.0,
                "severe_low": 8.5,
                "normal_range": (12.0, 16.0),  # Added normal ranges
                "action_critical": "Seek immediate medical attention - may require blood transfusion",
                "action_severe": "Urgent hematology consultation required"
            },
            "glucose_fasting": {
                "critical_high": 400,
                "severe_high": 250,  # Lowered from 300 for better sensitivity
                "critical_low": 50,
                "severe_low": 70,    # Added severe low threshold
                "normal_range": (70, 100),
                "action_critical_high": "Emergency room visit required - risk of diabetic coma",
                "action_severe_high": "Urgent diabetes management required",
                "action_critical_low": "Emergency treatment for hypoglycemia required",
                "action_severe_low": "Monitor for hypoglycemia symptoms"
            },
            "creatinine": {
                "critical_high": 4.0,
                "severe_high": 2.0,  # Lowered for better sensitivity
                "normal_range": (0.6, 1.3),
                "action_critical": "Immediate nephrology consultation required",
                "action_severe": "Urgent kidney function evaluation needed"
            },
            "sodium": {
                "critical_low": 120,
                "critical_high": 160,
                "severe_low": 125,   # Added severe thresholds
                "severe_high": 155,
                "normal_range": (135, 145),
                "action_critical_low": "Emergency treatment for hyponatremia required",
                "action_critical_high": "Emergency treatment for hypernatremia required",
                "action_severe_low": "Urgent sodium correction needed",
                "action_severe_high": "Urgent sodium correction needed"
            },
            "potassium": {
                "critical_low": 2.5,
                "critical_high": 6.5,
                "severe_low": 3.0,   # Added severe thresholds
                "severe_high": 5.5,
                "normal_range": (3.5, 5.0),
                "action_critical_low": "Emergency treatment for hypokalemia required",
                "action_critical_high": "Emergency treatment for hyperkalemia required",
                "action_severe_low": "Urgent potassium replacement needed",
                "action_severe_high": "Urgent potassium management required"
            },
            "esr": {
                "critical_high": 100,
                "severe_high": 50,
                "normal_range": (0, 20),
                "action_critical": "Urgent medical evaluation for severe infection or inflammatory condition",
                "action_severe": "Urgent evaluation for infection or inflammatory condition"
            },
            "platelets": {
                "critical_low": 20,
                "severe_low": 50,
                "normal_range": (150, 450),
                "action_critical": "Emergency hematology consultation - risk of bleeding",
                "action_severe": "Urgent hematology evaluation required"
            },
            "wbc_count": {
                "critical_low": 1.0,
                "critical_high": 30.0,
                "severe_low": 2.0,   # Added severe thresholds
                "severe_high": 20.0,
                "normal_range": (4.0, 11.0),
                "action_critical_low": "Emergency evaluation for severe infection risk",
                "action_critical_high": "Emergency evaluation for severe infection or leukemia",
                "action_severe_low": "Urgent evaluation for low white cell count",
                "action_severe_high": "Urgent evaluation for high white cell count"
            },
            "bilirubin_total": {  # Added new test
                "critical_high": 20.0,
                "severe_high": 10.0,
                "normal_range": (0.2, 1.2),
                "action_critical": "Emergency evaluation for severe liver dysfunction",
                "action_severe": "Urgent liver function evaluation required"
            },
            "troponin": {  # Added cardiac marker
                "critical_high": 10.0,
                "severe_high": 1.0,
                "normal_range": (0.0, 0.04),
                "action_critical": "Emergency cardiology consultation - possible heart attack",
                "action_severe": "Urgent cardiac evaluation required"
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models with synthetic training data"""
        try:
            # Generate synthetic training data for demonstration
            # In production, this would use real historical data
            training_data = self._generate_synthetic_training_data()
            
            if len(training_data) > 0:
                self._train_models(training_data)
                logger.info("ML models initialized successfully")
            else:
                logger.warning("No training data available, ML features disabled")
                self.enable_ml = False
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
            self.enable_ml = False
    
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for ML models"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal cases
        normal_data = []
        for _ in range(int(n_samples * 0.8)):
            record = {
                'age': np.random.normal(45, 15),
                'gender_encoded': np.random.choice([0, 1]),  # 0: female, 1: male
                'hemoglobin': np.random.normal(14, 2),
                'glucose_fasting': np.random.normal(90, 15),
                'creatinine': np.random.normal(1.0, 0.3),
                'sodium': np.random.normal(140, 5),
                'potassium': np.random.normal(4.2, 0.5),
                'wbc_count': np.random.normal(7, 2),
                'platelets': np.random.normal(250, 50),
                'esr': np.random.normal(15, 8),
                'emergency': 0
            }
            normal_data.append(record)
        
        # Generate emergency cases
        emergency_data = []
        for _ in range(int(n_samples * 0.2)):
            # Create emergency patterns
            emergency_type = np.random.choice(['anemia', 'diabetes', 'kidney', 'infection'])
            
            if emergency_type == 'anemia':
                record = {
                    'age': np.random.normal(55, 15),
                    'gender_encoded': np.random.choice([0, 1]),
                    'hemoglobin': np.random.normal(6.5, 1),  # Critical low
                    'glucose_fasting': np.random.normal(90, 15),
                    'creatinine': np.random.normal(1.0, 0.3),
                    'sodium': np.random.normal(140, 5),
                    'potassium': np.random.normal(4.2, 0.5),
                    'wbc_count': np.random.normal(7, 2),
                    'platelets': np.random.normal(30, 10),  # Low platelets
                    'esr': np.random.normal(60, 20),
                    'emergency': 1
                }
            elif emergency_type == 'diabetes':
                record = {
                    'age': np.random.normal(50, 15),
                    'gender_encoded': np.random.choice([0, 1]),
                    'hemoglobin': np.random.normal(12, 2),
                    'glucose_fasting': np.random.normal(350, 50),  # Critical high
                    'creatinine': np.random.normal(1.5, 0.5),
                    'sodium': np.random.normal(140, 5),
                    'potassium': np.random.normal(4.2, 0.5),
                    'wbc_count': np.random.normal(10, 3),
                    'platelets': np.random.normal(250, 50),
                    'esr': np.random.normal(25, 10),
                    'emergency': 1
                }
            elif emergency_type == 'kidney':
                record = {
                    'age': np.random.normal(60, 15),
                    'gender_encoded': np.random.choice([0, 1]),
                    'hemoglobin': np.random.normal(10, 2),
                    'glucose_fasting': np.random.normal(90, 15),
                    'creatinine': np.random.normal(3.5, 0.5),  # Critical high
                    'sodium': np.random.normal(125, 5),  # Critical low
                    'potassium': np.random.normal(6.0, 0.5),  # High
                    'wbc_count': np.random.normal(7, 2),
                    'platelets': np.random.normal(200, 40),
                    'esr': np.random.normal(40, 15),
                    'emergency': 1
                }
            else:  # infection
                record = {
                    'age': np.random.normal(40, 20),
                    'gender_encoded': np.random.choice([0, 1]),
                    'hemoglobin': np.random.normal(11, 2),
                    'glucose_fasting': np.random.normal(120, 20),
                    'creatinine': np.random.normal(1.0, 0.3),
                    'sodium': np.random.normal(140, 5),
                    'potassium': np.random.normal(4.2, 0.5),
                    'wbc_count': np.random.normal(25, 5),  # Critical high
                    'platelets': np.random.normal(400, 80),
                    'esr': np.random.normal(80, 20),  # High
                    'emergency': 1
                }
            
            emergency_data.append(record)
        
        all_data = normal_data + emergency_data
        return pd.DataFrame(all_data)
    
    def _train_models(self, training_data: pd.DataFrame):
        """Train ML models with the provided data"""
        try:
            # Prepare features and target
            feature_columns = ['age', 'gender_encoded', 'hemoglobin', 'glucose_fasting', 
                             'creatinine', 'sodium', 'potassium', 'wbc_count', 'platelets', 'esr']
            X = training_data[feature_columns]
            y = training_data['emergency']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train anomaly detector (unsupervised)
            self.anomaly_detector.fit(X_train_scaled)
            
            # Train emergency classifier (supervised)
            self.emergency_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.emergency_classifier.predict(X_test_scaled)
            logger.info("Emergency classifier performance:")
            logger.info("\n" + classification_report(y_test, y_pred))
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            self.enable_ml = False
    
    def _prepare_ml_features(self, patient_data: PatientData) -> Optional[np.ndarray]:
        """Prepare features for ML model prediction"""
        try:
            # Create test dictionary
            test_dict = {test["test_name"].lower().replace(" ", "_"): test["value"] 
                        for test in patient_data.test_results}
            
            # Gender encoding
            gender_encoded = 1 if patient_data.gender.lower() == 'male' else 0
            
            # Extract features (use defaults for missing values)
            features = [
                patient_data.age,
                gender_encoded,
                test_dict.get('hemoglobin', 13.0),
                test_dict.get('glucose_fasting', 90.0),
                test_dict.get('creatinine', 1.0),
                test_dict.get('sodium', 140.0),
                test_dict.get('potassium', 4.2),
                test_dict.get('wbc_count', 7.0),
                test_dict.get('platelets', 250.0),
                test_dict.get('esr', 15.0)
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return None
    
    def check_emergency_flags(self, patient_data: PatientData) -> List[EmergencyFlag]:
        """
        Check for emergency conditions using both rule-based and ML approaches.
        
        Args:
            patient_data: PatientData object containing patient information
            
        Returns:
            List of EmergencyFlag objects
        """
        emergency_flags = []
        
        # Rule-based checking
        test_dict = {test["test_name"].lower().replace(" ", "_"): test["value"] 
                    for test in patient_data.test_results}
        
        # Check each test for emergency conditions
        for test_name, value in test_dict.items():
            flag = self._check_single_test_emergency(test_name, value, patient_data.age, patient_data.gender)
            if flag:
                emergency_flags.append(flag)
        
        # Check combination emergencies
        combination_flags = self.check_combination_emergencies(patient_data)
        emergency_flags.extend(combination_flags)
        
        # ML-based checking
        if self.enable_ml and self.is_trained:
            ml_flags = self._check_ml_emergencies(patient_data)
            emergency_flags.extend(ml_flags)
        
        # Remove duplicates and sort by urgency
        emergency_flags = self._deduplicate_flags(emergency_flags)
        emergency_flags = sorted(emergency_flags, key=lambda x: 0 if x.urgency == "CRITICAL" else 1)
        
        return emergency_flags
    
    def _check_single_test_emergency(self, test_name: str, value: float, age: int, gender: str) -> Optional[EmergencyFlag]:
        """Check if a single test result indicates an emergency condition"""
        
        if test_name not in self.emergency_thresholds:
            return None
        
        thresholds = self.emergency_thresholds[test_name]
        
        # Adjust thresholds based on age and gender if needed
        adjusted_thresholds = self._adjust_thresholds_for_demographics(thresholds, age, gender, test_name)
        
        # Check for critical values
        if "critical_low" in adjusted_thresholds and value <= adjusted_thresholds["critical_low"]:
            return EmergencyFlag(
                condition=f"Critical Low {test_name.replace('_', ' ').title()}",
                value=f"{test_name.replace('_', ' ').title()}: {value}",
                action=adjusted_thresholds.get("action_critical_low", adjusted_thresholds.get("action_critical", "Seek immediate medical attention")),
                urgency="CRITICAL",
                threshold=adjusted_thresholds["critical_low"],
                test_name=test_name
            )
        
        if "critical_high" in adjusted_thresholds and value >= adjusted_thresholds["critical_high"]:
            return EmergencyFlag(
                condition=f"Critical High {test_name.replace('_', ' ').title()}",
                value=f"{test_name.replace('_', ' ').title()}: {value}",
                action=adjusted_thresholds.get("action_critical_high", adjusted_thresholds.get("action_critical", "Seek immediate medical attention")),
                urgency="CRITICAL",
                threshold=adjusted_thresholds["critical_high"],
                test_name=test_name
            )
        
        # Check for severe values
        if "severe_low" in adjusted_thresholds and value <= adjusted_thresholds["severe_low"]:
            return EmergencyFlag(
                condition=f"Severe Low {test_name.replace('_', ' ').title()}",
                value=f"{test_name.replace('_', ' ').title()}: {value}",
                action=adjusted_thresholds.get("action_severe_low", adjusted_thresholds.get("action_severe", "Urgent medical evaluation required")),
                urgency="HIGH",
                threshold=adjusted_thresholds["severe_low"],
                test_name=test_name
            )
        
        if "severe_high" in adjusted_thresholds and value >= adjusted_thresholds["severe_high"]:
            return EmergencyFlag(
                condition=f"Severe High {test_name.replace('_', ' ').title()}",
                value=f"{test_name.replace('_', ' ').title()}: {value}",
                action=adjusted_thresholds.get("action_severe_high", adjusted_thresholds.get("action_severe", "Urgent medical evaluation required")),
                urgency="HIGH",
                threshold=adjusted_thresholds["severe_high"],
                test_name=test_name
            )
        
        return None
    
    def _adjust_thresholds_for_demographics(self, thresholds: Dict, age: int, gender: str, test_name: str) -> Dict:
        """Adjust thresholds based on patient demographics"""
        adjusted = thresholds.copy()
        
        # Age-based adjustments
        if test_name == "hemoglobin":
            if age > 65:
                # Elderly patients may have slightly lower acceptable hemoglobin
                if "severe_low" in adjusted:
                    adjusted["severe_low"] = max(7.5, adjusted["severe_low"] - 0.5)
        
        elif test_name == "creatinine":
            if age > 70:
                # Elderly may have higher baseline creatinine
                if "severe_high" in adjusted:
                    adjusted["severe_high"] = adjusted["severe_high"] + 0.3
        
        # Gender-based adjustments
        if test_name == "hemoglobin":
            if gender.lower() == "female":
                # Women typically have lower hemoglobin
                if "severe_low" in adjusted:
                    adjusted["severe_low"] = max(7.0, adjusted["severe_low"] - 0.5)
        
        return adjusted
    
    def _check_ml_emergencies(self, patient_data: PatientData) -> List[EmergencyFlag]:
        """Use ML models to detect emergency conditions"""
        ml_flags = []
        
        try:
            features = self._prepare_ml_features(patient_data)
            if features is None:
                return ml_flags
            
            features_scaled = self.scaler.transform(features)
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
            
            if is_anomaly:
                ml_flags.append(EmergencyFlag(
                    condition="ML Anomaly Detection Alert",
                    value=f"Anomaly Score: {anomaly_score:.3f}",
                    action="Review all test results - unusual pattern detected by AI",
                    urgency="HIGH",
                    confidence=abs(anomaly_score),
                    ml_prediction=True
                ))
            
            # Emergency classification
            emergency_prob = self.emergency_classifier.predict_proba(features_scaled)[0][1]
            emergency_prediction = self.emergency_classifier.predict(features_scaled)[0]
            
            if emergency_prediction == 1 and emergency_prob > 0.7:
                ml_flags.append(EmergencyFlag(
                    condition="ML Emergency Pattern Alert",
                    value=f"Emergency Probability: {emergency_prob:.1%}",
                    action="Urgent medical evaluation recommended based on AI pattern analysis",
                    urgency="CRITICAL" if emergency_prob > 0.9 else "HIGH",
                    confidence=emergency_prob,
                    ml_prediction=True
                ))
            
        except Exception as e:
            logger.error(f"Error in ML emergency checking: {e}")
        
        return ml_flags
    
    def check_combination_emergencies(self, patient_data: PatientData) -> List[EmergencyFlag]:
        """Check for emergency conditions that arise from combinations of test results."""
        combination_flags = []
        test_dict = {test["test_name"].lower().replace(" ", "_"): test["value"] 
                    for test in patient_data.test_results}
        
        # Existing combination checks with improvements
        if ("hemoglobin" in test_dict and test_dict["hemoglobin"] < 8.0 and 
            "platelets" in test_dict and test_dict["platelets"] < 50):
            combination_flags.append(EmergencyFlag(
                condition="Severe Anemia with Thrombocytopenia",
                value=f"Hemoglobin: {test_dict['hemoglobin']} g/dL, Platelets: {test_dict['platelets']} K/μL",
                action="Emergency hematology consultation - high bleeding risk",
                urgency="CRITICAL"
            ))
        
        # Enhanced infection indicators
        infection_indicators = 0
        infection_details = []
        
        # Normalize WBC count if in cells/cu.mm format (values > 100 suggest cells/cu.mm)
        wbc_normalized = test_dict.get("wbc_count", 0)
        if wbc_normalized > 100:  # Convert cells/cu.mm to K/μL
            wbc_normalized = wbc_normalized / 1000.0
        
        if "wbc_count" in test_dict and wbc_normalized > 15:
            infection_indicators += 1
            infection_details.append(f"WBC: {wbc_normalized:.2f} K/μL")
        
        if "esr" in test_dict and test_dict["esr"] > 50:
            infection_indicators += 1
            infection_details.append(f"ESR: {test_dict['esr']} mm/hr")
        
        # Add fever proxy (if available in future)
        if patient_data.age > 65 and infection_indicators >= 1:
            # Elderly patients need lower threshold
            combination_flags.append(EmergencyFlag(
                condition="Possible Severe Infection (Elderly Patient)",
                value=", ".join(infection_details),
                action="Emergency evaluation for infection in elderly patient",
                urgency="CRITICAL"
            ))
        elif infection_indicators >= 2:
            combination_flags.append(EmergencyFlag(
                condition="Severe Infection Indicators",
                value=", ".join(infection_details),
                action="Emergency evaluation for severe infection",
                urgency="CRITICAL"
            ))
        
        # Kidney dysfunction with electrolyte imbalance
        kidney_risk = 0
        kidney_details = []
        
        if "creatinine" in test_dict and test_dict["creatinine"] > 2.0:
            kidney_risk += 2
            kidney_details.append(f"Creatinine: {test_dict['creatinine']} mg/dL")
        
        if "sodium" in test_dict and (test_dict["sodium"] < 130 or test_dict["sodium"] > 150):
            kidney_risk += 1
            kidney_details.append(f"Sodium: {test_dict['sodium']} mmol/L")
        
        if "potassium" in test_dict and (test_dict["potassium"] < 3.0 or test_dict["potassium"] > 5.5):
            kidney_risk += 1
            kidney_details.append(f"Potassium: {test_dict['potassium']} mmol/L")
        
        if kidney_risk >= 3:
            combination_flags.append(EmergencyFlag(
                condition="Severe Kidney Dysfunction with Electrolyte Imbalance",
                value=", ".join(kidney_details),
                action="Emergency nephrology consultation - risk of kidney failure",
                urgency="CRITICAL"
            ))
        
        # Diabetic emergency patterns
        if ("glucose_fasting" in test_dict and test_dict["glucose_fasting"] > 250 and
            "sodium" in test_dict and test_dict["sodium"] > 145):
            combination_flags.append(EmergencyFlag(
                condition="Possible Diabetic Ketoacidosis",
                value=f"Glucose: {test_dict['glucose_fasting']} mg/dL, Sodium: {test_dict['sodium']} mmol/L",
                action="Emergency evaluation for diabetic ketoacidosis",
                urgency="CRITICAL"
            ))
        
        return combination_flags
    
    def _deduplicate_flags(self, flags: List[EmergencyFlag]) -> List[EmergencyFlag]:
        """Remove duplicate emergency flags"""
        seen_conditions = set()
        unique_flags = []
        
        for flag in flags:
            if flag.condition not in seen_conditions:
                seen_conditions.add(flag.condition)
                unique_flags.append(flag)
        
        return unique_flags
    
    def emergency_flags_to_dict(self, emergency_flags: List[EmergencyFlag]) -> List[Dict]:
        """Convert emergency flags to dictionary format for JSON serialization."""
        return [
            {
                "condition": flag.condition,
                "value": flag.value,
                "action": flag.action,
                "urgency": flag.urgency,
                "threshold": flag.threshold,
                "test_name": flag.test_name,
                "confidence": flag.confidence,
                "ml_prediction": flag.ml_prediction
            }
            for flag in emergency_flags
        ]
    
    def get_emergency_summary(self, emergency_flags: List[EmergencyFlag]) -> Dict:
        """Generate comprehensive summary of emergency conditions."""
        if not emergency_flags:
            return {
                "has_emergencies": False,
                "critical_count": 0,
                "high_count": 0,
                "ml_predictions": 0,
                "summary": "No emergency conditions detected"
            }
        
        critical_flags = [flag for flag in emergency_flags if flag.urgency == "CRITICAL"]
        high_flags = [flag for flag in emergency_flags if flag.urgency == "HIGH"]
        ml_flags = [flag for flag in emergency_flags if flag.ml_prediction]
        
        return {
            "has_emergencies": True,
            "critical_count": len(critical_flags),
            "high_count": len(high_flags),
            "total_count": len(emergency_flags),
            "ml_predictions": len(ml_flags),
            "summary": f"Detected {len(critical_flags)} critical and {len(high_flags)} high urgency conditions",
            "critical_conditions": [flag.condition for flag in critical_flags],
            "high_conditions": [flag.condition for flag in high_flags],
            "ml_conditions": [flag.condition for flag in ml_flags],
            "max_confidence": max([flag.confidence for flag in emergency_flags if flag.confidence], default=0)
        }
    
    def save_model(self, filepath: str):
        """Save trained ML models"""
        if self.is_trained:
            model_data = {
                'scaler': self.scaler,
                'anomaly_detector': self.anomaly_detector,
                'emergency_classifier': self.emergency_classifier,
                'thresholds': self.emergency_thresholds
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pre-trained ML models"""
        try:
            model_data = joblib.load(filepath)
            self.scaler = model_data['scaler']
            self.anomaly_detector = model_data['anomaly_detector'] 
            self.emergency_classifier = model_data['emergency_classifier']
            self.emergency_thresholds = model_data['thresholds']
            self.is_trained = True
            self.enable_ml = True
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.enable_ml = False


# Example usage and testing
if __name__ == "__main__":
    # Initialize emergency checker
    checker = EmergencyChecker(enable_ml=True)
    
    # Example patient data
    test_results = [
        {"test_name": "hemoglobin", "value": 6.5},
        {"test_name": "glucose_fasting", "value": 95},
        {"test_name": "sodium", "value": 142},
        {"test_name": "potassium", "value": 4.1},
        {"test_name": "creatinine", "value": 1.1},
        {"test_name": "wbc_count", "value": 8.5},
        {"test_name": "platelets", "value": 45},
        {"test_name": "esr", "value": 25}
    ]
    
    patient = PatientData(
        age=65,
        gender="female",
        test_results=test_results,
        medical_history=["diabetes", "hypertension"],
        medications=["metformin", "lisinopril"]
    )
    
    # Check for emergencies
    flags = checker.check_emergency_flags(patient)
    summary = checker.get_emergency_summary(flags)
    
    print("Emergency Analysis Results:")
    print("=" * 50)
    print(f"Summary: {summary['summary']}")
    print(f"Total emergencies: {summary['total_count']}")
    print(f"Critical: {summary['critical_count']}, High: {summary['high_count']}")
    print(f"ML predictions: {summary['ml_predictions']}")