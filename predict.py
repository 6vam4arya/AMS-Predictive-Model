"""
Prediction/Inference Module for AMS Prediction.

Provides functionality to:
- Load trained models
- Preprocess new participant data
- Generate AMS risk predictions
- Provide interpretable risk explanations
"""

import numpy as np
import pandas as pd
import os
import joblib
import json

from config import (
    DATA_CONFIG,
    TRAINING_CONFIG,
    PHYSIOLOGICAL_FEATURES,
    GENE_FEATURES,
    CLINICAL_FEATURES,
)
from models import load_models


class AMSPredictor:
    """
    AMS Risk Prediction System.
    Loads trained models and provides predictions for new participants.
    """

    def __init__(self, model_dir=None):
        self.model_dir = model_dir or TRAINING_CONFIG["model_save_dir"]
        self.models = None
        self.preprocessor = None
        self.risk_thresholds = {
            "low": 0.2,
            "moderate": 0.4,
            "high": 0.6,
            "very_high": 0.8,
        }

    def load(self):
        """Load trained models and preprocessor."""
        print("Loading AMS prediction models...")

        # Load models
        self.models = load_models(self.model_dir)
        if not self.models:
            raise FileNotFoundError(
                f"No trained models found in {self.model_dir}. "
                "Run train.py first."
            )

        # Load preprocessor
        preprocessor_path = os.path.join(self.model_dir, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            print("WARNING: Preprocessor not found. Using default scaling.")

        print(f"  Loaded models: {list(self.models.keys())}")
        return self

    def predict_single_participant(self, participant_data):
        """
        Predict AMS risk for a single participant.

        Args:
            participant_data: dict with keys:
                - 'clinical': dict of clinical/demographic features
                - 'physiological': list of dicts (one per time point)
                - 'transcriptomic': list of dicts (one per time point)
                - 'lls': list of dicts (one per time point)

        Returns:
            dict with risk prediction and explanation
        """
        # Prepare feature vector
        features = self._prepare_features(participant_data)

        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if name == "lstm":
                continue  # LSTM needs sequential input
            if name == "ensemble":
                continue  # Will use ensemble separately

            try:
                proba = model.predict_proba(features.values.reshape(1, -1))[0]
                predictions[name] = float(proba)
            except Exception as e:
                print(f"  Warning: {name} prediction failed: {e}")

        # Ensemble prediction (average if ensemble model not available)
        if "ensemble" in self.models:
            try:
                ensemble_proba = self.models["ensemble"].predict_proba(
                    features.values.reshape(1, -1)
                )[0]
                predictions["ensemble"] = float(ensemble_proba)
            except Exception:
                predictions["ensemble"] = np.mean(list(predictions.values()))
        else:
            predictions["ensemble"] = np.mean(list(predictions.values()))

        # Final risk score
        risk_score = predictions.get("ensemble", predictions.get("xgboost", 0.5))

        # Risk category
        risk_category = self._categorize_risk(risk_score)

        # Generate explanation
        explanation = self._generate_explanation(
            participant_data, risk_score, risk_category
        )

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "model_predictions": predictions,
            "explanation": explanation,
            "recommendation": self._get_recommendation(risk_category),
        }

    def predict_batch(self, participants_data):
        """Predict AMS risk for multiple participants."""
        results = []
        for i, pdata in enumerate(participants_data):
            print(f"  Predicting participant {i+1}/{len(participants_data)}...")
            result = self.predict_single_participant(pdata)
            result["participant_id"] = i
            results.append(result)

        return results

    def _prepare_features(self, participant_data):
        """Convert participant data dict to feature vector."""
        features = {}

        # Clinical features
        clinical = participant_data.get("clinical", {})
        for key in CLINICAL_FEATURES.keys():
            features[key] = clinical.get(key, 0)

        # Physiological features (flatten across time points)
        physio_list = participant_data.get("physiological", [])
        physio_cols = list(PHYSIOLOGICAL_FEATURES.keys())

        for t_idx, physio in enumerate(physio_list):
            day = DATA_CONFIG["time_points"][t_idx] if t_idx < len(DATA_CONFIG["time_points"]) else t_idx + 1
            for col in physio_cols:
                features[f"{col}_day{day}"] = physio.get(col, 0)

        # Transcriptomic features (flatten across time points)
        trans_list = participant_data.get("transcriptomic", [])
        for t_idx, trans in enumerate(trans_list):
            day = DATA_CONFIG["time_points"][t_idx] if t_idx < len(DATA_CONFIG["time_points"]) else t_idx + 1
            for gene in GENE_FEATURES:
                features[f"{gene}_day{day}"] = trans.get(gene, 8.0)  # Default baseline

        # LLS features (flatten across time points)
        lls_list = participant_data.get("lls", [])
        lls_components = ["lls_headache", "lls_gi_symptoms", "lls_fatigue",
                          "lls_dizziness", "lls_total"]
        for t_idx, lls in enumerate(lls_list):
            day = DATA_CONFIG["time_points"][t_idx] if t_idx < len(DATA_CONFIG["time_points"]) else t_idx + 1
            for comp in lls_components:
                features[f"{comp}_day{day}"] = lls.get(comp, 0)

        return pd.Series(features)

    def _categorize_risk(self, risk_score):
        """Categorize risk score into clinical risk levels."""
        if risk_score < self.risk_thresholds["low"]:
            return "LOW"
        elif risk_score < self.risk_thresholds["moderate"]:
            return "MODERATE"
        elif risk_score < self.risk_thresholds["high"]:
            return "HIGH"
        elif risk_score < self.risk_thresholds["very_high"]:
            return "VERY HIGH"
        else:
            return "CRITICAL"

    def _generate_explanation(self, participant_data, risk_score, risk_category):
        """Generate interpretable explanation for the prediction."""
        explanations = []

        # Check physiological risk factors
        physio_list = participant_data.get("physiological", [])
        if physio_list:
            latest_physio = physio_list[-1]

            if latest_physio.get("spo2", 100) < 90:
                explanations.append(
                    f"Low oxygen saturation (SpO2: {latest_physio['spo2']:.1f}%) "
                    "indicates significant hypoxemia"
                )
            if latest_physio.get("heart_rate", 72) > 100:
                explanations.append(
                    f"Elevated heart rate ({latest_physio['heart_rate']:.0f} bpm) "
                    "suggests cardiovascular stress"
                )
            if latest_physio.get("respiratory_rate", 16) > 22:
                explanations.append(
                    f"Elevated respiratory rate ({latest_physio['respiratory_rate']:.0f}/min) "
                    "indicates respiratory compensation"
                )

        # Check LLS progression
        lls_list = participant_data.get("lls", [])
        if len(lls_list) >= 2:
            lls_trend = lls_list[-1].get("lls_total", 0) - lls_list[0].get("lls_total", 0)
            if lls_trend > 2:
                explanations.append(
                    f"Lake Louise Score increasing (Δ={lls_trend}), "
                    "suggesting symptom progression"
                )

        # Check transcriptomic markers
        trans_list = participant_data.get("transcriptomic", [])
        if trans_list:
            latest_trans = trans_list[-1]
            if latest_trans.get("IL6", 8) > 10:
                explanations.append(
                    "Elevated IL-6 expression indicates inflammatory response"
                )
            if latest_trans.get("HIF1A", 8) > 10:
                explanations.append(
                    "Elevated HIF-1α suggests hypoxia pathway activation"
                )

        # Check clinical risk factors
        clinical = participant_data.get("clinical", {})
        if clinical.get("prior_ams_history", 0) == 1:
            explanations.append("Prior history of AMS increases recurrence risk")
        if clinical.get("ascent_rate", 500) > 700:
            explanations.append(
                f"Rapid ascent rate ({clinical['ascent_rate']:.0f} m/day) "
                "exceeds recommended guidelines"
            )

        if not explanations:
            explanations.append("No specific high-risk indicators identified")

        return explanations

    def _get_recommendation(self, risk_category):
        """Provide clinical recommendations based on risk level."""
        recommendations = {
            "LOW": (
                "Continue normal acclimatization schedule. "
                "Monitor symptoms and repeat assessment in 24-48 hours."
            ),
            "MODERATE": (
                "Consider slowing ascent rate. Increase hydration. "
                "Monitor symptoms closely. Consider prophylactic acetazolamide "
                "if ascending further."
            ),
            "HIGH": (
                "HALT further ascent. Rest at current altitude for 24-48 hours. "
                "Start acetazolamide (125-250mg BID) if not already taking. "
                "Prepare for possible descent."
            ),
            "VERY HIGH": (
                "DESCEND immediately by at least 500-1000m. "
                "Administer supplemental oxygen if available. "
                "Start dexamethasone (4mg q6h) if descent delayed. "
                "Seek medical evaluation."
            ),
            "CRITICAL": (
                "EMERGENCY: Immediate descent required. "
                "Administer supplemental oxygen and dexamethasone. "
                "Prepare for possible HACE/HAPE. "
                "Evacuate to lower altitude medical facility."
            ),
        }
        return recommendations.get(risk_category, "Consult medical professional.")


def demo_prediction():
    """Demonstrate the prediction system with sample data."""
    print("=" * 60)
    print("AMS PREDICTION SYSTEM - DEMO")
    print("=" * 60)

    # Create sample participant data
    sample_participant = {
        "clinical": {
            "age": 32,
            "sex": 1,
            "bmi": 23.5,
            "altitude_experience": 1,
            "smoking_status": 0,
            "fitness_level": 3,
            "prior_ams_history": 0,
            "ascent_rate": 600,
            "target_altitude": 5000,
        },
        "physiological": [
            # Day 1
            {"spo2": 95.0, "heart_rate": 78, "systolic_bp": 125,
             "diastolic_bp": 80, "respiratory_rate": 18, "body_temperature": 36.9},
            # Day 7
            {"spo2": 89.0, "heart_rate": 92, "systolic_bp": 132,
             "diastolic_bp": 85, "respiratory_rate": 22, "body_temperature": 37.1},
            # Day 14
            {"spo2": 85.0, "heart_rate": 105, "systolic_bp": 140,
             "diastolic_bp": 90, "respiratory_rate": 25, "body_temperature": 37.3},
        ],
        "transcriptomic": [
            # Day 1 - baseline
            {gene: np.random.normal(8.0, 1.0) for gene in GENE_FEATURES},
            # Day 7 - moderate changes
            {gene: np.random.normal(9.5, 1.5) for gene in GENE_FEATURES},
            # Day 14 - significant changes
            {gene: np.random.normal(11.0, 2.0) for gene in GENE_FEATURES},
        ],
        "lls": [
            # Day 1
            {"lls_headache": 0, "lls_gi_symptoms": 0, "lls_fatigue": 1,
             "lls_dizziness": 0, "lls_total": 1},
            # Day 7
            {"lls_headache": 1, "lls_gi_symptoms": 1, "lls_fatigue": 2,
             "lls_dizziness": 1, "lls_total": 5},
            # Day 14
            {"lls_headache": 2, "lls_gi_symptoms": 2, "lls_fatigue": 2,
             "lls_dizziness": 2, "lls_total": 8},
        ],
    }

    # Try to load trained models
    try:
        predictor = AMSPredictor()
        predictor.load()

        result = predictor.predict_single_participant(sample_participant)

        print("\n" + "─" * 60)
        print("PREDICTION RESULT")
        print("─" * 60)
        print(f"\n  Risk Score: {result['risk_score']:.3f}")
        print(f"  Risk Category: {result['risk_category']}")
        print(f"\n  Model Predictions:")
        for model, score in result["model_predictions"].items():
            print(f"    {model:15s}: {score:.4f}")
        print(f"\n  Explanation:")
        for exp in result["explanation"]:
            print(f"    • {exp}")
        print(f"\n  Recommendation:")
        print(f"    {result['recommendation']}")

    except FileNotFoundError:
        print("\n  No trained models found.")
        print("  Run 'python train.py' first to train the models.")
        print("\n  Showing demo with sample data structure:")
        print(f"\n  Sample participant data keys: {list(sample_participant.keys())}")
        print(f"  Clinical features: {list(sample_participant['clinical'].keys())}")
        print(f"  Time points: {len(sample_participant['physiological'])}")
        print(f"  Genes tracked: {len(GENE_FEATURES)}")

    # Also demonstrate batch prediction format
    print("\n" + "─" * 60)
    print("BATCH PREDICTION FORMAT")
    print("─" * 60)
    print("""
    # For batch predictions:
    participants = [participant_data_1, participant_data_2, ...]
    predictor = AMSPredictor()
    predictor.load()
    results = predictor.predict_batch(participants)
    
    # Each result contains:
    # - risk_score (float, 0-1)
    # - risk_category (str: LOW/MODERATE/HIGH/VERY HIGH/CRITICAL)
    # - model_predictions (dict of model-specific scores)
    # - explanation (list of risk factor explanations)
    # - recommendation (str: clinical recommendation)
    """)


if __name__ == "__main__":
    demo_prediction()