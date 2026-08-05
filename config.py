"""
Configuration file for AMS Prediction Model.
Contains hyperparameters, data settings, and model configurations.
"""

import os

# ==============================================================================
# Data Configuration
# ==============================================================================
DATA_CONFIG = {
    "n_participants": 500,
    "time_points": [1, 7, 14],  # Days of measurement
    "ams_prevalence": 0.25,  # ~25% develop AMS (consistent with literature)
    "random_seed": 42,
    "output_dir": os.path.join(os.path.dirname(__file__), "data"),
}

# ==============================================================================
# Physiological Features (based on published AMS research)
# ==============================================================================
PHYSIOLOGICAL_FEATURES = {
    "spo2": {"baseline_mean": 97.0, "baseline_std": 1.5, "unit": "%"},
    "heart_rate": {"baseline_mean": 72.0, "baseline_std": 10.0, "unit": "bpm"},
    "systolic_bp": {"baseline_mean": 120.0, "baseline_std": 12.0, "unit": "mmHg"},
    "diastolic_bp": {"baseline_mean": 78.0, "baseline_std": 8.0, "unit": "mmHg"},
    "respiratory_rate": {"baseline_mean": 16.0, "baseline_std": 3.0, "unit": "breaths/min"},
    "body_temperature": {"baseline_mean": 36.8, "baseline_std": 0.3, "unit": "°C"},
}

# ==============================================================================
# Transcriptomic Features (Hypoxia-related genes from literature)
# ==============================================================================
GENE_FEATURES = [
    # HIF pathway genes
    "HIF1A", "HIF2A", "EPAS1", "VHL", "PHD2", "FIH1",
    # Erythropoiesis
    "EPO", "EPOR", "GATA1", "KLF1",
    # Vascular/angiogenesis
    "VEGFA", "VEGFR2", "ANG1", "TIE2", "NOS3",
    # Inflammation
    "IL6", "IL1B", "TNF", "CXCL8", "CRP",
    # Oxidative stress
    "SOD2", "CAT", "GPX1", "NRF2", "HMOX1",
    # Renin-angiotensin system
    "ACE", "ACE2", "AGT", "AGTR1",
    # Energy metabolism
    "LDHA", "PDK1", "GLUT1", "PGK1", "PKM2",
]

# ==============================================================================
# Clinical/Demographic Features
# ==============================================================================
CLINICAL_FEATURES = {
    "age": {"min": 18, "max": 65, "mean": 35, "std": 10},
    "sex": {"categories": [0, 1], "probs": [0.5, 0.5]},  # 0=Female, 1=Male
    "bmi": {"mean": 24.0, "std": 3.5, "min": 18.0, "max": 35.0},
    "altitude_experience": {"categories": [0, 1, 2], "probs": [0.4, 0.35, 0.25]},
    "smoking_status": {"categories": [0, 1], "probs": [0.75, 0.25]},
    "fitness_level": {"categories": [1, 2, 3, 4, 5], "probs": [0.1, 0.2, 0.35, 0.25, 0.1]},
    "prior_ams_history": {"categories": [0, 1], "probs": [0.8, 0.2]},
    "ascent_rate": {"mean": 500, "std": 150, "min": 200, "max": 1000},  # meters/day
    "target_altitude": {"mean": 4500, "std": 800, "min": 3000, "max": 6500},  # meters
}

# ==============================================================================
# Lake Louise Score Configuration
# ==============================================================================
LLS_CONFIG = {
    "max_score": 12,
    "ams_threshold": 3,  # LLS >= 3 with headache = AMS diagnosis
    "components": ["headache", "gi_symptoms", "fatigue", "dizziness"],
}

# ==============================================================================
# Model Hyperparameters
# ==============================================================================
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 3.0,  # Handle class imbalance
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "use_label_encoder": False,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
}

LSTM_PARAMS = {
    "hidden_units": 64,
    "dropout_rate": 0.3,
    "recurrent_dropout": 0.2,
    "dense_units": 32,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "patience": 15,  # Early stopping patience
}

ENSEMBLE_PARAMS = {
    "meta_learner": "logistic_regression",
    "cv_folds": 5,
    "calibration_method": "isotonic",
}

# ==============================================================================
# Training Configuration
# ==============================================================================
TRAINING_CONFIG = {
    "test_size": 0.2,
    "val_size": 0.15,
    "cv_folds": 5,
    "random_seed": 42,
    "early_stopping_rounds": 20,
    "model_save_dir": os.path.join(os.path.dirname(__file__), "models"),
    "results_dir": os.path.join(os.path.dirname(__file__), "results"),
}