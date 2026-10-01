"""
Configuration for the AMS (Acute Mountain Sickness) genetic analysis project.

Pipeline: preprocessing -> K-Means (K=4) -> top-5 genes per cluster -> Naive Bayes.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# Data Configuration
# ==============================================================================
DATA_CONFIG = {
    "n_participants": 500,           # used only by the synthetic data generator
    "time_points": [1, 7, 14],       # days of measurement
    "ams_prevalence": 0.25,
    "random_seed": 42,
    "output_dir": os.path.join(BASE_DIR, "data"),
    # Optional: if this Excel workbook exists in output_dir it is used instead of CSVs.
    "excel_file": "ams_data.xlsx",
}

# Sheet names (Excel) / file names (CSV) for each data modality
DATA_SOURCES = {
    "clinical": {"sheet": "clinical", "csv": "clinical_demographics.csv"},
    "physiological": {"sheet": "physiological", "csv": "physiological_data.csv"},
    "transcriptomic": {"sheet": "transcriptomic", "csv": "transcriptomic_data.csv"},
    "lls": {"sheet": "lls", "csv": "lake_louise_scores.csv"},
}

# ==============================================================================
# Physiological Features
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
# Transcriptomic Features (hypoxia-related genes)
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
# Clinical/Demographic Features (used only by the synthetic data generator)
# ==============================================================================
CLINICAL_FEATURES = {
    "age": {"min": 18, "max": 65, "mean": 35, "std": 10},
    "sex": {"categories": [0, 1], "probs": [0.5, 0.5]},
    "bmi": {"mean": 24.0, "std": 3.5, "min": 18.0, "max": 35.0},
    "altitude_experience": {"categories": [0, 1, 2], "probs": [0.4, 0.35, 0.25]},
    "smoking_status": {"categories": [0, 1], "probs": [0.75, 0.25]},
    "fitness_level": {"categories": [1, 2, 3, 4, 5], "probs": [0.1, 0.2, 0.35, 0.25, 0.1]},
    "prior_ams_history": {"categories": [0, 1], "probs": [0.8, 0.2]},
    "ascent_rate": {"mean": 500, "std": 150, "min": 200, "max": 1000},
    "target_altitude": {"mean": 4500, "std": 800, "min": 3000, "max": 6500},
}

LLS_CONFIG = {
    "max_score": 12,
    "ams_threshold": 3,
    "components": ["headache", "gi_symptoms", "fatigue", "dizziness"],
}

# ==============================================================================
# Stage 1: K-Means clustering
# ==============================================================================
CLUSTER_CONFIG = {
    "n_clusters": 4,
    "n_init": 20,
    "random_state": 42,
    # Participant-level features describing AMS symptoms, oxygen saturation and
    # physiological response over time.
    "features": ["ams_score_mean", "spo2_mean", "spo2_change", "heart_rate_mean"],
}

CLUSTER_NAMES = {
    1: "Low AMS and Fast Acclimatizers",
    2: "Low AMS and Slow Acclimatizers",
    3: "High AMS and Fast Acclimatizers",
    4: "High AMS and Poor Acclimatization",
}

# ==============================================================================
# Stage 2: gene selection (top genes per cluster)
# ==============================================================================
GENE_SELECTION_CONFIG = {
    "top_n_per_cluster": 5,   # 4 clusters x 5 genes = 20 genes
}

# ==============================================================================
# Stage 3: Naive Bayes + training
# ==============================================================================
NAIVE_BAYES_PARAMS = {
    "var_smoothing": 1e-9,
}

TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_seed": 42,
    "model_save_dir": os.path.join(BASE_DIR, "models"),
    "results_dir": os.path.join(BASE_DIR, "results"),
}
