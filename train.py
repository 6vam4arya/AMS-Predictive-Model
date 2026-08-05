"""
Training Pipeline for AMS Prediction Model.

Orchestrates the full training workflow:
1. Data generation (if needed)
2. Preprocessing
3. Feature engineering
4. Model training (XGBoost, LightGBM, RF, LSTM, Ensemble)
5. Evaluation
6. Model saving
"""

import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

from config import DATA_CONFIG, TRAINING_CONFIG
from data_generator import AMSDataGenerator
from preprocessing import AMSDataPreprocessor
from feature_engineering import AMSFeatureEngineer
from models import (
    XGBoostAMSModel,
    LightGBMAMSModel,
    RandomForestAMSModel,
    LSTMAMSModel,
    StackingEnsemble,
    save_models,
)
from evaluate import AMSModelEvaluator


def ensure_data_exists():
    """Generate synthetic data if it doesn't exist."""
    data_dir = DATA_CONFIG["output_dir"]
    if not os.path.exists(os.path.join(data_dir, "clinical_demographics.csv")):
        print("Data not found. Generating synthetic dataset...")
        generator = AMSDataGenerator()
        generator.generate_full_dataset()
    else:
        print("✓ Data files found.")


def run_training_pipeline():
    """Execute the complete training pipeline."""
    print("=" * 70)
    print("  AMS PREDICTION MODEL - TRAINING PIPELINE")
    print("  Longitudinal Multi-Modal Machine Learning")
    print("=" * 70)

    # Step 1: Ensure data exists
    print("\n" + "─" * 70)
    print("STEP 1: Data Preparation")
    print("─" * 70)
    ensure_data_exists()

    # Step 2: Preprocessing
    print("\n" + "─" * 70)
    print("STEP 2: Data Preprocessing")
    print("─" * 70)
    preprocessor = AMSDataPreprocessor()
    preprocessor.load_data()
    preprocessor.merge_modalities()
    preprocessor.merged_df = preprocessor.handle_missing_values(preprocessor.merged_df)

    # Step 3: Feature Engineering
    print("\n" + "─" * 70)
    print("STEP 3: Feature Engineering")
    print("─" * 70)
    fe = AMSFeatureEngineer()
    engineered_features = fe.engineer_all_features(preprocessor.merged_df)

    # Create base feature matrix
    X_wide, y = preprocessor.create_feature_matrix()

    # Combine base features with engineered features
    # Align indices
    common_idx = X_wide.index.intersection(engineered_features.index)
    X_combined = pd.concat([
        X_wide.loc[common_idx],
        engineered_features.loc[common_idx],
    ], axis=1)
    y_aligned = y.loc[common_idx]

    print(f"\nCombined feature matrix: {X_combined.shape}")
    print(f"Target distribution: {y_aligned.value_counts().to_dict()}")

    # Normalize
    X_normalized = preprocessor.normalize_features(X_combined.copy(), fit=True)

    # Step 4: Create sequential data for LSTM
    print("\n" + "─" * 70)
    print("STEP 4: Preparing Sequential Data for LSTM")
    print("─" * 70)
    X_seq, y_seq = preprocessor.create_sequential_data()

    # Step 5: Split data
    print("\n" + "─" * 70)
    print("STEP 5: Data Splitting")
    print("─" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_normalized, y_aligned
    )

    # Split sequential data with same indices
    from sklearn.model_selection import train_test_split

    test_size = TRAINING_CONFIG["test_size"]
    val_size = TRAINING_CONFIG["val_size"]
    seed = TRAINING_CONFIG["random_seed"]

    # Align sequential data with participant IDs
    seq_idx = np.arange(len(y_seq))
    seq_trainval, seq_test, y_seq_trainval, y_seq_test = train_test_split(
        seq_idx, y_seq, test_size=test_size, random_state=seed, stratify=y_seq
    )
    val_ratio = val_size / (1 - test_size)
    seq_train, seq_val, y_seq_train, y_seq_val = train_test_split(
        seq_trainval, y_seq_trainval, test_size=val_ratio,
        random_state=seed, stratify=y_seq_trainval
    )

    X_seq_train = X_seq[seq_train]
    X_seq_val = X_seq[seq_val]
    X_seq_test = X_seq[seq_test]

    # Step 6: Handle class imbalance
    print("\n" + "─" * 70)
    print("STEP 6: Handling Class Imbalance (SMOTE)")
    print("─" * 70)
    X_train_resampled, y_train_resampled = preprocessor.apply_smote(
        X_train, y_train
    )

    # Step 7: Train Models
    print("\n" + "─" * 70)
    print("STEP 7: Model Training")
    print("─" * 70)

    feature_names = list(X_combined.columns)
    trained_models = {}

    # 7a. XGBoost
    print("\n[7a] Training XGBoost...")
    xgb_model = XGBoostAMSModel()
    xgb_model.fit(X_train_resampled, y_train_resampled, X_val, y_val)
    trained_models["xgboost"] = xgb_model
    xgb_val_auc = _quick_auc(xgb_model, X_val, y_val)
    print(f"     XGBoost Validation AUC: {xgb_val_auc:.4f}")

    # 7b. LightGBM
    print("\n[7b] Training LightGBM...")
    lgbm_model = LightGBMAMSModel()
    lgbm_model.fit(X_train_resampled, y_train_resampled, X_val, y_val)
    trained_models["lightgbm"] = lgbm_model
    lgbm_val_auc = _quick_auc(lgbm_model, X_val, y_val)
    print(f"     LightGBM Validation AUC: {lgbm_val_auc:.4f}")

    # 7c. Random Forest
    print("\n[7c] Training Random Forest...")
    rf_model = RandomForestAMSModel()
    rf_model.fit(X_train_resampled, y_train_resampled)
    trained_models["random_forest"] = rf_model
    rf_val_auc = _quick_auc(rf_model, X_val, y_val)
    print(f"     Random Forest Validation AUC: {rf_val_auc:.4f}")

    # 7d. LSTM
    print("\n[7d] Training LSTM...")
    lstm_model = LSTMAMSModel()
    lstm_model.fit(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
    trained_models["lstm"] = lstm_model

    if lstm_model.model is not None:
        lstm_val_preds = lstm_model.predict_proba(X_seq_val)
        from sklearn.metrics import roc_auc_score
        lstm_val_auc = roc_auc_score(y_seq_val, lstm_val_preds)
        print(f"     LSTM Validation AUC: {lstm_val_auc:.4f}")
    else:
        print("     LSTM skipped (TensorFlow not available)")

    # 7e. Stacking Ensemble
    print("\n[7e] Training Stacking Ensemble...")
    base_models = [
        ("xgboost", xgb_model),
        ("lightgbm", lgbm_model),
        ("random_forest", rf_model),
    ]

    # Add LSTM if available
    if lstm_model.model is not None:
        base_models.append(("lstm", lstm_model))

    ensemble = StackingEnsemble(base_models=base_models)
    ensemble.fit(
        X_train_resampled, y_train_resampled,
        X_seq=X_seq_train if lstm_model.model is not None else None,
        X_val=X_val, y_val=y_val,
        X_val_seq=X_seq_val if lstm_model.model is not None else None,
    )
    trained_models["ensemble"] = ensemble

    ensemble_val_auc = _quick_auc_ensemble(
        ensemble, X_val, y_val,
        X_seq_val if lstm_model.model is not None else None
    )
    print(f"     Ensemble Validation AUC: {ensemble_val_auc:.4f}")

    # Step 8: Evaluation
    print("\n" + "─" * 70)
    print("STEP 8: Model Evaluation")
    print("─" * 70)

    evaluator = AMSModelEvaluator(feature_names=feature_names)

    print("\n--- Test Set Performance ---")
    for name, model in trained_models.items():
        if name == "ensemble":
            X_seq_for_eval = X_seq_test if lstm_model.model is not None else None
            y_pred = ensemble.predict(X_test, X_seq_for_eval)
            y_proba = ensemble.predict_proba(X_test, X_seq_for_eval)
        elif name == "lstm":
            y_pred = model.predict(X_seq_test)
            y_proba = model.predict_proba(X_seq_test)
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

        # Use appropriate y_test for LSTM
        if name == "lstm":
            metrics = evaluator.compute_metrics(y_seq_test, y_pred, y_proba)
        else:
            metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)

        print(f"\n  {name.upper()}:")
        print(f"    AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"    Specificity: {metrics['specificity']:.4f}")
        print(f"    F1-Score: {metrics['f1']:.4f}")
        print(f"    PPV: {metrics['ppv']:.4f}")
        print(f"    NPV: {metrics['npv']:.4f}")

    # Step 9: Feature Importance Analysis
    print("\n" + "─" * 70)
    print("STEP 9: Feature Importance Analysis")
    print("─" * 70)

    # XGBoost feature importance
    xgb_importance = xgb_model.get_feature_importance(feature_names)
    print("\nTop 20 Features (XGBoost):")
    print(xgb_importance.head(20).to_string(index=False))

    # Step 10: Save models
    print("\n" + "─" * 70)
    print("STEP 10: Saving Models")
    print("─" * 70)
    save_models(trained_models)
    preprocessor.save_preprocessor()

    # Generate evaluation report
    print("\n" + "─" * 70)
    print("STEP 11: Generating Evaluation Report")
    print("─" * 70)
    evaluator.generate_report(
        trained_models, X_test, y_test,
        X_seq_test=X_seq_test if lstm_model.model is not None else None,
        y_seq_test=y_seq_test,
    )

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Best model: Ensemble (AUC: {ensemble_val_auc:.4f})")
    print(f"  Models saved to: {TRAINING_CONFIG['model_save_dir']}/")
    print(f"  Results saved to: {TRAINING_CONFIG['results_dir']}/")

    return trained_models, evaluator


def _quick_auc(model, X, y):
    """Quick AUC computation for validation monitoring."""
    from sklearn.metrics import roc_auc_score
    try:
        y_proba = model.predict_proba(X)
        return roc_auc_score(y, y_proba)
    except Exception:
        return 0.0


def _quick_auc_ensemble(ensemble, X, y, X_seq=None):
    """Quick AUC for ensemble model."""
    from sklearn.metrics import roc_auc_score
    try:
        y_proba = ensemble.predict_proba(X, X_seq)
        return roc_auc_score(y, y_proba)
    except Exception:
        return 0.0


if __name__ == "__main__":
    trained_models, evaluator = run_training_pipeline()