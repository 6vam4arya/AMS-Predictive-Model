"""
Evaluation Module for AMS Prediction Model.

Provides comprehensive model evaluation including:
- Classification metrics (AUC, sensitivity, specificity, F1, PPV, NPV)
- Calibration analysis
- SHAP-based feature importance and interpretability
- Cross-validation performance
- Clinical utility analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    calibration_curve,
)
from sklearn.model_selection import StratifiedKFold
import os
import json

from config import TRAINING_CONFIG, GENE_FEATURES


class AMSModelEvaluator:
    """Comprehensive evaluation for AMS prediction models."""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.results = {}

    def compute_metrics(self, y_true, y_pred, y_proba=None):
        """Compute comprehensive classification metrics."""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics["tp"] = int(tp)
        metrics["fp"] = int(fp)
        metrics["fn"] = int(fn)
        metrics["tn"] = int(tn)

        # Probability-based metrics
        if y_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
            metrics["avg_precision"] = average_precision_score(y_true, y_proba)
            metrics["brier_score"] = brier_score_loss(y_true, y_proba)
        else:
            metrics["auc_roc"] = 0.0
            metrics["avg_precision"] = 0.0
            metrics["brier_score"] = 1.0

        return metrics

    def compute_optimal_threshold(self, y_true, y_proba):
        """
        Find optimal classification threshold using Youden's J statistic.
        Balances sensitivity and specificity for clinical use.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        j_statistic = tpr - fpr
        optimal_idx = np.argmax(j_statistic)
        optimal_threshold = thresholds[optimal_idx]

        return {
            "optimal_threshold": optimal_threshold,
            "sensitivity_at_optimal": tpr[optimal_idx],
            "specificity_at_optimal": 1 - fpr[optimal_idx],
            "youden_j": j_statistic[optimal_idx],
        }

    def compute_calibration(self, y_true, y_proba, n_bins=10):
        """Assess probability calibration quality."""
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="uniform"
        )

        # Expected Calibration Error (ECE)
        bin_counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
        bin_weights = bin_counts / len(y_proba)
        ece = np.sum(
            bin_weights[:len(fraction_positive)]
            * np.abs(fraction_positive - mean_predicted)
        )

        return {
            "ece": ece,
            "fraction_positive": fraction_positive.tolist(),
            "mean_predicted": mean_predicted.tolist(),
            "brier_score": brier_score_loss(y_true, y_proba),
        }

    def compute_clinical_utility(self, y_true, y_proba, prevalence=0.25):
        """
        Compute clinical utility metrics including Net Benefit
        for decision curve analysis.
        """
        thresholds = np.arange(0.05, 0.95, 0.05)
        net_benefits = []

        n = len(y_true)

        for threshold in thresholds:
            y_pred_t = (y_proba >= threshold).astype(int)
            tp = np.sum((y_pred_t == 1) & (y_true == 1))
            fp = np.sum((y_pred_t == 1) & (y_true == 0))

            # Net benefit formula
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)

        # Treat all strategy
        treat_all_benefits = []
        for threshold in thresholds:
            nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
            treat_all_benefits.append(nb_all)

        return {
            "thresholds": thresholds.tolist(),
            "net_benefits": net_benefits,
            "treat_all_benefits": treat_all_benefits,
        }

    def compute_shap_importance(self, model, X, feature_names=None):
        """
        Compute SHAP values for model interpretability.
        Identifies which features drive AMS predictions.
        """
        try:
            import shap

            feature_names = feature_names or self.feature_names

            # Use appropriate explainer based on model type
            if hasattr(model, "model") and hasattr(model.model, "get_booster"):
                # XGBoost
                explainer = shap.TreeExplainer(model.model)
            elif hasattr(model, "model") and hasattr(model.model, "estimators_"):
                # Random Forest
                explainer = shap.TreeExplainer(model.model)
            else:
                # Fallback to KernelSHAP (slower)
                X_sample = X[:100] if len(X) > 100 else X
                explainer = shap.KernelExplainer(
                    model.predict_proba, X_sample
                )

            # Compute SHAP values
            X_explain = X[:200] if len(X) > 200 else X
            shap_values = explainer.shap_values(X_explain)

            # Handle multi-output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (AMS positive)

            # Mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                "feature": feature_names[:len(mean_shap)] if feature_names else range(len(mean_shap)),
                "mean_abs_shap": mean_shap,
            }).sort_values("mean_abs_shap", ascending=False)

            return importance_df, shap_values

        except ImportError:
            print("WARNING: SHAP not available. Skipping SHAP analysis.")
            return None, None
        except Exception as e:
            print(f"WARNING: SHAP computation failed: {e}")
            return None, None

    def cross_validate(self, model_class, model_params, X, y, n_folds=5):
        """
        Perform stratified cross-validation and report metrics.
        """
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True,
            random_state=TRAINING_CONFIG["random_seed"]
        )

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model = model_class(**model_params) if model_params else model_class()
            if hasattr(model, "fit"):
                model.fit(X_fold_train, y_fold_train)

            # Evaluate
            y_pred = model.predict(X_fold_val)
            y_proba = model.predict_proba(X_fold_val)
            metrics = self.compute_metrics(y_fold_val, y_pred, y_proba)
            metrics["fold"] = fold + 1
            fold_metrics.append(metrics)

        # Aggregate
        metrics_df = pd.DataFrame(fold_metrics)
        summary = {
            "mean_auc": metrics_df["auc_roc"].mean(),
            "std_auc": metrics_df["auc_roc"].std(),
            "mean_sensitivity": metrics_df["sensitivity"].mean(),
            "std_sensitivity": metrics_df["sensitivity"].std(),
            "mean_specificity": metrics_df["specificity"].mean(),
            "std_specificity": metrics_df["specificity"].std(),
            "mean_f1": metrics_df["f1"].mean(),
            "std_f1": metrics_df["f1"].std(),
            "fold_details": fold_metrics,
        }

        return summary

    def generate_report(self, models_dict, X_test, y_test,
                        X_seq_test=None, y_seq_test=None):
        """Generate comprehensive evaluation report."""
        results_dir = TRAINING_CONFIG["results_dir"]
        os.makedirs(results_dir, exist_ok=True)

        report = {
            "model_performance": {},
            "best_model": None,
            "best_auc": 0,
        }

        for name, model in models_dict.items():
            print(f"\n  Evaluating: {name}")

            if name == "ensemble":
                y_pred = model.predict(X_test, X_seq_test)
                y_proba = model.predict_proba(X_test, X_seq_test)
                y_true = y_test
            elif name == "lstm":
                if X_seq_test is not None:
                    y_pred = model.predict(X_seq_test)
                    y_proba = model.predict_proba(X_seq_test)
                    y_true = y_seq_test
                else:
                    continue
            else:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                y_true = y_test

            # Metrics
            metrics = self.compute_metrics(y_true, y_pred, y_proba)

            # Optimal threshold
            threshold_info = self.compute_optimal_threshold(y_true, y_proba)
            metrics.update(threshold_info)

            # Calibration
            calibration = self.compute_calibration(y_true, y_proba)
            metrics["calibration_ece"] = calibration["ece"]

            # Clinical utility
            clinical = self.compute_clinical_utility(y_true, y_proba)
            metrics["max_net_benefit"] = max(clinical["net_benefits"])

            report["model_performance"][name] = metrics

            if metrics["auc_roc"] > report["best_auc"]:
                report["best_auc"] = metrics["auc_roc"]
                report["best_model"] = name

        # SHAP analysis for best tree-based model
        best_tree_model = None
        for name in ["xgboost", "lightgbm", "random_forest"]:
            if name in models_dict:
                best_tree_model = (name, models_dict[name])
                break

        if best_tree_model:
            print(f"\n  Computing SHAP values for {best_tree_model[0]}...")
            shap_df, _ = self.compute_shap_importance(
                best_tree_model[1], X_test, self.feature_names
            )
            if shap_df is not None:
                report["top_features_shap"] = shap_df.head(30).to_dict("records")
                shap_df.to_csv(
                    os.path.join(results_dir, "shap_feature_importance.csv"),
                    index=False,
                )

        # Save report
        # Convert numpy types for JSON serialization
        report_serializable = _make_serializable(report)
        with open(os.path.join(results_dir, "evaluation_report.json"), "w") as f:
            json.dump(report_serializable, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nBest Model: {report['best_model']} (AUC: {report['best_auc']:.4f})")
        print("\nAll Models:")
        for name, metrics in report["model_performance"].items():
            print(f"  {name:15s} | AUC: {metrics['auc_roc']:.4f} | "
                  f"Sens: {metrics['sensitivity']:.4f} | "
                  f"Spec: {metrics['specificity']:.4f} | "
                  f"F1: {metrics['f1']:.4f}")

        print(f"\n✓ Report saved to: {results_dir}/evaluation_report.json")

        self.results = report
        return report


def _make_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    # Standalone evaluation (requires trained models)
    from models import load_models
    from preprocessing import AMSDataPreprocessor

    print("Loading models and data for evaluation...")

    # Load data
    preprocessor = AMSDataPreprocessor()
    data = preprocessor.preprocess_pipeline()

    # Load models
    models = load_models()

    if models:
        evaluator = AMSModelEvaluator(feature_names=data["feature_names"])
        evaluator.generate_report(
            models,
            data["X_test"],
            data["y_test"],
        )
    else:
        print("No trained models found. Run train.py first.")