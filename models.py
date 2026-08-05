"""
Model Architectures for AMS Prediction.

Implements:
1. XGBoost - Gradient boosted trees for tabular features
2. Random Forest - For gene expression analysis
3. LSTM - Recurrent network for temporal patterns
4. Stacking Ensemble - Meta-learner combining all models
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

from config import (
    XGBOOST_PARAMS,
    RANDOM_FOREST_PARAMS,
    LSTM_PARAMS,
    ENSEMBLE_PARAMS,
    TRAINING_CONFIG,
)


class XGBoostAMSModel:
    """XGBoost model for AMS prediction using tabular features."""

    def __init__(self, params=None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_importance = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model with optional early stopping."""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.feature_importance = self.model.feature_importances_
        return self

    def predict(self, X):
        """Predict AMS labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict AMS probability."""
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names=None):
        """Get feature importance ranking."""
        if self.feature_importance is None:
            return None

        importance_df = pd.DataFrame({
            "feature": feature_names or range(len(self.feature_importance)),
            "importance": self.feature_importance,
        }).sort_values("importance", ascending=False)

        return importance_df


class LightGBMAMSModel:
    """LightGBM model as an alternative gradient boosting approach."""

    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model."""
        callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            self.model.fit(X_train, y_train)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class RandomForestAMSModel:
    """Random Forest model, particularly useful for gene expression features."""

    def __init__(self, params=None):
        self.params = params or RANDOM_FOREST_PARAMS.copy()
        self.model = RandomForestClassifier(**self.params)

    def fit(self, X_train, y_train, **kwargs):
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, feature_names=None):
        importance_df = pd.DataFrame({
            "feature": feature_names or range(len(self.model.feature_importances_)),
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance_df


class LSTMAMSModel:
    """LSTM model for capturing temporal patterns in longitudinal data."""

    def __init__(self, input_shape=None, params=None):
        self.params = params or LSTM_PARAMS.copy()
        self.input_shape = input_shape
        self.model = None
        self.history = None

    def _build_model(self, input_shape):
        """Build the LSTM architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                LSTM, Dense, Dropout, BatchNormalization, Bidirectional
            )
            from tensorflow.keras.optimizers import Adam

            model = Sequential([
                Bidirectional(
                    LSTM(
                        self.params["hidden_units"],
                        return_sequences=True,
                        dropout=self.params["dropout_rate"],
                        recurrent_dropout=self.params["recurrent_dropout"],
                    ),
                    input_shape=input_shape,
                ),
                BatchNormalization(),
                LSTM(
                    self.params["hidden_units"] // 2,
                    dropout=self.params["dropout_rate"],
                    recurrent_dropout=self.params["recurrent_dropout"],
                ),
                BatchNormalization(),
                Dense(self.params["dense_units"], activation="relu"),
                Dropout(self.params["dropout_rate"]),
                Dense(16, activation="relu"),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ])

            model.compile(
                optimizer=Adam(learning_rate=self.params["learning_rate"]),
                loss="binary_crossentropy",
                metrics=["AUC", "Precision", "Recall"],
            )

            return model

        except ImportError:
            print("WARNING: TensorFlow not available. LSTM model will be skipped.")
            return None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model."""
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.model = self._build_model(input_shape)

            if self.model is None:
                return self

            callbacks = [
                EarlyStopping(
                    monitor="val_loss" if X_val is not None else "loss",
                    patience=self.params["patience"],
                    restore_best_weights=True,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss" if X_val is not None else "loss",
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                ),
            ]

            validation_data = (X_val, y_val) if X_val is not None else None

            # Handle class imbalance with sample weights
            class_weight = {0: 1.0, 1: 3.0}

            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.params["epochs"],
                batch_size=self.params["batch_size"],
                validation_data=validation_data,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=0,
            )

        except ImportError:
            print("WARNING: TensorFlow not available. LSTM model skipped.")

        return self

    def predict(self, X):
        if self.model is None:
            return np.zeros(X.shape[0])
        proba = self.model.predict(X, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        if self.model is None:
            return np.full(X.shape[0], 0.5)
        return self.model.predict(X, verbose=0).flatten()


class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Stacking ensemble that combines predictions from multiple base models.
    Uses a meta-learner (logistic regression) to combine base model outputs.
    """

    def __init__(self, base_models=None, meta_learner=None, use_calibration=True):
        self.base_models = base_models or []
        self.meta_learner = meta_learner or LogisticRegression(
            C=1.0, random_state=42, max_iter=1000
        )
        self.use_calibration = use_calibration
        self.calibrated_meta = None

    def _get_base_predictions(self, X, X_seq=None):
        """Get probability predictions from all base models."""
        predictions = []

        for model_name, model in self.base_models:
            if model_name == "lstm" and X_seq is not None:
                pred = model.predict_proba(X_seq)
            else:
                pred = model.predict_proba(X)
            predictions.append(pred.reshape(-1, 1))

        return np.hstack(predictions)

    def fit(self, X, y, X_seq=None, X_val=None, y_val=None, X_val_seq=None):
        """
        Train the stacking ensemble.
        First trains base models, then trains meta-learner on their predictions.
        """
        print("\nTraining Stacking Ensemble...")

        # Get base model predictions on training data
        base_preds = self._get_base_predictions(X, X_seq)

        # Train meta-learner
        self.meta_learner.fit(base_preds, y)

        # Calibrate if requested
        if self.use_calibration and X_val is not None:
            val_preds = self._get_base_predictions(X_val, X_val_seq)
            self.calibrated_meta = CalibratedClassifierCV(
                self.meta_learner, method="isotonic", cv="prefit"
            )
            self.calibrated_meta.fit(val_preds, y_val)

        print("  ✓ Meta-learner trained")
        return self

    def predict(self, X, X_seq=None):
        """Predict AMS labels."""
        proba = self.predict_proba(X, X_seq)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X, X_seq=None):
        """Predict AMS probability using calibrated ensemble."""
        base_preds = self._get_base_predictions(X, X_seq)

        if self.calibrated_meta is not None:
            return self.calibrated_meta.predict_proba(base_preds)[:, 1]
        else:
            return self.meta_learner.predict_proba(base_preds)[:, 1]

    def get_model_weights(self):
        """Get the learned weights for each base model."""
        if hasattr(self.meta_learner, "coef_"):
            weights = self.meta_learner.coef_[0]
            model_names = [name for name, _ in self.base_models]
            return dict(zip(model_names, weights))
        return None


def save_models(models_dict, save_dir=None):
    """Save all trained models to disk."""
    save_dir = save_dir or TRAINING_CONFIG["model_save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models_dict.items():
        if name == "lstm":
            # Save Keras model separately
            try:
                if model.model is not None:
                    model.model.save(os.path.join(save_dir, "lstm_model.keras"))
            except Exception as e:
                print(f"  Warning: Could not save LSTM model: {e}")
        else:
            joblib.dump(model, os.path.join(save_dir, f"{name}_model.joblib"))

    print(f"\n✓ Models saved to: {save_dir}/")


def load_models(save_dir=None):
    """Load all trained models from disk."""
    save_dir = save_dir or TRAINING_CONFIG["model_save_dir"]
    models = {}

    for f in os.listdir(save_dir):
        if f.endswith("_model.joblib"):
            name = f.replace("_model.joblib", "")
            models[name] = joblib.load(os.path.join(save_dir, f))

    # Load LSTM if available
    lstm_path = os.path.join(save_dir, "lstm_model.keras")
    if os.path.exists(lstm_path):
        try:
            import tensorflow as tf
            lstm_model = LSTMAMSModel()
            lstm_model.model = tf.keras.models.load_model(lstm_path)
            models["lstm"] = lstm_model
        except ImportError:
            print("WARNING: TensorFlow not available, LSTM model not loaded.")

    return models