"""
Data Preprocessing Module for AMS Prediction.

Handles:
- Data loading and merging
- Missing value imputation
- Normalization and scaling
- Train/validation/test splitting
- Class imbalance handling
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import os
import joblib

from config import DATA_CONFIG, TRAINING_CONFIG, GENE_FEATURES


class AMSDataPreprocessor:
    """Preprocesses multi-modal longitudinal data for AMS prediction."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_CONFIG["output_dir"]
        self.physio_scaler = RobustScaler()
        self.gene_scaler = StandardScaler()
        self.clinical_scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.fitted = False

    def load_data(self):
        """Load all data modalities from CSV files."""
        print("Loading data from:", self.data_dir)

        self.clinical_df = pd.read_csv(
            os.path.join(self.data_dir, "clinical_demographics.csv")
        )
        self.physio_df = pd.read_csv(
            os.path.join(self.data_dir, "physiological_data.csv")
        )
        self.transcriptomic_df = pd.read_csv(
            os.path.join(self.data_dir, "transcriptomic_data.csv")
        )
        self.lls_df = pd.read_csv(
            os.path.join(self.data_dir, "lake_louise_scores.csv")
        )

        print(f"  Clinical: {self.clinical_df.shape}")
        print(f"  Physiological: {self.physio_df.shape}")
        print(f"  Transcriptomic: {self.transcriptomic_df.shape}")
        print(f"  LLS: {self.lls_df.shape}")

        return self

    def merge_modalities(self):
        """Merge all data modalities into a single longitudinal dataframe."""
        # Merge physiological and LLS data (both are longitudinal)
        longitudinal_df = pd.merge(
            self.physio_df,
            self.lls_df,
            on=["participant_id", "day"],
            how="outer",
        )

        # Merge transcriptomic data
        longitudinal_df = pd.merge(
            longitudinal_df,
            self.transcriptomic_df,
            on=["participant_id", "day"],
            how="outer",
        )

        # Merge clinical (static) data
        self.merged_df = pd.merge(
            longitudinal_df,
            self.clinical_df,
            on="participant_id",
            how="left",
        )

        print(f"\nMerged dataset shape: {self.merged_df.shape}")
        print(f"Columns: {list(self.merged_df.columns)[:10]}...")

        return self

    def handle_missing_values(self, df):
        """Handle missing values using KNN imputation for numerical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

        # Impute numerical columns
        if df[numeric_cols].isnull().any().any():
            df_numeric = pd.DataFrame(
                self.imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols,
                index=df.index,
            )
            df = pd.concat([df_numeric, df[non_numeric_cols]], axis=1)

        return df

    def normalize_features(self, df, fit=True):
        """Normalize different feature groups with appropriate scalers."""
        physio_cols = ["spo2", "heart_rate", "systolic_bp", "diastolic_bp",
                       "respiratory_rate", "body_temperature"]
        gene_cols = [g for g in GENE_FEATURES if g in df.columns]
        clinical_numeric = ["age", "bmi", "ascent_rate", "target_altitude"]

        if fit:
            # Fit and transform
            if physio_cols[0] in df.columns:
                df[physio_cols] = self.physio_scaler.fit_transform(df[physio_cols])
            if gene_cols:
                df[gene_cols] = self.gene_scaler.fit_transform(df[gene_cols])
            clinical_present = [c for c in clinical_numeric if c in df.columns]
            if clinical_present:
                df[clinical_present] = self.clinical_scaler.fit_transform(
                    df[clinical_present]
                )
            self.fitted = True
        else:
            # Transform only (for test data)
            if physio_cols[0] in df.columns:
                df[physio_cols] = self.physio_scaler.transform(df[physio_cols])
            if gene_cols:
                df[gene_cols] = self.gene_scaler.transform(df[gene_cols])
            clinical_present = [c for c in clinical_numeric if c in df.columns]
            if clinical_present:
                df[clinical_present] = self.clinical_scaler.transform(
                    df[clinical_present]
                )

        return df

    def create_feature_matrix(self):
        """
        Create the final feature matrix for model training.
        Pivots longitudinal data into wide format (one row per participant).
        """
        df = self.merged_df.copy()

        # Separate target variable
        labels = df.groupby("participant_id")["ams_label"].first()

        # Features to pivot (longitudinal)
        physio_cols = ["spo2", "heart_rate", "systolic_bp", "diastolic_bp",
                       "respiratory_rate", "body_temperature"]
        lls_cols = ["lls_headache", "lls_gi_symptoms", "lls_fatigue",
                    "lls_dizziness", "lls_total"]
        gene_cols = [g for g in GENE_FEATURES if g in df.columns]

        longitudinal_cols = physio_cols + lls_cols + gene_cols

        # Pivot longitudinal data to wide format
        pivot_dfs = []
        for day in DATA_CONFIG["time_points"]:
            day_data = df[df["day"] == day][["participant_id"] + longitudinal_cols].copy()
            day_data.columns = ["participant_id"] + [
                f"{col}_day{day}" for col in longitudinal_cols
            ]
            pivot_dfs.append(day_data)

        # Merge all time points
        wide_df = pivot_dfs[0]
        for pdf in pivot_dfs[1:]:
            wide_df = pd.merge(wide_df, pdf, on="participant_id", how="outer")

        # Add static clinical features
        clinical_cols = ["age", "sex", "bmi", "altitude_experience",
                         "smoking_status", "fitness_level", "prior_ams_history",
                         "ascent_rate", "target_altitude"]
        clinical_static = df.groupby("participant_id")[clinical_cols].first().reset_index()
        wide_df = pd.merge(wide_df, clinical_static, on="participant_id", how="left")

        # Align with labels
        wide_df = wide_df.set_index("participant_id")
        labels = labels.loc[wide_df.index]

        print(f"\nFinal feature matrix: {wide_df.shape}")
        print(f"Target distribution: {labels.value_counts().to_dict()}")

        self.feature_matrix = wide_df
        self.labels = labels

        return wide_df, labels

    def create_sequential_data(self):
        """
        Create 3D sequential data for LSTM model.
        Shape: (n_samples, n_timesteps, n_features)
        """
        df = self.merged_df.copy()

        physio_cols = ["spo2", "heart_rate", "systolic_bp", "diastolic_bp",
                       "respiratory_rate", "body_temperature"]
        lls_cols = ["lls_headache", "lls_gi_symptoms", "lls_fatigue",
                    "lls_dizziness", "lls_total"]
        gene_cols = [g for g in GENE_FEATURES if g in df.columns]

        seq_features = physio_cols + lls_cols + gene_cols
        n_timesteps = len(DATA_CONFIG["time_points"])
        n_features = len(seq_features)

        participants = sorted(df["participant_id"].unique())
        n_samples = len(participants)

        X_seq = np.zeros((n_samples, n_timesteps, n_features))
        y_seq = np.zeros(n_samples)

        for i, pid in enumerate(participants):
            participant_data = df[df["participant_id"] == pid].sort_values("day")
            y_seq[i] = participant_data["ams_label"].iloc[0]

            for t_idx, (_, row) in enumerate(participant_data.iterrows()):
                if t_idx < n_timesteps:
                    X_seq[i, t_idx, :] = row[seq_features].values

        print(f"\nSequential data shape: {X_seq.shape}")
        print(f"  (samples={n_samples}, timesteps={n_timesteps}, features={n_features})")

        self.X_sequential = X_seq
        self.y_sequential = y_seq

        return X_seq, y_seq

    def split_data(self, X, y, test_size=None, val_size=None):
        """Split data into train, validation, and test sets with stratification."""
        test_size = test_size or TRAINING_CONFIG["test_size"]
        val_size = val_size or TRAINING_CONFIG["val_size"]
        seed = TRAINING_CONFIG["random_seed"]

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio,
            random_state=seed, stratify=y_trainval
        )

        print(f"\nData splits:")
        print(f"  Train: {X_train.shape[0]} samples (AMS: {y_train.sum():.0f})")
        print(f"  Val:   {X_val.shape[0]} samples (AMS: {y_val.sum():.0f})")
        print(f"  Test:  {X_test.shape[0]} samples (AMS: {y_test.sum():.0f})")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance in training data."""
        smote = SMOTE(random_state=TRAINING_CONFIG["random_seed"], k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"\nAfter SMOTE:")
        print(f"  Original: {len(y_train)} (AMS: {y_train.sum():.0f})")
        print(f"  Resampled: {len(y_resampled)} (AMS: {y_resampled.sum():.0f})")

        return X_resampled, y_resampled

    def get_cross_validation_folds(self, X, y, n_folds=None):
        """Get stratified cross-validation fold indices."""
        n_folds = n_folds or TRAINING_CONFIG["cv_folds"]
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=TRAINING_CONFIG["random_seed"],
        )
        return list(skf.split(X, y))

    def save_preprocessor(self, path=None):
        """Save fitted preprocessor for inference."""
        path = path or os.path.join(
            TRAINING_CONFIG["model_save_dir"], "preprocessor.joblib"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "physio_scaler": self.physio_scaler,
                "gene_scaler": self.gene_scaler,
                "clinical_scaler": self.clinical_scaler,
                "imputer": self.imputer,
            },
            path,
        )
        print(f"Preprocessor saved to: {path}")

    def preprocess_pipeline(self):
        """Run the full preprocessing pipeline."""
        print("=" * 60)
        print("AMS Data Preprocessing Pipeline")
        print("=" * 60)

        self.load_data()
        self.merge_modalities()
        self.merged_df = self.handle_missing_values(self.merged_df)

        # Create wide feature matrix (for tree-based models)
        X_wide, y = self.create_feature_matrix()

        # Normalize
        X_wide_normalized = self.normalize_features(X_wide.copy(), fit=True)

        # Create sequential data (for LSTM)
        X_seq, y_seq = self.create_sequential_data()

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X_wide_normalized, y
        )

        # Save preprocessor
        self.save_preprocessor()

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "X_sequential": X_seq,
            "y_sequential": y_seq,
            "feature_names": list(X_wide.columns),
        }


if __name__ == "__main__":
    preprocessor = AMSDataPreprocessor()
    data = preprocessor.preprocess_pipeline()
    print("\n✓ Preprocessing complete!")
    print(f"  Training features: {data['X_train'].shape}")
    print(f"  Feature names (first 10): {data['feature_names'][:10]}")