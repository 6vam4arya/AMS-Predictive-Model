"""
Data Preprocessing for the AMS project.

Loads the data (Excel workbook or CSV files), merges the modalities, handles
missing/inconsistent values and builds one row per participant containing:
  - physiological summary features (used for K-Means clustering)
  - gene expression at the last time point (candidate features for Naive Bayes)
  - the AMS label
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_CONFIG, DATA_SOURCES, TRAINING_CONFIG, GENE_FEATURES,
    PHYSIOLOGICAL_FEATURES,
)

LLS_COLS = ["lls_headache", "lls_gi_symptoms", "lls_fatigue", "lls_dizziness", "lls_total"]


class AMSDataPreprocessor:
    """Preprocesses multi-modal longitudinal data for AMS analysis."""

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_CONFIG["output_dir"]
        self.gene_cols = None

    # ------------------------------------------------------------------ loading
    def load_data(self):
        """Load all modalities from an Excel workbook (if present) or CSV files."""
        xlsx_path = os.path.join(self.data_dir, DATA_CONFIG["excel_file"])
        frames = {}
        if os.path.exists(xlsx_path):
            print("Loading data from Excel workbook:", xlsx_path)
            sheets = pd.read_excel(xlsx_path, sheet_name=None)
            for key, src in DATA_SOURCES.items():
                frames[key] = sheets[src["sheet"]]
        else:
            print("Loading data from CSV files in:", self.data_dir)
            for key, src in DATA_SOURCES.items():
                frames[key] = pd.read_csv(os.path.join(self.data_dir, src["csv"]))

        self.clinical_df = frames["clinical"]
        self.physio_df = frames["physiological"]
        self.transcriptomic_df = frames["transcriptomic"]
        self.lls_df = frames["lls"]

        for name, df in frames.items():
            print(f"  {name}: {df.shape}")
        return self

    def merge_modalities(self):
        """Merge all modalities into a single long-format dataframe."""
        longitudinal = pd.merge(self.physio_df, self.lls_df,
                                on=["participant_id", "day"], how="outer")
        longitudinal = pd.merge(longitudinal, self.transcriptomic_df,
                                on=["participant_id", "day"], how="outer")
        clinical_cols = ["participant_id", "ams_label"]
        self.merged_df = pd.merge(longitudinal, self.clinical_df[clinical_cols],
                                  on="participant_id", how="left")
        print(f"\nMerged dataset shape: {self.merged_df.shape}")
        return self

    # ------------------------------------------------------------------ cleaning
    def clean_data(self):
        """Remove duplicates/unlabelled rows and impute missing numeric values."""
        df = self.merged_df.copy()
        n0 = len(df)
        df = df.drop_duplicates(subset=["participant_id", "day"])
        df = df.dropna(subset=["ams_label"])

        self.gene_cols = [g for g in GENE_FEATURES if g in df.columns]
        value_cols = list(PHYSIOLOGICAL_FEATURES) + LLS_COLS + self.gene_cols
        value_cols = [c for c in value_cols if c in df.columns]

        n_missing = int(df[value_cols].isnull().sum().sum())
        # Median imputation within each time point
        for col in value_cols:
            df[col] = df[col].fillna(df.groupby("day")[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

        # Gene expression cannot be negative
        df[self.gene_cols] = df[self.gene_cols].clip(lower=0)

        self.merged_df = df.reset_index(drop=True)
        print(f"\nCleaning: removed {n0 - len(df)} rows, imputed {n_missing} missing values")
        return self

    # ------------------------------------------------------------ participant table
    def build_participant_table(self):
        """Create one row per participant (summary features + genes + label)."""
        df = self.merged_df.sort_values(["participant_id", "day"])
        days = sorted(df["day"].unique())
        first_day, last_day = days[0], days[-1]
        g = df.groupby("participant_id")

        table = pd.DataFrame(index=sorted(df["participant_id"].unique()))
        table.index.name = "participant_id"

        # AMS symptom score (Lake Louise Score) summaries
        table["ams_score_mean"] = g["lls_total"].mean()
        table["ams_score_max"] = g["lls_total"].max()

        # Physiological response summaries
        for col in ("spo2", "heart_rate"):
            table[f"{col}_mean"] = g[col].mean()
            wide = df.pivot(index="participant_id", columns="day", values=col)
            table[f"{col}_change"] = wide[last_day] - wide[first_day]

        # Gene expression at the last time point
        last = df[df["day"] == last_day].set_index("participant_id")
        table = table.join(last[self.gene_cols])

        table["ams_label"] = g["ams_label"].first().astype(int)
        table = table.fillna(table.median(numeric_only=True))

        print(f"\nParticipant table: {table.shape} "
              f"(AMS: {int(table['ams_label'].sum())}, "
              f"non-AMS: {int((1 - table['ams_label']).sum())})")
        self.participant_table = table
        return table

    def split_data(self, table=None, test_size=None):
        """Stratified train/test split of participants."""
        table = self.participant_table if table is None else table
        test_size = test_size or TRAINING_CONFIG["test_size"]
        train, test = train_test_split(
            table, test_size=test_size,
            random_state=TRAINING_CONFIG["random_seed"],
            stratify=table["ams_label"],
        )
        print(f"\nTrain: {len(train)} (AMS: {int(train['ams_label'].sum())}) | "
              f"Test: {len(test)} (AMS: {int(test['ams_label'].sum())})")
        return train, test

    def preprocess_pipeline(self):
        """Run load -> merge -> clean -> participant table -> split."""
        print("=" * 60)
        print("AMS Data Preprocessing Pipeline")
        print("=" * 60)
        self.load_data()
        self.merge_modalities()
        self.clean_data()
        table = self.build_participant_table()
        train, test = self.split_data(table)
        return {"table": table, "train": train, "test": test, "gene_cols": self.gene_cols}


if __name__ == "__main__":
    data = AMSDataPreprocessor().preprocess_pipeline()
    print(data["train"].head())
