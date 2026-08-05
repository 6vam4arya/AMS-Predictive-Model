"""
Feature Engineering Module for AMS Prediction.

Extracts longitudinal features including:
- Temporal trends and slopes
- Rate of change between time points
- Statistical aggregations
- Interaction features between modalities
- Gene expression ratios and pathway scores
"""

import numpy as np
import pandas as pd
from scipy import stats
from config import DATA_CONFIG, GENE_FEATURES


class AMSFeatureEngineer:
    """Extracts advanced features from longitudinal multi-modal data."""

    def __init__(self):
        self.time_points = DATA_CONFIG["time_points"]
        self.feature_names = []

    def compute_temporal_slopes(self, df, feature_cols, participant_col="participant_id"):
        """
        Compute linear regression slope for each feature over time.
        Captures the rate of change in physiological/molecular markers.
        """
        slopes = {}
        participants = df[participant_col].unique()

        for col in feature_cols:
            col_slopes = []
            for pid in participants:
                pdata = df[df[participant_col] == pid].sort_values("day")
                if len(pdata) >= 2:
                    slope, _, _, _, _ = stats.linregress(
                        pdata["day"].values, pdata[col].values
                    )
                    col_slopes.append(slope)
                else:
                    col_slopes.append(0.0)
            slopes[f"{col}_slope"] = col_slopes

        return pd.DataFrame(slopes, index=participants)

    def compute_rate_of_change(self, df, feature_cols, participant_col="participant_id"):
        """
        Compute rate of change between consecutive time points.
        Useful for detecting rapid deterioration patterns.
        """
        roc_features = {}
        participants = df[participant_col].unique()
        time_points = sorted(df["day"].unique())

        for col in feature_cols:
            for i in range(1, len(time_points)):
                roc_key = f"{col}_roc_d{time_points[i-1]}_d{time_points[i]}"
                roc_values = []

                for pid in participants:
                    pdata = df[df[participant_col] == pid].sort_values("day")
                    if len(pdata) >= i + 1:
                        val_prev = pdata[pdata["day"] == time_points[i-1]][col].values
                        val_curr = pdata[pdata["day"] == time_points[i]][col].values
                        if len(val_prev) > 0 and len(val_curr) > 0:
                            roc = (val_curr[0] - val_prev[0]) / max(
                                (time_points[i] - time_points[i-1]), 1
                            )
                            roc_values.append(roc)
                        else:
                            roc_values.append(0.0)
                    else:
                        roc_values.append(0.0)

                roc_features[roc_key] = roc_values

        return pd.DataFrame(roc_features, index=participants)

    def compute_statistical_aggregations(self, df, feature_cols, participant_col="participant_id"):
        """
        Compute statistical summaries across time points for each participant.
        Includes mean, std, min, max, range, and coefficient of variation.
        """
        agg_features = {}
        participants = df[participant_col].unique()

        for col in feature_cols:
            means, stds, mins, maxs, ranges, cvs = [], [], [], [], [], []

            for pid in participants:
                values = df[df[participant_col] == pid][col].values
                means.append(np.mean(values))
                stds.append(np.std(values))
                mins.append(np.min(values))
                maxs.append(np.max(values))
                ranges.append(np.max(values) - np.min(values))
                cv = np.std(values) / (np.mean(values) + 1e-8)
                cvs.append(cv)

            agg_features[f"{col}_mean"] = means
            agg_features[f"{col}_std"] = stds
            agg_features[f"{col}_min"] = mins
            agg_features[f"{col}_max"] = maxs
            agg_features[f"{col}_range"] = ranges
            agg_features[f"{col}_cv"] = cvs

        return pd.DataFrame(agg_features, index=participants)

    def compute_gene_pathway_scores(self, df, participant_col="participant_id"):
        """
        Compute pathway-level scores by aggregating related genes.
        Based on known biological pathways involved in AMS.
        """
        pathways = {
            "hif_pathway": ["HIF1A", "HIF2A", "EPAS1", "VHL", "PHD2", "FIH1"],
            "erythropoiesis": ["EPO", "EPOR", "GATA1", "KLF1"],
            "angiogenesis": ["VEGFA", "VEGFR2", "ANG1", "TIE2", "NOS3"],
            "inflammation": ["IL6", "IL1B", "TNF", "CXCL8", "CRP"],
            "oxidative_stress": ["SOD2", "CAT", "GPX1", "NRF2", "HMOX1"],
            "ras_system": ["ACE", "ACE2", "AGT", "AGTR1"],
            "glycolysis": ["LDHA", "PDK1", "GLUT1", "PGK1", "PKM2"],
        }

        pathway_features = {}
        participants = df[participant_col].unique()

        for pathway_name, genes in pathways.items():
            available_genes = [g for g in genes if g in df.columns]
            if not available_genes:
                continue

            pathway_means = []
            pathway_vars = []

            for pid in participants:
                pdata = df[df[participant_col] == pid]
                # Use last time point for pathway score
                last_day = pdata["day"].max()
                last_data = pdata[pdata["day"] == last_day][available_genes].values

                if len(last_data) > 0:
                    pathway_means.append(np.mean(last_data))
                    pathway_vars.append(np.var(last_data))
                else:
                    pathway_means.append(0.0)
                    pathway_vars.append(0.0)

            pathway_features[f"pathway_{pathway_name}_mean"] = pathway_means
            pathway_features[f"pathway_{pathway_name}_var"] = pathway_vars

        return pd.DataFrame(pathway_features, index=participants)

    def compute_interaction_features(self, df, participant_col="participant_id"):
        """
        Compute interaction features between different modalities.
        E.g., SpO2 × inflammatory gene expression, HR × HIF pathway.
        """
        interaction_features = {}
        participants = df[participant_col].unique()

        # Define meaningful interactions based on AMS pathophysiology
        interactions = [
            ("spo2", "HIF1A", "spo2_x_hif1a"),
            ("spo2", "EPO", "spo2_x_epo"),
            ("heart_rate", "IL6", "hr_x_il6"),
            ("heart_rate", "CRP", "hr_x_crp"),
            ("systolic_bp", "ACE", "sbp_x_ace"),
            ("respiratory_rate", "VEGFA", "rr_x_vegfa"),
        ]

        for feat1, feat2, name in interactions:
            if feat1 not in df.columns or feat2 not in df.columns:
                continue

            values = []
            for pid in participants:
                pdata = df[df[participant_col] == pid]
                last_day = pdata["day"].max()
                last_data = pdata[pdata["day"] == last_day]

                if len(last_data) > 0:
                    v1 = last_data[feat1].values[0]
                    v2 = last_data[feat2].values[0]
                    values.append(v1 * v2)
                else:
                    values.append(0.0)

            interaction_features[name] = values

        return pd.DataFrame(interaction_features, index=participants)

    def compute_lls_trajectory_features(self, df, participant_col="participant_id"):
        """
        Extract features from Lake Louise Score trajectory.
        Captures symptom progression patterns.
        """
        lls_features = {}
        participants = df[participant_col].unique()

        lls_cols = ["lls_total", "lls_headache", "lls_gi_symptoms",
                    "lls_fatigue", "lls_dizziness"]

        for col in lls_cols:
            if col not in df.columns:
                continue

            max_vals, last_vals, increases = [], [], []

            for pid in participants:
                pdata = df[df[participant_col] == pid].sort_values("day")
                values = pdata[col].values

                max_vals.append(np.max(values))
                last_vals.append(values[-1] if len(values) > 0 else 0)

                # Number of increases between consecutive measurements
                if len(values) > 1:
                    diffs = np.diff(values)
                    increases.append(np.sum(diffs > 0))
                else:
                    increases.append(0)

            lls_features[f"{col}_max"] = max_vals
            lls_features[f"{col}_last"] = last_vals
            lls_features[f"{col}_n_increases"] = increases

        return pd.DataFrame(lls_features, index=participants)

    def compute_early_warning_features(self, df, participant_col="participant_id"):
        """
        Compute early warning features from Day 1 data only.
        These features enable early prediction before symptoms appear.
        """
        early_features = {}
        participants = df[participant_col].unique()
        first_day = min(self.time_points)

        day1_data = df[df["day"] == first_day]

        physio_cols = ["spo2", "heart_rate", "systolic_bp", "diastolic_bp",
                       "respiratory_rate", "body_temperature"]

        for col in physio_cols:
            if col not in day1_data.columns:
                continue
            values = []
            for pid in participants:
                pdata = day1_data[day1_data[participant_col] == pid]
                if len(pdata) > 0:
                    values.append(pdata[col].values[0])
                else:
                    values.append(np.nan)
            early_features[f"early_{col}"] = values

        # Early gene expression markers
        key_early_genes = ["HIF1A", "EPO", "IL6", "CRP", "VEGFA", "NOS3"]
        for gene in key_early_genes:
            if gene not in day1_data.columns:
                continue
            values = []
            for pid in participants:
                pdata = day1_data[day1_data[participant_col] == pid]
                if len(pdata) > 0:
                    values.append(pdata[gene].values[0])
                else:
                    values.append(np.nan)
            early_features[f"early_{gene}"] = values

        return pd.DataFrame(early_features, index=participants)

    def engineer_all_features(self, merged_df):
        """
        Run the complete feature engineering pipeline.
        Returns enriched feature matrix with all engineered features.
        """
        print("=" * 60)
        print("Feature Engineering Pipeline")
        print("=" * 60)

        physio_cols = ["spo2", "heart_rate", "systolic_bp", "diastolic_bp",
                       "respiratory_rate", "body_temperature"]
        gene_cols = [g for g in GENE_FEATURES if g in merged_df.columns]

        # 1. Temporal slopes
        print("\n[1/7] Computing temporal slopes...")
        slopes_physio = self.compute_temporal_slopes(merged_df, physio_cols)
        slopes_genes = self.compute_temporal_slopes(
            merged_df, gene_cols[:10]  # Top 10 genes for efficiency
        )

        # 2. Rate of change
        print("[2/7] Computing rate of change...")
        roc_physio = self.compute_rate_of_change(merged_df, physio_cols)

        # 3. Statistical aggregations
        print("[3/7] Computing statistical aggregations...")
        agg_physio = self.compute_statistical_aggregations(merged_df, physio_cols)

        # 4. Gene pathway scores
        print("[4/7] Computing gene pathway scores...")
        pathway_scores = self.compute_gene_pathway_scores(merged_df)

        # 5. Interaction features
        print("[5/7] Computing interaction features...")
        interactions = self.compute_interaction_features(merged_df)

        # 6. LLS trajectory features
        print("[6/7] Computing LLS trajectory features...")
        lls_trajectory = self.compute_lls_trajectory_features(merged_df)

        # 7. Early warning features
        print("[7/7] Computing early warning features...")
        early_warning = self.compute_early_warning_features(merged_df)

        # Combine all engineered features
        engineered_df = pd.concat([
            slopes_physio,
            slopes_genes,
            roc_physio,
            agg_physio,
            pathway_scores,
            interactions,
            lls_trajectory,
            early_warning,
        ], axis=1)

        # Handle any NaN values from feature engineering
        engineered_df = engineered_df.fillna(0)

        self.feature_names = list(engineered_df.columns)
        print(f"\n✓ Engineered {len(self.feature_names)} additional features")
        print(f"  Feature categories:")
        print(f"    - Temporal slopes: {slopes_physio.shape[1] + slopes_genes.shape[1]}")
        print(f"    - Rate of change: {roc_physio.shape[1]}")
        print(f"    - Statistical aggregations: {agg_physio.shape[1]}")
        print(f"    - Pathway scores: {pathway_scores.shape[1]}")
        print(f"    - Interactions: {interactions.shape[1]}")
        print(f"    - LLS trajectory: {lls_trajectory.shape[1]}")
        print(f"    - Early warning: {early_warning.shape[1]}")

        return engineered_df


if __name__ == "__main__":
    from preprocessing import AMSDataPreprocessor

    # Load and preprocess data
    preprocessor = AMSDataPreprocessor()
    preprocessor.load_data()
    preprocessor.merge_modalities()

    # Engineer features
    fe = AMSFeatureEngineer()
    engineered_features = fe.engineer_all_features(preprocessor.merged_df)
    print(f"\nFinal engineered feature matrix: {engineered_features.shape}")
    print(f"Sample features:\n{engineered_features.head()}")