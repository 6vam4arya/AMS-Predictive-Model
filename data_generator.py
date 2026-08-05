"""
Synthetic Data Generator for AMS Prediction Model.

Generates realistic longitudinal, multi-modal data based on published research
findings about Acute Mountain Sickness from PubMed/Google Scholar sources.

References:
- Luks et al. (2017) - Wilderness Medical Society Practice Guidelines for AMS
- Roach et al. (2018) - Lake Louise AMS Score Consensus
- Mishra et al. (2015) - Transcriptomic changes at high altitude
- Zhang et al. (2014) - Gene expression and AMS susceptibility
- MacInnis et al. (2010) - Physiological predictors of AMS
"""

import numpy as np
import pandas as pd
import os
from config import (
    DATA_CONFIG,
    PHYSIOLOGICAL_FEATURES,
    GENE_FEATURES,
    CLINICAL_FEATURES,
    LLS_CONFIG,
)


class AMSDataGenerator:
    """Generates synthetic longitudinal multi-modal data for AMS prediction."""

    def __init__(self, seed=None):
        self.seed = seed or DATA_CONFIG["random_seed"]
        self.rng = np.random.default_rng(self.seed)
        self.n_participants = DATA_CONFIG["n_participants"]
        self.time_points = DATA_CONFIG["time_points"]
        self.ams_prevalence = DATA_CONFIG["ams_prevalence"]

    def generate_clinical_demographics(self):
        """Generate baseline clinical and demographic data."""
        n = self.n_participants
        data = {}

        for feature, params in CLINICAL_FEATURES.items():
            if "categories" in params:
                data[feature] = self.rng.choice(
                    params["categories"], size=n, p=params["probs"]
                )
            else:
                values = self.rng.normal(params["mean"], params["std"], size=n)
                values = np.clip(values, params.get("min", -np.inf), params.get("max", np.inf))
                data[feature] = values

        return pd.DataFrame(data)

    def _compute_ams_risk(self, clinical_df):
        """
        Compute individual AMS risk based on known risk factors.
        Based on literature: age, fitness, prior history, ascent rate, altitude.
        """
        n = self.n_participants
        risk_score = np.zeros(n)

        # Age effect (younger adults slightly higher risk in some studies)
        age_norm = (clinical_df["age"].values - 35) / 10
        risk_score += 0.1 * np.abs(age_norm)

        # Lower fitness increases risk
        fitness = clinical_df["fitness_level"].values
        risk_score += 0.15 * (5 - fitness) / 4

        # Prior AMS history strongly increases risk
        risk_score += 0.25 * clinical_df["prior_ams_history"].values

        # Faster ascent rate increases risk
        ascent_norm = (clinical_df["ascent_rate"].values - 500) / 150
        risk_score += 0.15 * np.clip(ascent_norm, 0, 2)

        # Higher target altitude increases risk
        alt_norm = (clinical_df["target_altitude"].values - 4500) / 800
        risk_score += 0.2 * np.clip(alt_norm, -1, 2)

        # Smoking slightly increases risk
        risk_score += 0.05 * clinical_df["smoking_status"].values

        # No altitude experience increases risk
        risk_score += 0.1 * (2 - clinical_df["altitude_experience"].values) / 2

        # Add noise
        risk_score += self.rng.normal(0, 0.1, size=n)

        # Convert to probability
        risk_prob = 1 / (1 + np.exp(-3 * (risk_score - 0.3)))

        # Assign AMS labels
        ams_labels = (self.rng.random(n) < risk_prob).astype(int)

        # Ensure approximate target prevalence
        current_prev = ams_labels.mean()
        if current_prev < self.ams_prevalence * 0.8:
            n_flip = int((self.ams_prevalence - current_prev) * n)
            non_ams_idx = np.where(ams_labels == 0)[0]
            flip_idx = self.rng.choice(non_ams_idx, size=min(n_flip, len(non_ams_idx)), replace=False)
            ams_labels[flip_idx] = 1
        elif current_prev > self.ams_prevalence * 1.2:
            n_flip = int((current_prev - self.ams_prevalence) * n)
            ams_idx = np.where(ams_labels == 1)[0]
            flip_idx = self.rng.choice(ams_idx, size=min(n_flip, len(ams_idx)), replace=False)
            ams_labels[flip_idx] = 0

        return ams_labels, risk_prob

    def generate_physiological_data(self, ams_labels):
        """
        Generate longitudinal physiological measurements.
        AMS individuals show: lower SpO2, higher HR, altered BP at altitude.
        """
        n = self.n_participants
        time_points = self.time_points
        records = []

        for i in range(n):
            is_ams = ams_labels[i]

            for t_idx, day in enumerate(time_points):
                record = {"participant_id": i, "day": day}

                # SpO2: decreases more in AMS individuals at altitude
                base_spo2 = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["spo2"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["spo2"]["baseline_std"],
                )
                altitude_effect = -3.0 * (t_idx + 1) / len(time_points)
                ams_effect = -4.0 * is_ams * (t_idx + 1) / len(time_points)
                record["spo2"] = np.clip(
                    base_spo2 + altitude_effect + ams_effect + self.rng.normal(0, 1),
                    70, 100,
                )

                # Heart rate: increases more in AMS
                base_hr = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["heart_rate"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["heart_rate"]["baseline_std"],
                )
                hr_altitude = 8.0 * (t_idx + 1) / len(time_points)
                hr_ams = 12.0 * is_ams * (t_idx + 1) / len(time_points)
                record["heart_rate"] = np.clip(
                    base_hr + hr_altitude + hr_ams + self.rng.normal(0, 3),
                    45, 160,
                )

                # Systolic BP: tends to increase at altitude
                base_sbp = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["systolic_bp"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["systolic_bp"]["baseline_std"],
                )
                sbp_altitude = 5.0 * (t_idx + 1) / len(time_points)
                sbp_ams = 8.0 * is_ams * (t_idx + 1) / len(time_points)
                record["systolic_bp"] = np.clip(
                    base_sbp + sbp_altitude + sbp_ams + self.rng.normal(0, 4),
                    80, 180,
                )

                # Diastolic BP
                base_dbp = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["diastolic_bp"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["diastolic_bp"]["baseline_std"],
                )
                dbp_altitude = 3.0 * (t_idx + 1) / len(time_points)
                dbp_ams = 5.0 * is_ams * (t_idx + 1) / len(time_points)
                record["diastolic_bp"] = np.clip(
                    base_dbp + dbp_altitude + dbp_ams + self.rng.normal(0, 3),
                    50, 120,
                )

                # Respiratory rate: increases at altitude, more in AMS
                base_rr = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["respiratory_rate"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["respiratory_rate"]["baseline_std"],
                )
                rr_altitude = 3.0 * (t_idx + 1) / len(time_points)
                rr_ams = 4.0 * is_ams * (t_idx + 1) / len(time_points)
                record["respiratory_rate"] = np.clip(
                    base_rr + rr_altitude + rr_ams + self.rng.normal(0, 1.5),
                    8, 35,
                )

                # Body temperature
                base_temp = self.rng.normal(
                    PHYSIOLOGICAL_FEATURES["body_temperature"]["baseline_mean"],
                    PHYSIOLOGICAL_FEATURES["body_temperature"]["baseline_std"],
                )
                temp_ams = 0.3 * is_ams * (t_idx + 1) / len(time_points)
                record["body_temperature"] = np.clip(
                    base_temp + temp_ams + self.rng.normal(0, 0.2),
                    35.5, 38.5,
                )

                records.append(record)

        return pd.DataFrame(records)

    def generate_transcriptomic_data(self, ams_labels):
        """
        Generate gene expression data for hypoxia-related genes.
        Based on published transcriptomic studies at high altitude.
        AMS individuals show differential expression in HIF pathway,
        inflammatory, and oxidative stress genes.
        """
        n = self.n_participants
        time_points = self.time_points
        n_genes = len(GENE_FEATURES)
        records = []

        # Define differential expression patterns for AMS
        # Positive = upregulated in AMS, Negative = downregulated
        gene_ams_effects = {
            # HIF pathway - upregulated in AMS
            "HIF1A": 1.5, "HIF2A": 1.2, "EPAS1": 1.3, "VHL": -0.8,
            "PHD2": -0.6, "FIH1": -0.5,
            # Erythropoiesis - upregulated
            "EPO": 2.0, "EPOR": 1.0, "GATA1": 0.8, "KLF1": 0.6,
            # Vascular - mixed
            "VEGFA": 1.8, "VEGFR2": 1.2, "ANG1": 0.5, "TIE2": 0.4, "NOS3": -1.0,
            # Inflammation - strongly upregulated in AMS
            "IL6": 2.5, "IL1B": 2.0, "TNF": 1.8, "CXCL8": 1.5, "CRP": 2.2,
            # Oxidative stress - upregulated
            "SOD2": 1.5, "CAT": 1.0, "GPX1": 0.8, "NRF2": 1.2, "HMOX1": 1.3,
            # RAS - altered
            "ACE": 1.2, "ACE2": -0.8, "AGT": 0.9, "AGTR1": 0.7,
            # Metabolism - upregulated (glycolytic shift)
            "LDHA": 1.8, "PDK1": 1.5, "GLUT1": 1.6, "PGK1": 1.2, "PKM2": 1.0,
        }

        for i in range(n):
            is_ams = ams_labels[i]

            # Individual baseline expression (log2 scale, centered around 8)
            individual_baseline = self.rng.normal(8.0, 1.5, size=n_genes)

            for t_idx, day in enumerate(time_points):
                record = {"participant_id": i, "day": day}

                time_factor = (t_idx + 1) / len(time_points)

                for g_idx, gene in enumerate(GENE_FEATURES):
                    base_expr = individual_baseline[g_idx]

                    # General altitude response (all individuals)
                    altitude_response = gene_ams_effects.get(gene, 0) * 0.3 * time_factor

                    # AMS-specific differential expression
                    ams_diff_expr = gene_ams_effects.get(gene, 0) * is_ams * time_factor

                    # Add biological noise
                    noise = self.rng.normal(0, 0.5)

                    expression = base_expr + altitude_response + ams_diff_expr + noise
                    record[gene] = max(0, expression)  # Expression can't be negative

                records.append(record)

        return pd.DataFrame(records)

    def generate_lake_louise_scores(self, ams_labels):
        """
        Generate Lake Louise Scores across time points.
        LLS components: headache, GI symptoms, fatigue/weakness, dizziness.
        Each scored 0-3, AMS diagnosed when total >= 3 with headache present.
        """
        n = self.n_participants
        time_points = self.time_points
        records = []

        for i in range(n):
            is_ams = ams_labels[i]

            for t_idx, day in enumerate(time_points):
                record = {"participant_id": i, "day": day}
                time_factor = (t_idx + 1) / len(time_points)

                if is_ams:
                    # AMS individuals: scores increase over time
                    headache_prob = np.clip(0.2 + 0.6 * time_factor, 0, 1)
                    symptom_severity = np.clip(0.5 * time_factor + self.rng.normal(0, 0.2), 0, 1)
                else:
                    # Non-AMS: mild or no symptoms
                    headache_prob = np.clip(0.1 + 0.1 * time_factor, 0, 0.3)
                    symptom_severity = np.clip(0.1 * time_factor + self.rng.normal(0, 0.1), 0, 0.4)

                # Generate component scores (0-3)
                has_headache = self.rng.random() < headache_prob
                record["lls_headache"] = int(has_headache) * min(3, max(0, int(
                    self.rng.normal(1.5 * symptom_severity * 3, 0.5)
                )))

                for component in ["gi_symptoms", "fatigue", "dizziness"]:
                    score = int(np.clip(
                        self.rng.normal(symptom_severity * 2, 0.8), 0, 3
                    ))
                    record[f"lls_{component}"] = score

                record["lls_total"] = (
                    record["lls_headache"]
                    + record["lls_gi_symptoms"]
                    + record["lls_fatigue"]
                    + record["lls_dizziness"]
                )

                records.append(record)

        return pd.DataFrame(records)

    def generate_full_dataset(self):
        """Generate the complete multi-modal longitudinal dataset."""
        print("=" * 60)
        print("AMS Synthetic Data Generator")
        print("Based on published research from PubMed/Google Scholar")
        print("=" * 60)

        # Step 1: Generate clinical demographics
        print("\n[1/5] Generating clinical/demographic data...")
        clinical_df = self.generate_clinical_demographics()

        # Step 2: Compute AMS risk and assign labels
        print("[2/5] Computing AMS risk scores and labels...")
        ams_labels, risk_probs = self._compute_ams_risk(clinical_df)
        clinical_df["ams_label"] = ams_labels
        clinical_df["ams_risk_prob"] = risk_probs
        clinical_df["participant_id"] = range(self.n_participants)

        print(f"    AMS prevalence: {ams_labels.mean():.2%} "
              f"({ams_labels.sum()}/{self.n_participants})")

        # Step 3: Generate physiological data
        print("[3/5] Generating longitudinal physiological data...")
        physio_df = self.generate_physiological_data(ams_labels)

        # Step 4: Generate transcriptomic data
        print("[4/5] Generating transcriptomic (gene expression) data...")
        transcriptomic_df = self.generate_transcriptomic_data(ams_labels)

        # Step 5: Generate Lake Louise Scores
        print("[5/5] Generating Lake Louise Scores...")
        lls_df = self.generate_lake_louise_scores(ams_labels)

        # Save datasets
        output_dir = DATA_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        clinical_df.to_csv(os.path.join(output_dir, "clinical_demographics.csv"), index=False)
        physio_df.to_csv(os.path.join(output_dir, "physiological_data.csv"), index=False)
        transcriptomic_df.to_csv(os.path.join(output_dir, "transcriptomic_data.csv"), index=False)
        lls_df.to_csv(os.path.join(output_dir, "lake_louise_scores.csv"), index=False)

        print(f"\n✓ Data saved to: {output_dir}/")
        print(f"  - clinical_demographics.csv: {clinical_df.shape}")
        print(f"  - physiological_data.csv: {physio_df.shape}")
        print(f"  - transcriptomic_data.csv: {transcriptomic_df.shape}")
        print(f"  - lake_louise_scores.csv: {lls_df.shape}")

        return clinical_df, physio_df, transcriptomic_df, lls_df


if __name__ == "__main__":
    generator = AMSDataGenerator()
    clinical_df, physio_df, transcriptomic_df, lls_df = generator.generate_full_dataset()

    print("\n" + "=" * 60)
    print("Dataset Summary Statistics")
    print("=" * 60)
    print(f"\nClinical Demographics:")
    print(clinical_df.describe().round(2))
    print(f"\nPhysiological Data (first time point):")
    print(physio_df[physio_df["day"] == 1].describe().round(2))
    print(f"\nGene Expression Range:")
    gene_cols = [c for c in transcriptomic_df.columns if c not in ["participant_id", "day"]]
    print(f"  Min: {transcriptomic_df[gene_cols].min().min():.2f}")
    print(f"  Max: {transcriptomic_df[gene_cols].max().max():.2f}")
    print(f"  Mean: {transcriptomic_df[gene_cols].mean().mean():.2f}")