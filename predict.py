"""
Inference: predict AMS susceptibility of new individuals from gene expression.

Usage:
    python predict.py                   # demo with a random synthetic individual
    python predict.py --csv genes.csv   # CSV with one row per individual and the
                                        # selected gene columns
"""

import argparse
import numpy as np
import pandas as pd

from config import GENE_FEATURES
from models import load_artifacts


class AMSPredictor:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir

    def load(self):
        art = load_artifacts(self.model_dir)
        if "naive_bayes" not in art:
            raise FileNotFoundError("No trained model found. Run train.py first.")
        self.clf = art["naive_bayes"]
        self.genes = self.clf.genes
        return self

    def predict(self, expression):
        """expression: DataFrame (rows = individuals) or dict {gene: value}."""
        if isinstance(expression, dict):
            expression = pd.DataFrame([expression])
        missing = [g for g in self.genes if g not in expression.columns]
        if missing:
            raise ValueError(f"Missing required gene columns: {missing}")
        proba = self.clf.predict_proba(expression)
        return pd.DataFrame({
            "ams_probability": proba,
            "prediction": np.where(proba >= 0.5, "AMS", "Non-AMS"),
        }, index=expression.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV file with gene expression columns")
    args = ap.parse_args()

    predictor = AMSPredictor().load()
    print("Genes used by the model:", predictor.genes)
    if args.csv:
        data = pd.read_csv(args.csv)
    else:
        rng = np.random.default_rng(0)
        data = pd.DataFrame([{g: rng.normal(9.5, 1.5) for g in GENE_FEATURES}])
    print(predictor.predict(data).to_string())
