"""
Stage 3: Naive Bayes classifier for AMS vs non-AMS prediction.

The classifier uses only the genes selected from the K-Means clusters as input
features. Gaussian Naive Bayes is used because gene expression values are
continuous: for each class it learns a per-gene normal distribution and assigns
a new individual to the class with the higher posterior probability.
"""

import os
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB

from config import NAIVE_BAYES_PARAMS, TRAINING_CONFIG


class AMSNaiveBayes:
    def __init__(self, genes=None, params=None):
        self.genes = list(genes) if genes is not None else None
        self.model = GaussianNB(**(params or NAIVE_BAYES_PARAMS))

    def fit(self, X, y):
        X = self._select(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(self._select(X))

    def predict_proba(self, X):
        """Probability of the AMS class (class 1)."""
        return self.model.predict_proba(self._select(X))[:, 1]

    def _select(self, X):
        return X[self.genes] if self.genes is not None else X


def save_artifacts(artifacts, save_dir=None):
    """Save dict of fitted objects (clusterer, selector, classifier) to disk."""
    save_dir = save_dir or TRAINING_CONFIG["model_save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    for name, obj in artifacts.items():
        joblib.dump(obj, os.path.join(save_dir, f"{name}.joblib"))
    print(f"\n✓ Models saved to: {save_dir}/")


def load_artifacts(save_dir=None):
    save_dir = save_dir or TRAINING_CONFIG["model_save_dir"]
    out = {}
    if not os.path.isdir(save_dir):
        return out
    for f in os.listdir(save_dir):
        if f.endswith(".joblib"):
            out[f[:-7]] = joblib.load(os.path.join(save_dir, f))
    return out
