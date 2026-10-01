"""
Evaluation for the AMS Naive Bayes classifier.

Reports accuracy, precision, recall, F1 and AUC-ROC, saves the confusion matrix
and ROC figures, and writes cluster / gene tables for the project report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

from config import TRAINING_CONFIG, CLUSTER_NAMES


class AMSModelEvaluator:
    def __init__(self, results_dir=None):
        self.results_dir = results_dir or TRAINING_CONFIG["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    def compute_metrics(self, y_true, y_pred, y_proba=None):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }
        if y_proba is not None:
            m["auc_roc"] = roc_auc_score(y_true, y_proba)
        return m

    def plot_confusion_matrix(self, y_true, y_pred, filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Non-AMS", "AMS"])
        ax.set_yticks([0, 1], ["Non-AMS", "AMS"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Naive Bayes - Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        fig.tight_layout()
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_roc(self, y_true, y_proba, filename="roc_curve.png"):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Naive Bayes - ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_clusters(self, X_scaled, labels, filename="kmeans_clusters.png"):
        """2-D PCA view of the four K-Means clusters."""
        pcs = PCA(n_components=2, random_state=0).fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for c in sorted(labels.unique()):
            idx = (labels == c).values
            ax.scatter(pcs[idx, 0], pcs[idx, 1], s=14, alpha=0.7,
                       label=f"Cluster {c}: {CLUSTER_NAMES[c]}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("K-Means Clusters (K=4)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(self.results_dir, filename)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def save_report(self, report):
        path = os.path.join(self.results_dir, "evaluation_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_to_native)
        return path


def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
