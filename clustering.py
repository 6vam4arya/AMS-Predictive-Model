"""
Stage 1: K-Means clustering (K=4) of participants.

Participants are grouped by AMS symptom score, oxygen saturation and
physiological response. Clusters are then given the project's biological
profile names:

  1 - Low AMS,  Fast acclimatizers
  2 - Low AMS,  Slow acclimatizers
  3 - High AMS, Fast acclimatizers
  4 - High AMS, Poor acclimatization
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import CLUSTER_CONFIG, CLUSTER_NAMES


class AMSClusterer:
    """K-Means clustering with biologically-ordered cluster ids (1-4)."""

    def __init__(self, config=None):
        self.config = config or CLUSTER_CONFIG
        self.features = self.config["features"]
        self.scaler = StandardScaler()
        self.kmeans = KMeans(
            n_clusters=self.config["n_clusters"],
            n_init=self.config["n_init"],
            random_state=self.config["random_state"],
        )
        self.label_map = None  # raw KMeans label -> project cluster id (1-4)

    def fit(self, table):
        X = self.scaler.fit_transform(table[self.features])
        raw = self.kmeans.fit_predict(X)
        self.label_map = self._build_label_map(table, raw)
        labels = pd.Series(raw, index=table.index).map(self.label_map)
        self.silhouette = float(silhouette_score(X, raw))
        print(f"\nK-Means fitted (K={self.config['n_clusters']}), "
              f"silhouette = {self.silhouette:.3f}")
        return labels.rename("cluster")

    def predict(self, table):
        X = self.scaler.transform(table[self.features])
        raw = self.kmeans.predict(X)
        return pd.Series(raw, index=table.index).map(self.label_map).rename("cluster")

    def _build_label_map(self, table, raw):
        """
        Order clusters by mean AMS score (low pair / high pair), then within each
        pair by mean SpO2 (higher SpO2 = faster acclimatization).
        """
        stats = table.assign(raw=raw).groupby("raw")[["ams_score_mean", "spo2_mean"]].mean()
        by_ams = stats.sort_values("ams_score_mean").index.tolist()
        low, high = by_ams[:2], by_ams[2:]
        low = sorted(low, key=lambda r: -stats.loc[r, "spo2_mean"])    # fast first
        high = sorted(high, key=lambda r: -stats.loc[r, "spo2_mean"])  # fast first
        return {raw_id: i + 1 for i, raw_id in enumerate(low + high)}

    def summarize(self, table, labels):
        """Per-cluster size, AMS prevalence and mean feature values."""
        df = table.assign(cluster=labels)
        summary = df.groupby("cluster").agg(
            n_participants=("ams_label", "size"),
            ams_cases=("ams_label", "sum"),
            **{f: (f, "mean") for f in self.features},
        )
        summary.insert(0, "profile", [CLUSTER_NAMES[c] for c in summary.index])
        return summary.round(3)


if __name__ == "__main__":
    from preprocessing import AMSDataPreprocessor
    d = AMSDataPreprocessor().preprocess_pipeline()
    c = AMSClusterer()
    lab = c.fit(d["train"])
    print(c.summarize(d["train"], lab).to_string())
