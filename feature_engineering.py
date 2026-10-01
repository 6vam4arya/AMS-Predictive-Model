"""
Stage 2: gene (feature) selection - top genes per cluster.

For each cluster, every gene is scored by how strongly its expression in that
cluster differs from all other clusters (absolute Welch t-statistic, one-vs-rest).
The top N genes per cluster (default 5) are selected. Genes are picked in a
round-robin over the clusters so each gene is assigned to exactly one cluster
(4 clusters x 5 genes = 20 distinct genes) and no cluster dominates the list.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import GENE_SELECTION_CONFIG, CLUSTER_NAMES


class ClusterGeneSelector:
    def __init__(self, top_n=None):
        self.top_n = top_n or GENE_SELECTION_CONFIG["top_n_per_cluster"]
        self.scores_ = None
        self.genes_by_cluster = {}
        self.selected_genes = []

    def fit(self, gene_df, cluster_labels):
        """gene_df: participants x genes (training data only). cluster_labels: Series."""
        clusters = sorted(cluster_labels.unique())
        scores = {}
        for c in clusters:
            in_c = (cluster_labels == c).values
            t_stats, p_vals = stats.ttest_ind(
                gene_df.values[in_c], gene_df.values[~in_c],
                axis=0, equal_var=False,
            )
            scores[c] = pd.DataFrame(
                {"t_stat": np.nan_to_num(t_stats), "p_value": np.nan_to_num(p_vals, nan=1.0)},
                index=gene_df.columns,
            )
            scores[c]["abs_t"] = scores[c]["t_stat"].abs()
            scores[c]["mean_in_cluster"] = gene_df.values[in_c].mean(axis=0)
            scores[c]["mean_other"] = gene_df.values[~in_c].mean(axis=0)
        self.scores_ = scores

        # Round-robin selection of unique genes
        taken = set()
        self.genes_by_cluster = {c: [] for c in clusters}
        ranked = {c: scores[c].sort_values("abs_t", ascending=False).index.tolist() for c in clusters}
        for _ in range(self.top_n):
            for c in clusters:
                for gene in ranked[c]:
                    if gene not in taken:
                        taken.add(gene)
                        self.genes_by_cluster[c].append(gene)
                        break
        self.selected_genes = [g for c in clusters for g in self.genes_by_cluster[c]]
        return self

    def to_table(self):
        """Tidy table of selected genes (cluster, rank, gene, statistics)."""
        rows = []
        for c, genes in self.genes_by_cluster.items():
            for rank, gene in enumerate(genes, 1):
                r = self.scores_[c].loc[gene]
                rows.append({
                    "cluster": c, "profile": CLUSTER_NAMES.get(c, str(c)), "rank": rank,
                    "gene": gene, "t_stat": round(r["t_stat"], 3),
                    "p_value": float(f"{r['p_value']:.3g}"),
                    "mean_in_cluster": round(r["mean_in_cluster"], 3),
                    "mean_other_clusters": round(r["mean_other"], 3),
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    from preprocessing import AMSDataPreprocessor
    from clustering import AMSClusterer
    d = AMSDataPreprocessor().preprocess_pipeline()
    clusterer = AMSClusterer()
    labels = clusterer.fit(d["train"])
    sel = ClusterGeneSelector().fit(d["train"][d["gene_cols"]], labels)
    print(sel.to_table().to_string(index=False))
