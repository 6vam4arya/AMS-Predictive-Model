"""
Training pipeline for the AMS project:

  1. Data (generate synthetic data if none exists) and preprocessing
  2. Train/test split of participants
  3. K-Means clustering (K=4)                      [training data only]
  4. Top-5 genes per cluster (20 genes)            [training data only]
  5. Naive Bayes classifier on the selected genes  [training data only]
  6. Evaluation on the held-out test set

Clustering and gene selection use the training set only, so no information
from the test participants leaks into the classifier.
"""

import os
import warnings
warnings.filterwarnings("ignore")

from config import DATA_CONFIG, TRAINING_CONFIG, DATA_SOURCES, CLUSTER_NAMES
from data_generator import AMSDataGenerator
from preprocessing import AMSDataPreprocessor
from clustering import AMSClusterer
from feature_engineering import ClusterGeneSelector
from models import AMSNaiveBayes, save_artifacts
from evaluate import AMSModelEvaluator


def ensure_data_exists():
    """Generate a synthetic dataset if no data is available."""
    data_dir = DATA_CONFIG["output_dir"]
    has_xlsx = os.path.exists(os.path.join(data_dir, DATA_CONFIG["excel_file"]))
    has_csv = os.path.exists(os.path.join(data_dir, DATA_SOURCES["clinical"]["csv"]))
    if has_xlsx or has_csv:
        print("✓ Data files found.")
    else:
        print("Data not found. Generating synthetic dataset...")
        AMSDataGenerator().generate_full_dataset()


def run_training_pipeline():
    print("=" * 70)
    print("  AMS PREDICTION - K-MEANS + NAIVE BAYES PIPELINE")
    print("=" * 70)

    ensure_data_exists()

    # --- Preprocessing & split -------------------------------------------------
    data = AMSDataPreprocessor().preprocess_pipeline()
    train, test, gene_cols = data["train"], data["test"], data["gene_cols"]
    evaluator = AMSModelEvaluator()
    results_dir = TRAINING_CONFIG["results_dir"]

    # --- Stage 1: K-Means ------------------------------------------------------
    print("\n" + "-" * 70 + "\nSTAGE 1: K-Means clustering (K=4)\n" + "-" * 70)
    clusterer = AMSClusterer()
    train_clusters = clusterer.fit(train)
    cluster_summary = clusterer.summarize(train, train_clusters)
    print(cluster_summary.to_string())
    cluster_summary.to_csv(os.path.join(results_dir, "cluster_summary.csv"))
    evaluator.plot_clusters(
        clusterer.scaler.transform(train[clusterer.features]), train_clusters
    )

    # --- Stage 2: top genes per cluster ----------------------------------------
    print("\n" + "-" * 70 + "\nSTAGE 2: Top 5 genes per cluster\n" + "-" * 70)
    selector = ClusterGeneSelector().fit(train[gene_cols], train_clusters)
    gene_table = selector.to_table()
    print(gene_table.to_string(index=False))
    gene_table.to_csv(os.path.join(results_dir, "selected_genes.csv"), index=False)
    print(f"\nSelected genes ({len(selector.selected_genes)}): {selector.selected_genes}")

    # --- Stage 3: Naive Bayes --------------------------------------------------
    print("\n" + "-" * 70 + "\nSTAGE 3: Naive Bayes classifier\n" + "-" * 70)
    clf = AMSNaiveBayes(genes=selector.selected_genes)
    clf.fit(train, train["ams_label"])

    y_test = test["ams_label"]
    y_pred = clf.predict(test)
    y_proba = clf.predict_proba(test)
    metrics = evaluator.compute_metrics(y_test, y_pred, y_proba)

    print("\nTest-set performance:")
    for k in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc"]:
        print(f"  {k:12s}: {metrics[k]:.4f}")
    print(f"  Confusion matrix  TN={metrics['tn']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TP={metrics['tp']}")

    evaluator.plot_confusion_matrix(y_test, y_pred)
    evaluator.plot_roc(y_test, y_proba)

    report = {
        "n_train": len(train), "n_test": len(test),
        "silhouette_score": clusterer.silhouette,
        "cluster_summary": cluster_summary.reset_index().to_dict("records"),
        "selected_genes": {int(c): g for c, g in selector.genes_by_cluster.items()},
        "test_metrics": metrics,
    }
    evaluator.save_report(report)

    save_artifacts({"clusterer": clusterer, "gene_selector": selector, "naive_bayes": clf})
    print(f"\n✓ Results (tables, figures, report) saved to: {results_dir}/")
    return report


if __name__ == "__main__":
    run_training_pipeline()
