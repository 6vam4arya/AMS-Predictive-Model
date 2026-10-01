# AMS-Predictive-Model

Machine-learning-based genetic analysis of Acute Mountain Sickness (AMS).

## Pipeline
1. **Preprocessing** (`preprocessing.py`) – load Excel/CSV data, merge clinical, physiological, transcriptomic and Lake Louise (AMS) score tables, clean/impute, build one row per participant, stratified train/test split.
2. **K-Means clustering, K=4** (`clustering.py`) – groups individuals by AMS score, SpO2 and physiological response into:
   1. Low AMS, fast acclimatizers
   2. Low AMS, slow acclimatizers
   3. High AMS, fast acclimatizers
   4. High AMS, poor acclimatization
3. **Gene selection** (`feature_engineering.py`) – top 5 genes per cluster (cluster-vs-rest Welch t-statistic, unique genes) = 20 genes.
4. **Naive Bayes classifier** (`models.py`) – Gaussian NB on the 20 selected genes predicts AMS vs non-AMS.
5. **Evaluation** (`evaluate.py`) – accuracy, precision, recall, F1, AUC, confusion matrix and ROC figures.

Clustering and gene selection are fitted on the training set only (no test-set leakage).

## Run
```
pip install -r requirements.txt
python train.py      # generates synthetic data if data/ is empty, trains, evaluates
python predict.py    # predict from gene expression (see --csv option)
```

## Using real data
Put an Excel workbook `data/ams_data.xlsx` with sheets `clinical`, `physiological`, `transcriptomic`, `lls` (same columns as the synthetic CSVs in `data/`; `clinical` must contain `participant_id` and `ams_label`), or replace the CSVs. Adjust names in `config.py` (`DATA_SOURCES`, `GENE_FEATURES`, `CLUSTER_CONFIG["features"]`).

## Outputs (`results/`)
`cluster_summary.csv`, `selected_genes.csv`, `confusion_matrix.png`, `roc_curve.png`, `kmeans_clusters.png`, `evaluation_report.json`.
