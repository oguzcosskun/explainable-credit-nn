import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn
from src.models.tabnet_model import get_trained_tabnet

import torch

SEED     = 42
DATASETS = ["german_credit", "heloc", "adult", "gmsc"]
results  = []

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*60}")

    row = {"dataset": dataset}

    # === FNN ===
    try:
        model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
            get_trained_fnn(dataset, seed=SEED)
        model.eval()
        with torch.no_grad():
            y_proba = torch.sigmoid(model(X_test_t)).numpy().flatten()

        auc  = roc_auc_score(y_test, y_proba)
        pred = (y_proba >= 0.5).astype(int)
        row["fnn_features"]  = X_train.shape[1]
        row["fnn_auc"]       = round(auc, 4)
        row["fnn_recall"]    = round(recall_score(y_test, pred), 4)
        row["fnn_precision"] = round(precision_score(y_test, pred), 4)
        row["fnn_f1"]        = round(f1_score(y_test, pred), 4)
        print(f"  FNN  | AUC={auc:.4f} | "
              f"Recall={row['fnn_recall']:.4f} | "
              f"F1={row['fnn_f1']:.4f}")
    except Exception as e:
        print(f"  FNN ERROR: {e}")

    # === TabNet ===
    try:
        tn_model, X_train_tn, X_test_tn, y_train_tn, y_test_tn = \
            get_trained_tabnet(dataset, seed=SEED)
        y_proba_tn = tn_model.predict_proba(X_test_tn.values)[:, 1]

        auc_tn  = roc_auc_score(y_test_tn, y_proba_tn)
        pred_tn = (y_proba_tn >= 0.5).astype(int)
        row["tabnet_features"]  = X_train_tn.shape[1]
        row["tabnet_auc"]       = round(auc_tn, 4)
        row["tabnet_recall"]    = round(recall_score(y_test_tn, pred_tn), 4)
        row["tabnet_precision"] = round(precision_score(y_test_tn, pred_tn), 4)
        row["tabnet_f1"]        = round(f1_score(y_test_tn, pred_tn), 4)
        print(f"  TabNet | AUC={auc_tn:.4f} | "
              f"Recall={row['tabnet_recall']:.4f} | "
              f"F1={row['tabnet_f1']:.4f}")
    except Exception as e:
        print(f"  TabNet ERROR: {e}")

    results.append(row)

# === SUMMARY ===
print(f"\n{'='*60}")
print("SUMMARY — FNN vs TabNet")
print(f"{'='*60}")
print(f"{'Dataset':<15} {'FNN AUC':<10} {'TabNet AUC':<12} {'Winner'}")
print(f"{'-'*50}")
for row in results:
    fnn_auc    = row.get('fnn_auc', 0)
    tabnet_auc = row.get('tabnet_auc', 0)
    winner     = "FNN" if fnn_auc >= tabnet_auc else "TabNet"
    print(f"{row['dataset']:<15} {str(fnn_auc):<10} {str(tabnet_auc):<12} {winner}")

# === CSV ===
os.makedirs("reports", exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("reports/fnn_vs_tabnet_benchmark.csv", index=False)
print(f"\nSaved → reports/fnn_vs_tabnet_benchmark.csv")