import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch

SEED    = 42
DATASETS = ["german_credit", "heloc", "adult", "gmsc"]
results  = []

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*60}")

    try:
        model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
            get_trained_fnn(dataset, seed=SEED)

        model.eval()
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(X_test_t)).numpy().flatten()

        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n  --- Results ---")
        row = {"dataset": dataset, "features": X_train.shape[1], "auc": round(auc, 4)}

        for thr in [0.5, 0.45, 0.4]:
            y_pred = (y_pred_proba >= thr).astype(int)
            rec  = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred)
            print(f"  Threshold={thr} | AUC={auc:.4f} | "
                  f"Recall={rec:.4f} | Precision={prec:.4f} | F1={f1:.4f}")
            thr_str = str(thr).replace(".", "")
            row[f"recall_{thr_str}"]    = round(rec,  4)
            row[f"precision_{thr_str}"] = round(prec, 4)
            row[f"f1_{thr_str}"]        = round(f1,   4)

        results.append(row)

        if auc >= 0.75:
            print(f"  ✓ AUC={auc:.4f} — target (>=0.75) ACHIEVED")
        else:
            print(f"  ✗ AUC={auc:.4f} — below target")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({"dataset": dataset, "auc": None})

# === SUMMARY ===
print(f"\n{'='*60}")
print("SUMMARY — All Datasets")
print(f"{'='*60}")
print(f"{'Dataset':<15} {'Features':<10} {'AUC-ROC':<10} {'Target'}")
print(f"{'-'*45}")
for row in results:
    auc      = row.get('auc')
    features = row.get('features', '-')
    status   = '✓' if auc and auc >= 0.75 else '✗'
    auc_str  = f"{auc:.4f}" if auc else "ERROR"
    print(f"{row['dataset']:<15} {str(features):<10} {auc_str:<10} {status}")

# === CSV'YE KAYDET ===
os.makedirs("reports", exist_ok=True)
df_results = pd.DataFrame(results)
df_results.to_csv("reports/fnn_benchmark_results.csv", index=False)
print(f"\nResults saved → reports/fnn_benchmark_results.csv")