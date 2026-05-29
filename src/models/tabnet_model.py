import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare

from pytorch_tabnet.tab_model import TabNetClassifier

SEED = 42
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_trained_tabnet(dataset="german_credit", seed=SEED,
                       force_retrain=False):
    """
    TabNet modelini egitir veya checkpoint'ten yukler.
    Returns: model, X_train, X_test, y_train, y_test
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR,
                             f"tabnet_best_{dataset}")

    np.random.seed(seed)

    X_train_full, X_test, y_train_full, y_test, _ = prepare(
        dataset, "tabnet", random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15, random_state=seed, stratify=y_train_full
    )
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)

    # pos_weight hesapla
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_w = n_neg / n_pos

    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | "
          f"Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")
    print(f"  pos_weight: {pos_w:.3f}")

    model = TabNetClassifier(
    n_d=32, n_a=32,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    momentum=0.02,
    epsilon=1e-15,
    seed=seed,
    verbose=0,
)

    # Checkpoint varsa yukle
    if os.path.exists(ckpt_path + ".zip") and not force_retrain:
        model.load_model(ckpt_path + ".zip")
        val_proba = model.predict_proba(X_val.values)[:, 1]
        val_auc   = roc_auc_score(y_val, val_proba)
        print(f"  Loaded checkpoint: {ckpt_path} | val_auc={val_auc:.4f}")
        return model, X_train, X_test, y_train, y_test

    # Egit
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=10,
        batch_size=64,
        virtual_batch_size=32,
        weights=1,
    )

    # Kaydet
    model.save_model(ckpt_path)

    val_proba = model.predict_proba(X_val.values)[:, 1]
    val_auc   = roc_auc_score(y_val, val_proba)
    print(f"  TabNet trained & saved | val_auc={val_auc:.4f} | "
          f"features={X_train.shape[1]} | checkpoint={ckpt_path}")

    return model, X_train, X_test, y_train, y_test


# === TEST ===
if __name__ == "__main__":
    import traceback
    from sklearn.metrics import (roc_auc_score, recall_score,
                                  precision_score, classification_report)

    DATASETS = ["german_credit", "heloc", "adult", "gmsc"]
    results  = {}

    for dataset in DATASETS:
        print(f"\n{'='*60}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*60}")
        try:
            model, X_train, X_test, y_train, y_test = \
                get_trained_tabnet(dataset)
            y_proba = model.predict_proba(X_test.values)[:, 1]
            auc     = roc_auc_score(y_test, y_proba)
            y_pred  = (y_proba >= 0.5).astype(int)
            recall  = recall_score(y_test, y_pred)
            prec    = precision_score(y_test, y_pred)
            print(f"\n  AUC-ROC:   {auc:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  Precision: {prec:.4f}")
            results[dataset] = {"auc": round(auc, 4), 
                                "features": X_train.shape[1]}
            status = "✓" if auc >= 0.75 else "✗"
            print(f"  {status} Target (>=0.75)")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results[dataset] = {"auc": None}

    print(f"\n{'='*60}")
    print("SUMMARY — TabNet All Datasets")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'Features':<10} {'AUC-ROC':<10} {'Target'}")
    print(f"{'-'*45}")
    for ds, res in results.items():
        auc      = res.get('auc')
        features = res.get('features', '-')
        status   = '✓' if auc and auc >= 0.75 else '✗'
        auc_str  = f"{auc:.4f}" if auc else "ERROR"
        print(f"{ds:<15} {str(features):<10} {auc_str:<10} {status}")