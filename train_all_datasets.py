import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.fnn import FNN

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DATASETS = ["german_credit", "heloc", "adult", "gmsc"]
results  = {}

for dataset in DATASETS:
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*60}")

    try:
        # === DATA ===
        X_train_full, X_test, y_train_full, y_test, _ = prepare(
            dataset, "fnn", random_state=SEED
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=0.15, random_state=SEED, stratify=y_train_full
        )
        X_train = X_train.reset_index(drop=True)
        X_val   = X_val.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_val   = y_val.reset_index(drop=True)

        print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | "
              f"Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

        # === TENSORS ===
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
        X_val_t   = torch.tensor(X_val.values,   dtype=torch.float32)
        X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=32, shuffle=True
        )

        # === MODEL ===
        n_neg     = (y_train == 0).sum()
        n_pos     = (y_train == 1).sum()
        pos_w     = torch.tensor([n_neg / n_pos], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        model     = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # === TRAINING ===
        best_val_loss = float('inf')
        best_state    = None
        patience      = 10
        counter       = 0
        best_epoch    = 0

        for epoch in range(1, 101):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(
                    model(X_val_t),
                    torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
                ).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch    = epoch
                counter       = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch={best_epoch})")
                    break

        model.load_state_dict(best_state)

        # === EVALUATION ===
        model.eval()
        with torch.no_grad():
            y_pred_proba = torch.sigmoid(model(X_test_t)).numpy().flatten()

        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n  --- Results ---")
        for thr in [0.5, 0.45, 0.4]:
            y_pred = (y_pred_proba >= thr).astype(int)
            rec  = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred)
            print(f"  Threshold={thr} | AUC={auc:.4f} | "
                  f"Recall={rec:.4f} | Precision={prec:.4f} | F1={f1:.4f}")

        results[dataset] = {"auc": auc, "features": X_train.shape[1]}

        if auc >= 0.75:
            print(f"  ✓ AUC={auc:.4f} — target (>=0.75) ACHIEVED")
        else:
            print(f"  ✗ AUC={auc:.4f} — below target")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[dataset] = {"auc": None}

# === SUMMARY ===
print(f"\n{'='*60}")
print("SUMMARY — All Datasets")
print(f"{'='*60}")
print(f"{'Dataset':<15} {'Features':<10} {'AUC-ROC':<10} {'Target'}")
print(f"{'-'*45}")
for ds, res in results.items():
    auc      = res.get('auc')
    features = res.get('features', '-')
    status   = '✓' if auc and auc >= 0.75 else '✗'
    auc_str  = f"{auc:.4f}" if auc else "ERROR"
    print(f"{ds:<15} {str(features):<10} {auc_str:<10} {status}")