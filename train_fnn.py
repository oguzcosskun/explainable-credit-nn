import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from sklearn.metrics import (roc_auc_score, recall_score,
                              precision_score, classification_report)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.fnn import FNN

# === REPRODUCIBILITY ===
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_and_evaluate():
    print("=" * 60)
    print("FNN BASELINE -- German Credit Dataset (UCI)")
    print("=" * 60)

    # === 1. DATA ===
    print("\n[1/5] Loading and preparing data...")
    X_train_full, X_test, y_train_full, y_test, _ = prepare(
        "german_credit", "fnn", random_state=SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15, random_state=SEED, stratify=y_train_full
    )
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)

    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # === 2. TENSORS ===
    print("\n[2/5] Converting to tensors...")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val.values,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32, shuffle=True
    )

    # === 3. MODEL ===
    print("\n[3/5] Building model...")
    model = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.4)

    # Dynamic pos_weight — ekibin yontemi
    n_neg     = (y_train == 0).sum()
    n_pos     = (y_train == 1).sum()
    pos_w     = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"  Architecture: {X_train.shape[1]} -> 64 (BN) -> 32 (BN) -> 1")
    print(f"  pos_weight: {pos_w.item():.3f} (neg={n_neg}, pos={n_pos})")
    print(f"  Optimizer: Adam (lr=1e-3, weight_decay=1e-4)")
    print(f"  Loss: BCEWithLogitsLoss")

    # === 4. TRAINING — EARLY STOPPING ===
    print("\n[4/5] Training with early stopping (patience=10, max=100)...")

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
            val_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
        val_auc = roc_auc_score(y_val, val_proba)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            counter       = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"| best epoch={best_epoch} "
                      f"| best val_loss={best_val_loss:.4f} "
                      f"| val_auc={val_auc:.4f}")
                break

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} "
                  f"| val_auc={val_auc:.4f}"
                  f"{'  *' if counter == 0 else ''}")

    model.load_state_dict(best_state)

    # === 5. EVALUATION ===
    print("\n[5/5] Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred_proba = torch.sigmoid(model(X_test_t)).numpy().flatten()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_pred_proba)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Recall:    {recall:.4f}  (default class detection rate)")
    print(f"  Precision: {precision:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Good (0)", "Bad (1)"]))

    print("=" * 60)
    if auc >= 0.75:
        print(f"AUC-ROC = {auc:.4f} -- Proposal target (>=0.75) ACHIEVED!")
    elif auc >= 0.70:
        print(f"AUC-ROC = {auc:.4f} -- Close to target.")
    else:
        print(f"AUC-ROC = {auc:.4f} -- Below target.")
    print("=" * 60)

    return model, auc, recall, precision, X_test_t, y_test


if __name__ == "__main__":
    import traceback
    try:
        model, auc, recall, precision, X_test_t, y_test = train_and_evaluate()
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()