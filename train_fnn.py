import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
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
    X_train_full, X_test, y_train_full, y_test, class_weights = prepare(
        "german_credit", "fnn", random_state=SEED
    )

    # Validation split from train (15% of total = ~18.75% of train)
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
    print(f"  Class weights: good={class_weights[0]:.3f}, bad={class_weights[1]:.3f}")

    # === 2. TENSORS ===
    print("\n[2/5] Converting to tensors...")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val.values,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

    # Sample weights for weighted loss
    sample_weights = torch.tensor(
        [class_weights[int(y)] for y in y_train.values],
        dtype=torch.float32
    ).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, sample_weights),
        batch_size=32, shuffle=True
    )

    # === 3. MODEL ===
    print("\n[3/5] Building model...")
    model     = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    print(f"  Architecture: {X_train.shape[1]} -> 64 (BN) -> 32 (BN) -> 1")
    print(f"  Optimizer: Adam (lr=1e-3)")
    print(f"  Loss: Weighted BCEWithLogitsLoss")

    # === 4. TRAINING ===
    print("\n[4/5] Training started (150 epochs)...")
    epochs       = 150
    best_val_auc = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches    = 0

        for X_batch, y_batch, w_batch in train_loader:
            optimizer.zero_grad()
            logits        = model(X_batch)
            loss          = (criterion(logits, y_batch) * w_batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches    += 1

        # Validation AUC her 10 epoch'ta
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
            val_auc = roc_auc_score(y_val, val_proba)

            # Best model kaydet
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"Loss={epoch_loss/batches:.4f}  Val AUC={val_auc:.4f}"
                  f"{'  *' if val_auc == best_val_auc else ''}")
            model.train()

    # Best modeli yükle
    model.load_state_dict(best_state)
    print(f"\n  Best Val AUC: {best_val_auc:.4f}")

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
        import traceback
        traceback.print_exc()