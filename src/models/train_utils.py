import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare
from src.models.fnn import FNN

SEED = 42
CHECKPOINT_DIR = "models"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_trained_fnn(dataset="german_credit", seed=SEED, force_retrain=False):
    """
    Checkpoint varsa yukle, yoksa egit ve kaydet.
    force_retrain=True ile her zaman yeniden egitir.
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"fnn_best_{dataset}.pt")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_full, X_test, y_train_full, y_test, _ = prepare(
        dataset, "fnn", random_state=seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15, random_state=seed, stratify=y_train_full
    )
    X_train = X_train.reset_index(drop=True)
    X_val   = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val   = y_val.reset_index(drop=True)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val.values,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

    model = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.4)

    # Checkpoint varsa yukle
    if os.path.exists(ckpt_path) and not force_retrain:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            val_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"  Loaded checkpoint: {ckpt_path} | val_auc={val_auc:.4f} | "
              f"features={X_train.shape[1]}")
        return model, X_train, X_test, y_train, y_test, X_train_t, X_test_t

    # Checkpoint yoksa egit
    n_neg     = (y_train == 0).sum()
    n_pos     = (y_train == 1).sum()
    pos_w     = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32, shuffle=True
    )

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
                      f"(best epoch={best_epoch}, val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Kaydet
    torch.save(model.state_dict(), ckpt_path)

    with torch.no_grad():
        val_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
    val_auc = roc_auc_score(y_val, val_proba)
    print(f"  Model trained & saved | val_auc={val_auc:.4f} | "
          f"features={X_train.shape[1]} | checkpoint={ckpt_path}")

    return model, X_train, X_test, y_train, y_test, X_train_t, X_test_t