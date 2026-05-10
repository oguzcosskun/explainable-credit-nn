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


def get_trained_fnn(dataset="german_credit", seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_full, X_test, y_train_full, y_test, class_weights = prepare(
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

    sample_weights = torch.tensor(
        [class_weights[int(y)] for y in y_train.values],
        dtype=torch.float32
    ).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, sample_weights),
        batch_size=32, shuffle=True
    )

    model     = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_auc = 0.0
    best_state   = None

    for epoch in range(150):
        model.train()
        for X_batch, y_batch, w_batch in train_loader:
            optimizer.zero_grad()
            loss = (criterion(model(X_batch), y_batch) * w_batch).mean()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == 149:
            model.eval()
            with torch.no_grad():
                val_proba = torch.sigmoid(model(X_val_t)).numpy().flatten()
            val_auc = roc_auc_score(y_val, val_proba)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Model trained | best_val_auc={best_val_auc:.4f} | "
          f"features={X_train.shape[1]} | seed={seed}")

    return model, X_train, X_test, y_train, y_test, X_train_t, X_test_t