import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, classification_report

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.fnn import FNN

# Reproducibility
SEED = 21
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_and_evaluate():
    print("=" * 60)
    print("FNN BASELINE -- German Credit Dataset")
    print("=" * 60)

    print("\n[1/5] Loading and preparing data...")
    X_train, X_test, y_train, y_test, class_weights = prepare("german_credit", "fnn", random_state=SEED)
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Class weights: good={class_weights[0]:.3f}, bad={class_weights[1]:.3f}")

    print("\n[2/5] Converting to tensors...")
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

    sample_weights = torch.tensor(
        [class_weights[int(y)] for y in y_train.values],
        dtype=torch.float32
    ).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t, sample_weights)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

    print("\n[3/5] Building model...")
    model     = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss(reduction='none')
    print(f"  Architecture: {X_train.shape[1]} -> 64 -> 32 -> 1")
    print(f"  Optimizer: Adam (lr=5e-4)")
    print(f"  Loss: Weighted Binary Cross-Entropy")

    print("\n[4/5] Training started (150 epochs)...")
    epochs = 150

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches    = 0

        for X_batch, y_batch, w_batch in train_loader:
            optimizer.zero_grad()
            pred               = model(X_batch)
            loss_per_sample    = criterion(pred, y_batch)
            weighted_loss      = (loss_per_sample * w_batch).mean()
            weighted_loss.backward()
            optimizer.step()
            epoch_loss += weighted_loss.item()
            batches    += 1

        avg_loss = epoch_loss / batches

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: Loss = {avg_loss:.4f}")

    print("\n[5/5] Evaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_t).numpy().flatten()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

    auc       = roc_auc_score(y_test, y_pred_proba)
    recall    = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Recall:    {recall:.4f}  (default class detection rate)")
    print(f"  Precision: {precision:.4f}  (of predicted defaults, how many are real)")
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Good (0)", "Bad (1)"]))

    print("=" * 60)
    if auc >= 0.75:
        print(f"AUC-ROC = {auc:.4f} -- Proposal target (>=0.75) ACHIEVED!")
    elif auc >= 0.70:
        print(f"AUC-ROC = {auc:.4f} -- Close to target, can be improved with tuning.")
    else:
        print(f"AUC-ROC = {auc:.4f} -- Below target, model improvement needed.")
    print("=" * 60)

    return auc, recall, precision

if __name__ == "__main__":
    import traceback
    try:
        auc, recall, precision = train_and_evaluate()
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()