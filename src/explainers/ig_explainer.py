import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare
from src.models.fnn import FNN


def run_ig_explanation(save_plots=True):
    print("=" * 60)
    print("INTEGRATED GRADIENTS -- FNN on German Credit")
    print("=" * 60)

    # === 1. MODELİ EĞİT ===
    print("\n[1/4] Training FNN model...")
    SEED = 21
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, class_weights = prepare(
        "german_credit", "fnn", random_state=SEED
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    sample_weights = torch.tensor(
        [class_weights[int(y)] for y in y_train.values],
        dtype=torch.float32
    ).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t, sample_weights),
        batch_size=32, shuffle=True
    )

    model     = FNN(input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(150):
        model.train()
        for X_batch, y_batch, w_batch in train_loader:
            optimizer.zero_grad()
            loss = (criterion(model(X_batch), y_batch) * w_batch).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    print("  Model trained (seed=21, matches train_fnn.py).")

    # === 2. IG EXPLAINER ===
    print("\n[2/4] Setting up Integrated Gradients...")

    # Captum'un IntegratedGradients'i doğrudan PyTorch modeline bağlanır
    # Model çıktısı [n, 1] shape'inde — target=0 ile ilk (tek) çıktıyı seç
    ig = IntegratedGradients(model)
    print("  IntegratedGradients explainer ready.")

    # === 3. IG DEĞERLERİNİ HESAPLA ===
    print("\n[3/4] Computing IG attributions...")

    # Baseline: sıfır vektörü (feature'lar olmasa ne olur?)
    baseline = torch.zeros_like(X_test_t)

    # Tüm test seti için attribution hesapla
    # n_steps=50: integral için 50 adım (daha fazla = daha doğru ama yavaş)
    attributions = ig.attribute(
        X_test_t,
        baselines=baseline,
        target=0,           # tek çıktı için 0
        n_steps=50,
        return_convergence_delta=False
    )

    # Shape: (n_test, n_features)
    attr_numpy = attributions.detach().numpy()
    print(f"  IG attributions shape: {attr_numpy.shape}")

    feature_names = X_train.columns.tolist()

    # === 4. GÖRSELLEŞTİR ===
    print("\n[4/4] Generating IG plots...")
    os.makedirs("reports/figures", exist_ok=True)

    # --- Global: Ortalama |IG| değerleri (SHAP bar chart'ına eşdeğer) ---
    mean_abs_attr = np.abs(attr_numpy).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_attr)[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_abs_attr[sorted_idx[::-1]],
        color="steelblue"
    )
    plt.xlabel("Mean |IG Attribution| (average impact on model output)")
    plt.title("Global Feature Importance (Integrated Gradients) -- FNN on German Credit")
    plt.tight_layout()

    if save_plots:
        plt.savefig("reports/figures/ig_global_bar.png", dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/ig_global_bar.png")
    plt.show()

    # --- Local: Tek bir "bad credit" için IG attribution bar ---
    bad_indices = [i for i, v in enumerate(y_test.values) if v == 1]
    if bad_indices:
        idx        = bad_indices[0]
        local_attr = attr_numpy[idx]

        # Feature'ları IG değerine göre sırala
        local_sorted_idx = np.argsort(np.abs(local_attr))[::-1]

        colors = ["red" if v > 0 else "blue" for v in local_attr[local_sorted_idx[::-1]]]

        plt.figure(figsize=(10, 8))
        plt.barh(
            [feature_names[i] for i in local_sorted_idx[::-1]],
            local_attr[local_sorted_idx[::-1]],
            color=colors
        )
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("IG Attribution (positive = increases default risk)")
        plt.title(f"Local IG Explanation -- Sample #{idx} (Bad Credit)")
        plt.tight_layout()

        if save_plots:
            plt.savefig("reports/figures/ig_local_bar.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: reports/figures/ig_local_bar.png")
        plt.show()

    # --- SHAP vs IG Karşılaştırması ---
    # İki metodun top-5 feature sıralamasını karşılaştır
    print("\n--- SHAP vs IG: Top-5 Feature Comparison ---")
    print(f"  IG Top-5:   {[feature_names[i] for i in sorted_idx[:5]]}")

    # SHAP değerlerini de yükle (daha önce kaydedildiyse)
    shap_path = "reports/figures/shap_global_bar.png"
    if os.path.exists(shap_path):
        print(f"  SHAP plot exists at {shap_path}")
        print("  (Run consistency analysis with OpenXAI metrics later)")

    print("\n" + "=" * 60)
    print("Integrated Gradients explanation complete.")
    print("Plots saved to reports/figures/")
    print("=" * 60)

    return attr_numpy, feature_names


if __name__ == "__main__":
    import traceback
    try:
        attributions, feature_names = run_ig_explanation(save_plots=True)
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()