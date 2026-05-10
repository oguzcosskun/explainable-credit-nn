import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare
from src.models.fnn import FNN
from train_fnn import train_and_evaluate


def run_shap_explanation(save_plots=True):
    print("=" * 60)
    print("SHAP EXPLANATION -- FNN on German Credit")
    print("=" * 60)

    # === 1. MODELİ EĞİT ===
    print("\n[1/4] Training FNN model...")
    SEED = 21
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, class_weights = prepare(
        "german_credit", "fnn", random_state=SEED
    )

    # Modeli yeniden eğit (train_fnn'den bağımsız, burada kontrol bizde)
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

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
            pred          = model(X_batch)
            loss          = (criterion(pred, y_batch) * w_batch).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    print("  Model trained. AUC will match train_fnn.py (same seed).")

    # === 2. SHAP EXPLAINER OLUŞTUR ===
    print("\n[2/4] Creating SHAP explainer...")

    # DeepExplainer: PyTorch modellerine özgü, gradient tabanlı
    # Background: train setinden 100 örnek (SHAP baseline için)
    background = X_train_t[:100]
    explainer  = shap.DeepExplainer(model, background)
    print("  DeepExplainer created with 100 background samples.")

    # === 3. SHAP DEĞERLERİNİ HESAPLA ===
    print("\n[3/4] Computing SHAP values for test set...")

    # Test setinin tamamı için SHAP değerlerini hesapla
    shap_values = explainer.shap_values(X_test_t)

    # shap_values shape: (n_test, n_features)
    # Squeeze gerekebilir (model çıktısı [n, 1] ise)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Features: {X_test.shape[1]}")

    # === 4. GÖRSELLEŞTİR ===
    print("\n[4/4] Generating SHAP plots...")

    feature_names = X_train.columns.tolist()

    # --- Global: Summary Plot (Bar) ---
    # Her feature'ın ortalama |SHAP| değeri = global önem
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test.values,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title("Global Feature Importance (SHAP) -- FNN on German Credit")
    plt.tight_layout()

    if save_plots:
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig("reports/figures/shap_global_bar.png", dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/shap_global_bar.png")
    plt.show()

    # --- Global: Summary Plot (Beeswarm) ---
    # Her noktanın rengi feature değerini, x ekseni SHAP etkisini gösterir
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test.values,
        feature_names=feature_names,
        show=False
    )
    plt.title("SHAP Beeswarm Plot -- FNN on German Credit")
    plt.tight_layout()

    if save_plots:
        plt.savefig("reports/figures/shap_beeswarm.png", dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/shap_beeswarm.png")
    plt.show()

    # --- Local: Tek bir başvuru için waterfall ---
    # İlk "bad" krediyi bul ve açıkla
    bad_indices = [i for i, v in enumerate(y_test.values) if v == 1]
    if bad_indices:
        idx = bad_indices[0]
        print(f"\n  Local explanation for test sample #{idx} (Bad credit):")

        shap_explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=float(np.array(explainer.expected_value).flatten()[0]),
            data=X_test.values[idx],
            feature_names=feature_names
        )

        plt.figure()
        shap.plots.waterfall(shap_explanation, show=False)
        plt.title(f"Local SHAP Explanation -- Sample #{idx} (Bad Credit)")
        plt.tight_layout()

        if save_plots:
            plt.savefig("reports/figures/shap_local_waterfall.png", dpi=150, bbox_inches="tight")
            print("  Saved: reports/figures/shap_local_waterfall.png")
        plt.show()

    print("\n" + "=" * 60)
    print("SHAP explanation complete.")
    print("Plots saved to reports/figures/")
    print("=" * 60)

    return shap_values, feature_names


if __name__ == "__main__":
    import traceback
    try:
        shap_values, feature_names = run_shap_explanation(save_plots=True)
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()