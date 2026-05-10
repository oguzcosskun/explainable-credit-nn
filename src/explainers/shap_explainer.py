import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn


def run_shap_explanation(save_plots=True):
    print("=" * 60)
    print("SHAP EXPLANATION -- FNN on German Credit (UCI)")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/4] Training FNN model...")
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn("german_credit")

    # === 2. EXPLAINER ===
    print("\n[2/4] Creating SHAP GradientExplainer...")
    torch.manual_seed(42)
    bg_idx  = torch.randperm(len(X_train_t))[:100]
    bg_data = X_train_t[bg_idx]

    # GradientExplainer — ekibin tercihi, FNN icin uygundur
    explainer = shap.GradientExplainer(model, bg_data)
    print("  GradientExplainer ready (100 background samples).")

    # === 3. SHAP DEGERLERI ===
    print("\n[3/4] Computing SHAP values...")
    N_SHAP      = min(200, len(X_test_t))
    X_shap      = X_test_t[:N_SHAP]
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    print(f"  SHAP values shape: {shap_values.shape}")

    feature_names = X_train.columns.tolist()
    os.makedirs("reports/figures", exist_ok=True)

    # === 4. GORSELLER ===
    print("\n[4/4] Generating SHAP plots...")

    # Global bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.values[:N_SHAP],
        feature_names=feature_names,
        plot_type="bar", show=False
    )
    plt.title("Global Feature Importance (SHAP) -- FNN on German Credit")
    plt.tight_layout()
    if save_plots:
        plt.savefig("reports/figures/shap_global_bar.png",
                    dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/shap_global_bar.png")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.values[:N_SHAP],
        feature_names=feature_names, show=False
    )
    plt.title("SHAP Beeswarm Plot -- FNN on German Credit")
    plt.tight_layout()
    if save_plots:
        plt.savefig("reports/figures/shap_beeswarm.png",
                    dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/shap_beeswarm.png")
    plt.close()

    # Local waterfall — ilk bad credit ornegi
    bad_indices = [i for i, v in enumerate(y_test.values[:N_SHAP]) if v == 1]
    if bad_indices:
        idx = bad_indices[0]
        base_val = float(np.mean(shap_values))
        shap_exp = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test.values[idx],
            feature_names=feature_names
        )
        plt.figure()
        shap.plots.waterfall(shap_exp, show=False)
        plt.title(f"Local SHAP -- Sample #{idx} (Bad Credit)")
        plt.tight_layout()
        if save_plots:
            plt.savefig("reports/figures/shap_local_waterfall.png",
                        dpi=150, bbox_inches="tight")
            print("  Saved: reports/figures/shap_local_waterfall.png")
        plt.close()

    # Top-5 ozet
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]
    print("\n  SHAP Top-5 features:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {feature_names[idx]} ({mean_abs[idx]:.4f})")

    print("\n" + "=" * 60)
    print("SHAP explanation complete. Plots saved to reports/figures/")
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
        traceback.print_exc()