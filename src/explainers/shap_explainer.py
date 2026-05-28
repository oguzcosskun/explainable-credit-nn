import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_shap_explanation(dataset="german_credit", save_plots=True):
    print("=" * 60)
    print(f"SHAP EXPLANATION -- FNN on {dataset.upper()}")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/4] Loading model...")
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset)

    feature_names = X_train.columns.tolist()

    # === 2. EXPLAINER ===
    print("\n[2/4] Creating SHAP GradientExplainer...")
    torch.manual_seed(42)
    bg_idx  = torch.randperm(len(X_train_t))[:100]
    bg_data = X_train_t[bg_idx]
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

    os.makedirs(f"reports/figures/{dataset}", exist_ok=True)

    # === 4. GORSELLER ===
    print("\n[4/4] Generating SHAP plots...")

    # Global bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.values[:N_SHAP],
        feature_names=feature_names,
        plot_type="bar", show=False
    )
    plt.title(f"Global Feature Importance (SHAP) -- FNN on {dataset.upper()}")
    plt.tight_layout()
    if save_plots:
        path = f"reports/figures/{dataset}/shap_global_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    # Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test.values[:N_SHAP],
        feature_names=feature_names, show=False
    )
    plt.title(f"SHAP Beeswarm -- FNN on {dataset.upper()}")
    plt.tight_layout()
    if save_plots:
        path = f"reports/figures/{dataset}/shap_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    # Local waterfall
    bad_indices = [i for i, v in enumerate(y_test.values[:N_SHAP]) if v == 1]
    if bad_indices:
        idx      = bad_indices[0]
        base_val = float(np.mean(shap_values))
        shap_exp = shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_test.values[idx],
            feature_names=feature_names
        )
        plt.figure()
        shap.plots.waterfall(shap_exp, show=False)
        plt.title(f"Local SHAP -- Sample #{idx} (Bad Credit) -- {dataset.upper()}")
        plt.tight_layout()
        if save_plots:
            path = f"reports/figures/{dataset}/shap_local_waterfall.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close()

    # Top-5 ozet
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs)[::-1][:5]
    print(f"\n  SHAP Top-5 features ({dataset}):")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {feature_names[idx]} ({mean_abs[idx]:.4f})")

    print("\n" + "=" * 60)
    print(f"SHAP complete for {dataset}. Plots saved to reports/figures/{dataset}/")
    print("=" * 60)

    return shap_values, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="german_credit",
                        choices=["german_credit", "heloc", "adult", "gmsc", "all"])
    args = parser.parse_args()

    datasets = ["german_credit", "heloc", "adult", "gmsc"] \
               if args.dataset == "all" else [args.dataset]

    import traceback
    for ds in datasets:
        try:
            run_shap_explanation(ds, save_plots=True)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            traceback.print_exc()