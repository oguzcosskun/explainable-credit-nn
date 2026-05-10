import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


def run_ig_explanation(save_plots=True):
    print("=" * 60)
    print("INTEGRATED GRADIENTS -- FNN on German Credit (UCI)")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/4] Training FNN model...")
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn("german_credit")

    feature_names = X_train.columns.tolist()

    # === 2. IG EXPLAINER ===
    print("\n[2/4] Setting up Integrated Gradients...")
    ig       = IntegratedGradients(model)
    print("  IntegratedGradients ready (zero baseline, 50 steps).")

    # === 3. IG ATTRIBUTION ===
    print("\n[3/4] Computing IG attributions...")
    N_IG     = min(200, len(X_test_t))
    X_ig     = X_test_t[:N_IG]
    baseline = torch.zeros_like(X_ig)

    batch_size = 50
    ig_list    = []
    for i in range(0, N_IG, batch_size):
        batch    = X_ig[i:i+batch_size]
        base_b   = baseline[i:i+batch_size]
        attrs    = ig.attribute(batch, baselines=base_b,
                                target=0, n_steps=50)
        ig_list.append(attrs.detach())

    ig_attrs  = torch.cat(ig_list, dim=0).numpy()
    print(f"  IG attributions shape: {ig_attrs.shape}")

    os.makedirs("reports/figures", exist_ok=True)

    # === 4. GORSELLER ===
    print("\n[4/4] Generating IG plots...")

    # Global bar
    mean_abs     = np.abs(ig_attrs).mean(axis=0)
    sorted_idx   = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in sorted_idx[:20][::-1]],
        mean_abs[sorted_idx[:20][::-1]],
        color="steelblue"
    )
    plt.xlabel("Mean |IG Attribution|")
    plt.title("Global Feature Importance (IG) -- FNN on German Credit")
    plt.tight_layout()
    if save_plots:
        plt.savefig("reports/figures/ig_global_bar.png",
                    dpi=150, bbox_inches="tight")
        print("  Saved: reports/figures/ig_global_bar.png")
    plt.close()

    # Local bar — ilk bad credit
    bad_indices = [i for i, v in enumerate(y_test.values[:N_IG]) if v == 1]
    if bad_indices:
        idx        = bad_indices[0]
        local_attr = ig_attrs[idx]
        local_sorted = np.argsort(np.abs(local_attr))[::-1]
        colors = ["red" if v > 0 else "blue"
                  for v in local_attr[local_sorted[:20][::-1]]]

        plt.figure(figsize=(10, 8))
        plt.barh(
            [feature_names[i] for i in local_sorted[:20][::-1]],
            local_attr[local_sorted[:20][::-1]],
            color=colors
        )
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("IG Attribution")
        plt.title(f"Local IG -- Sample #{idx} (Bad Credit)")
        plt.tight_layout()
        if save_plots:
            plt.savefig("reports/figures/ig_local_bar.png",
                        dpi=150, bbox_inches="tight")
            print("  Saved: reports/figures/ig_local_bar.png")
        plt.close()

    # Top-5 ozet + SHAP karsilastirma
    top5_idx = sorted_idx[:5]
    print("\n  IG Top-5 features:")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {feature_names[idx]} ({mean_abs[idx]:.4f})")

    print("\n" + "=" * 60)
    print("IG explanation complete. Plots saved to reports/figures/")
    print("=" * 60)

    return ig_attrs, feature_names


if __name__ == "__main__":
    import traceback
    try:
        ig_attrs, feature_names = run_ig_explanation(save_plots=True)
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()