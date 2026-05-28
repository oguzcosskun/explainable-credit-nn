import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


def run_ig_explanation(dataset="german_credit", save_plots=True):
    print("=" * 60)
    print(f"INTEGRATED GRADIENTS -- FNN on {dataset.upper()}")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/4] Loading model...")
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset)

    feature_names = X_train.columns.tolist()

    # === 2. IG EXPLAINER ===
    print("\n[2/4] Setting up Integrated Gradients...")
    ig = IntegratedGradients(model)
    print("  IntegratedGradients ready (zero baseline, 50 steps).")

    # === 3. IG ATTRIBUTION ===
    print("\n[3/4] Computing IG attributions...")
    N_IG      = min(200, len(X_test_t))
    X_ig      = X_test_t[:N_IG]
    baseline  = torch.zeros_like(X_ig)

    batch_size = 50
    ig_list    = []
    delta_list = []

    for i in range(0, N_IG, batch_size):
        batch  = X_ig[i:i+batch_size]
        base_b = baseline[i:i+batch_size]
        attrs, delta = ig.attribute(
            batch, baselines=base_b,
            target=0, n_steps=50,
            return_convergence_delta=True
        )
        ig_list.append(attrs.detach())
        delta_list.append(delta.detach())

    ig_attrs   = torch.cat(ig_list,    dim=0).numpy()
    all_deltas = torch.cat(delta_list, dim=0)
    mean_delta = all_deltas.abs().mean().item()

    print(f"  IG attributions shape: {ig_attrs.shape}")
    print(f"  Mean convergence delta: {mean_delta:.6f} "
          f"(closer to 0 = better approximation)")

    os.makedirs(f"reports/figures/{dataset}", exist_ok=True)

    # === 4. GORSELLER ===
    print("\n[4/4] Generating IG plots...")

    # Global bar
    mean_abs   = np.abs(ig_attrs).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_n      = min(20, len(feature_names))

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in sorted_idx[:top_n][::-1]],
        mean_abs[sorted_idx[:top_n][::-1]],
        color="steelblue"
    )
    plt.xlabel("Mean |IG Attribution|")
    plt.title(f"Global Feature Importance (IG) -- FNN on {dataset.upper()}")
    plt.tight_layout()
    if save_plots:
        path = f"reports/figures/{dataset}/ig_global_bar.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    # Local bar
    bad_indices = [i for i, v in enumerate(y_test.values[:N_IG]) if v == 1]
    if bad_indices:
        idx        = bad_indices[0]
        local_attr = ig_attrs[idx]
        local_sorted = np.argsort(np.abs(local_attr))[::-1]
        top_n_local  = min(20, len(feature_names))
        colors = ["red" if v > 0 else "blue"
                  for v in local_attr[local_sorted[:top_n_local][::-1]]]

        plt.figure(figsize=(10, 8))
        plt.barh(
            [feature_names[i] for i in local_sorted[:top_n_local][::-1]],
            local_attr[local_sorted[:top_n_local][::-1]],
            color=colors
        )
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.xlabel("IG Attribution")
        plt.title(f"Local IG -- Sample #{idx} -- {dataset.upper()}")
        plt.tight_layout()
        if save_plots:
            path = f"reports/figures/{dataset}/ig_local_bar.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close()

    # Top-5 ozet
    top5_idx = sorted_idx[:5]
    print(f"\n  IG Top-5 features ({dataset}):")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {feature_names[idx]} ({mean_abs[idx]:.4f})")

    print("\n" + "=" * 60)
    print(f"IG complete for {dataset}. Plots saved to reports/figures/{dataset}/")
    print("=" * 60)

    return ig_attrs, feature_names


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
            run_ig_explanation(ds, save_plots=True)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            traceback.print_exc()