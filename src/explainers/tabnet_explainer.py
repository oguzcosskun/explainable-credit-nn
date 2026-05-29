import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.abspath("."))
from src.models.tabnet_model import get_trained_tabnet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_tabnet_explanation(dataset="german_credit", save_plots=True):
    print("=" * 60)
    print(f"TABNET ATTENTION MASKS -- {dataset.upper()}")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/3] Loading model...")
    model, X_train, X_test, y_train, y_test = get_trained_tabnet(dataset)
    feature_names = X_train.columns.tolist()

    # === 2. ATTENTION MASKS ===
    print("\n[2/3] Extracting attention masks...")

    # explain() — TabNet'in dahili attention mask'larini döndürür
    # Her step için feature importance matrisi
    explain_matrix, masks = model.explain(X_test.values)

    # explain_matrix: (n_samples, n_features) — global feature importance
    # masks: list of (n_samples, n_features) — her step için ayrı mask

    print(f"  Explain matrix shape: {explain_matrix.shape}")
    print(f"  Number of steps: {len(masks)}")

    os.makedirs(f"reports/figures/{dataset}", exist_ok=True)

    # === 3. GORSELLER ===
    print("\n[3/3] Generating plots...")

    # Global feature importance — tüm test seti ortalaması
    mean_importance = explain_matrix.mean(axis=0)
    sorted_idx      = np.argsort(mean_importance)[::-1]
    top_n           = min(20, len(feature_names))

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in sorted_idx[:top_n][::-1]],
        mean_importance[sorted_idx[:top_n][::-1]],
        color="darkorange"
    )
    plt.xlabel("Mean Attention Weight")
    plt.title(f"TabNet Global Feature Importance -- {dataset.upper()}")
    plt.tight_layout()
    if save_plots:
        path = f"reports/figures/{dataset}/tabnet_global_importance.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()

    # Step-by-step masks
    valid_masks = [m for m in masks if hasattr(m, 'mean')]
    if valid_masks:
        n_steps = len(valid_masks)
        fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 8))
        if n_steps == 1:
            axes = [axes]

        for step_idx, mask in enumerate(valid_masks):
            mean_mask = mask.mean(axis=0)
            sorted_m  = np.argsort(mean_mask)[::-1]
            top_m     = min(10, len(feature_names))
            axes[step_idx].barh(
                [feature_names[i] for i in sorted_m[:top_m][::-1]],
                mean_mask[sorted_m[:top_m][::-1]],
                color=f"C{step_idx}"
            )
            axes[step_idx].set_title(f"Step {step_idx + 1}")
            axes[step_idx].set_xlabel("Attention Weight")

        plt.suptitle(f"TabNet Step-wise Attention Masks -- {dataset.upper()}")
        plt.tight_layout()
        if save_plots:
            path = f"reports/figures/{dataset}/tabnet_step_masks.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")
        plt.close()
    else:
        print("  Step masks not available for this model version.")

    # Top-5 ozet
    print(f"\n  TabNet Top-5 features ({dataset}):")
    for i, idx in enumerate(sorted_idx[:5]):
        print(f"    {i+1}. {feature_names[idx]} ({mean_importance[idx]:.4f})")

    print("\n" + "=" * 60)
    print(f"TabNet explanation complete for {dataset}.")
    print("=" * 60)

    return explain_matrix, masks, feature_names


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
            run_tabnet_explanation(ds, save_plots=True)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            traceback.print_exc()