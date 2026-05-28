import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import shap
from captum.attr import IntegratedGradients


def get_input_gradient(model, x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
    x_t.requires_grad_(True)
    out   = torch.sigmoid(model(x_t)).squeeze()
    grads = torch.autograd.grad(out, x_t)[0]
    return grads.detach().numpy().flatten()


def compute_faithfulness(model, inputs_t, attributions, k=5):
    scores = []
    with torch.no_grad():
        orig_preds = torch.sigmoid(model(inputs_t)).numpy().flatten()
    for i in range(len(inputs_t)):
        topk_idx  = np.argsort(np.abs(attributions[i]))[::-1][:k].copy()
        perturbed = inputs_t[i].clone()
        perturbed[topk_idx] = 0.0
        with torch.no_grad():
            pert_pred = torch.sigmoid(model(perturbed.unsqueeze(0))).item()
        scores.append(abs(orig_preds[i] - pert_pred))
    return np.array(scores)


def compute_ris(model, inputs_t, n_neighbors=5, noise_std=0.01, seed=42):
    np.random.seed(seed)
    X_np  = inputs_t.numpy()
    diffs = []
    for i in range(len(X_np)):
        orig_attr      = get_input_gradient(model, X_np[i])
        neighbor_diffs = []
        for _ in range(n_neighbors):
            noisy      = np.clip(
                X_np[i] + np.random.normal(0, noise_std, X_np[i].shape),
                0, 1).astype(np.float32)
            neigh_attr = get_input_gradient(model, noisy)
            denom      = np.linalg.norm(orig_attr) + 1e-8
            neighbor_diffs.append(
                np.linalg.norm(orig_attr - neigh_attr) / denom
            )
        diffs.append(np.mean(neighbor_diffs))
    ris_score = max(0.0, 1.0 - np.mean(diffs))
    return ris_score, np.std(diffs)


def run_metrics(dataset="german_credit"):
    print("=" * 60)
    print(f"XAI METRICS -- FNN on {dataset.upper()}")
    print("Faithfulness | Stability (RIS) | Consistency")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/4] Loading model...")
    SEED = 42
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset, seed=SEED)

    n_eval   = min(100, len(X_test_t))
    X_eval_t = X_test_t[:n_eval]

    # === 2. ATTRIBUTIONS ===
    print("\n[2/4] Computing SHAP and IG attributions...")
    torch.manual_seed(SEED)
    bg_idx   = torch.randperm(len(X_train_t))[:100]
    bg_data  = X_train_t[bg_idx]
    shap_exp  = shap.GradientExplainer(model, bg_data)
    shap_vals = shap_exp.shap_values(X_eval_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]
    print(f"  SHAP shape: {shap_vals.shape}")

    ig       = IntegratedGradients(model)
    baseline = torch.zeros_like(X_eval_t)
    ig_list  = []
    for i in range(0, n_eval, 50):
        attrs = ig.attribute(X_eval_t[i:i+50],
                             baselines=baseline[i:i+50],
                             target=0, n_steps=50)
        ig_list.append(attrs.detach())
    ig_vals = torch.cat(ig_list, dim=0).numpy()
    print(f"  IG shape:   {ig_vals.shape}")

    # === 3. FAITHFULNESS ===
    print("\n[3/4] Computing Faithfulness (k=5, k=10)...")
    faith_results = {}
    for k in [5, 10]:
        f_shap = compute_faithfulness(model, X_eval_t, shap_vals, k=k)
        f_ig   = compute_faithfulness(model, X_eval_t, ig_vals,   k=k)
        print(f"  K={k} | SHAP: {np.mean(f_shap):.4f} +/- {np.std(f_shap):.4f}"
              f" | IG: {np.mean(f_ig):.4f} +/- {np.std(f_ig):.4f}")
        faith_results[k] = (np.mean(f_shap), np.mean(f_ig))

    # === 4. STABILITY ===
    print("\n[4/4] Computing Stability (RIS)...")
# Feature sayisina gore noise_std ayarla
    n_features = X_eval_t.shape[1]
    noise_std  = 0.01 if n_features >= 20 else 0.001

    ris_score, ris_std = compute_ris(model, X_eval_t,
                                      n_neighbors=5, noise_std=noise_std, seed=SEED)
    print(f"  RIS: {ris_score:.4f} +/- {ris_std:.4f}  (target > 0.80)"
          f"  [noise_std={noise_std}, features={n_features}]")

    # === CONSISTENCY ===
    print("\n--- Consistency: SHAP vs IG ---")
    feat_names = X_train.columns.tolist()
    consistency_results = {}

    for k in [5, 10]:
        fa_scores  = []
        sra_scores = []
        for s, g in zip(shap_vals, ig_vals):
            top_s = set(np.argsort(np.abs(s))[-k:])
            top_g = set(np.argsort(np.abs(g))[-k:])
            fa_scores.append(len(top_s & top_g) / k)

            rank_s = np.argsort(np.abs(s))[-k:][::-1]
            rank_g = np.argsort(np.abs(g))[-k:][::-1]
            sign_s = np.sign(s[rank_s])
            sign_g = np.sign(g[rank_g])
            match  = sum(
                1 for i, feat in enumerate(rank_s)
                if feat in rank_g and
                sign_s[i] == sign_g[list(rank_g).index(feat)]
            )
            sra_scores.append(match / k)

        print(f"  K={k} | FA: {np.mean(fa_scores):.3f} +/- {np.std(fa_scores):.3f}"
              f" | SRA: {np.mean(sra_scores):.3f} +/- {np.std(sra_scores):.3f}")
        consistency_results[k] = (np.mean(fa_scores), np.mean(sra_scores))

    # Global top-5
    shap_mean = np.abs(shap_vals).mean(axis=0)
    ig_mean   = np.abs(ig_vals).mean(axis=0)
    shap_top5 = set(np.argsort(shap_mean)[::-1][:5])
    ig_top5   = set(np.argsort(ig_mean)[::-1][:5])
    overlap   = len(shap_top5 & ig_top5)
    print(f"\n  Global Top-5 overlap: {overlap}/5")
    print(f"  SHAP: {[feat_names[i] for i in sorted(shap_top5, key=lambda x: shap_mean[x], reverse=True)]}")
    print(f"  IG:   {[feat_names[i] for i in sorted(ig_top5, key=lambda x: ig_mean[x], reverse=True)]}")

    # === OZET ===
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"  Faithfulness K=5  | SHAP: {faith_results[5][0]:.4f}"
          f"  | IG: {faith_results[5][1]:.4f}")
    print(f"  Stability RIS     | {ris_score:.4f} +/- {ris_std:.4f}"
          f"  (target > 0.80)")
    print(f"  Consistency K=5   | FA: {consistency_results[5][0]:.3f}"
          f"  | SRA: {consistency_results[5][1]:.3f}")
    print("=" * 60)

    # === CSV KAYDET ===
    os.makedirs("reports", exist_ok=True)
    row = {
        "dataset":          dataset,
        "faith_shap_k5":    round(faith_results[5][0], 4),
        "faith_ig_k5":      round(faith_results[5][1], 4),
        "faith_shap_k10":   round(faith_results[10][0], 4),
        "faith_ig_k10":     round(faith_results[10][1], 4),
        "ris":              round(ris_score, 4),
        "ris_std":          round(ris_std, 4),
        "fa_k5":            round(consistency_results[5][0], 4),
        "sra_k5":           round(consistency_results[5][1], 4),
        "fa_k10":           round(consistency_results[10][0], 4),
        "sra_k10":          round(consistency_results[10][1], 4),
        "top5_overlap":     overlap,
    }
    csv_path = "reports/openxai_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df["dataset"] != dataset]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    return row


if __name__ == "__main__":
    import traceback
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="german_credit",
                        choices=["german_credit", "heloc", "adult", "gmsc", "all"])
    args = parser.parse_args()

    if args.dataset == "all":
        datasets = ["german_credit", "heloc", "adult", "gmsc"]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        try:
            run_metrics(ds)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            traceback.print_exc()