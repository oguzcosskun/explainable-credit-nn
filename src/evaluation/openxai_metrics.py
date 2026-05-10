import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare
from src.models.fnn import FNN

import shap
from captum.attr import IntegratedGradients


def get_trained_model(X_train, y_train, class_weights, seed=21):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
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
    return model


def compute_faithfulness(model, inputs_t, attributions, k=5, n_perturb=50, seed=21):
    """
    Faithfulness: top-k feature'lari sifirla, tahmin ne kadar degisiyor?
    Yuksek faithfulness = aciklama modelin gercek kararini yansitiyordur.
    OpenXAI'in eval_pred_faithfulness ile ayni mantik.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    scores = []
    with torch.no_grad():
        orig_preds = model(inputs_t).numpy().flatten()

    for i in range(len(inputs_t)):
        attr    = attributions[i]
        inp     = inputs_t[i].clone()

        # Top-k en onemli feature'larin indekslerini bul
        topk_idx = np.argsort(np.abs(attr))[::-1][:k].copy()

        # Bu feature'lari sifirla (perturb)
        perturbed = inp.clone()
        perturbed[topk_idx] = 0.0

        with torch.no_grad():
            perturbed_pred = model(perturbed.unsqueeze(0)).item()

        # Tahmin farkini kaydet
        scores.append(abs(orig_preds[i] - perturbed_pred))

    return np.array(scores)


def compute_stability(model, inputs_t, attributions, n_neighbors=10,
                      noise_std=0.05, seed=21):
    """
    Stability (Relative Input Stability):
    Benzer inputlar benzer aciklamalar aliyor mu?
    Dusuk deger = stabil (aciklamalar cok degismiyordur).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    scores = []

    for i in range(len(inputs_t)):
        inp  = inputs_t[i].clone().numpy()
        attr = attributions[i]

        # Kucuk gaussian gurultu ekleyerek komsular uret
        neighbor_stabilities = []

        for _ in range(n_neighbors):
            noise    = np.random.normal(0, noise_std, size=inp.shape).astype(np.float32)
            neighbor = np.clip(inp + noise, 0, 1)
            neighbor_t = torch.tensor(neighbor).unsqueeze(0)

            # Komsu icin SHAP/IG hesapla — yaklasim olarak gradyani kullan
            neighbor_t.requires_grad_(True)
            pred = model(neighbor_t)
            pred.backward()
            neighbor_attr = neighbor_t.grad.squeeze().detach().numpy()
            neighbor_t.requires_grad_(False)

            # Aciklama farki
            attr_diff = np.linalg.norm(attr - neighbor_attr)
            inp_diff  = np.linalg.norm(inp - neighbor)

            if inp_diff > 1e-8:
                neighbor_stabilities.append(attr_diff / inp_diff)

        if neighbor_stabilities:
            scores.append(np.mean(neighbor_stabilities))

    return np.array(scores)


def run_metrics():
    print("=" * 60)
    print("XAI METRICS -- FNN on German Credit")
    print("(OpenXAI-inspired: Faithfulness, Stability, Consistency)")
    print("=" * 60)

    # === 1. VERİ VE MODEL ===
    print("\n[1/4] Preparing data and training model...")
    SEED = 21
    X_train, X_test, y_train, y_test, class_weights = prepare(
        "german_credit", "fnn", random_state=SEED
    )
    model = get_trained_model(X_train, y_train, class_weights, seed=SEED)
    print("  Model trained (seed=21).")

    X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)

    # Ilk 100 test ornegi uzerinde degerlendir
    n_eval   = 100
    X_eval_t = X_test_t[:n_eval]

    # === 2. ATTRIBUTION'LARI HESAPLA ===
    print("\n[2/4] Computing SHAP and IG attributions...")

    # SHAP
    background     = X_train_t[:100]
    shap_explainer = shap.DeepExplainer(model, background)
    shap_vals      = shap_explainer.shap_values(X_eval_t)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]
    print(f"  SHAP attributions shape: {shap_vals.shape}")

    # IG
    ig       = IntegratedGradients(model)
    baseline = torch.zeros_like(X_eval_t)
    ig_vals  = ig.attribute(X_eval_t, baselines=baseline,
                             target=0, n_steps=50).detach().numpy()
    print(f"  IG attributions shape:   {ig_vals.shape}")

    # === 3. FAITHFULNESS ===
    print("\n[3/4] Computing Faithfulness (top-5 feature perturbation)...")

    faith_shap = compute_faithfulness(model, X_eval_t, shap_vals, k=5)
    faith_ig   = compute_faithfulness(model, X_eval_t, ig_vals,   k=5)

    print(f"  Faithfulness (SHAP): {np.mean(faith_shap):.4f} "
          f"(mean prediction drop when top-5 features zeroed)")
    print(f"  Faithfulness (IG):   {np.mean(faith_ig):.4f}")

    # === 4. STABILITY ===
    print("\n[4/4] Computing Stability (relative input stability)...")

    stab_shap = compute_stability(model, X_eval_t, shap_vals,
                                   n_neighbors=10, noise_std=0.05)
    stab_ig   = compute_stability(model, X_eval_t, ig_vals,
                                   n_neighbors=10, noise_std=0.05)

    print(f"  Stability (SHAP): {np.mean(stab_shap):.4f} "
          f"(lower = more stable explanations)")
    print(f"  Stability (IG):   {np.mean(stab_ig):.4f}")

    # === CONSISTENCY ===
    print("\n--- Consistency: SHAP vs IG Top-5 Feature Agreement ---")
    feat_names = X_train.columns.tolist()
    shap_mean  = np.abs(shap_vals).mean(axis=0)
    ig_mean    = np.abs(ig_vals).mean(axis=0)
    shap_top5  = set(np.argsort(shap_mean)[::-1][:5])
    ig_top5    = set(np.argsort(ig_mean)[::-1][:5])
    overlap    = len(shap_top5 & ig_top5)
    consistency = overlap / 5

    print(f"  SHAP Top-5: {[feat_names[i] for i in sorted(shap_top5, key=lambda x: shap_mean[x], reverse=True)]}")
    print(f"  IG Top-5:   {[feat_names[i] for i in sorted(ig_top5, key=lambda x: ig_mean[x], reverse=True)]}")
    print(f"  Overlap: {overlap}/5 features")
    print(f"  Consistency Score: {consistency:.2f}")

    # === ÖZET ===
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"  Faithfulness (SHAP): {np.mean(faith_shap):.4f}")
    print(f"  Faithfulness (IG):   {np.mean(faith_ig):.4f}")
    print(f"  Stability    (SHAP): {np.mean(stab_shap):.4f}  (lower = better)")
    print(f"  Stability    (IG):   {np.mean(stab_ig):.4f}  (lower = better)")
    print(f"  Consistency  Score:  {consistency:.2f}  ({overlap}/5 features match)")
    print("=" * 60)

    return {
        "faithfulness_shap": float(np.mean(faith_shap)),
        "faithfulness_ig":   float(np.mean(faith_ig)),
        "stability_shap":    float(np.mean(stab_shap)),
        "stability_ig":      float(np.mean(stab_ig)),
        "consistency":       consistency
    }


if __name__ == "__main__":
    import traceback
    try:
        results = run_metrics()
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("\nFull traceback:")
        traceback.print_exc()