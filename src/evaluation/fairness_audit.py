import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn
from src.explainers.dice_explainer import _get_dataset_config, _align_columns

import torch
import shap
import dice_ml
from dice_ml import Dice


def compute_ris_for_group(model, X_group_t, noise_std=0.01, n_neighbors=5,
                           seed=42):
    np.random.seed(seed)
    X_np = X_group_t.numpy()
    diffs = []
    for i in range(len(X_np)):
        x      = torch.tensor(X_np[i], dtype=torch.float32).unsqueeze(0)
        x.requires_grad_(True)
        out    = torch.sigmoid(model(x)).squeeze()
        g_orig = torch.autograd.grad(out, x)[0].detach().numpy().flatten()

        nd = []
        for _ in range(n_neighbors):
            noisy = np.clip(
                X_np[i] + np.random.normal(0, noise_std, X_np[i].shape),
                0, 1).astype(np.float32)
            xn = torch.tensor(noisy).unsqueeze(0)
            xn.requires_grad_(True)
            on = torch.sigmoid(model(xn)).squeeze()
            gn = torch.autograd.grad(on, xn)[0].detach().numpy().flatten()
            nd.append(np.linalg.norm(g_orig - gn) / (np.linalg.norm(g_orig) + 1e-8))
        diffs.append(np.mean(nd))
    return max(0.0, 1.0 - np.mean(diffs))


def compute_dice_proximity_for_group(model, X_train, df_group, cat_cols,
                                      num_cols, immutable, raw_cols,
                                      ref_cols, n_eval=30, seed=42):
    scaler = MinMaxScaler()
    df_g   = df_group.copy()
    df_g[num_cols] = scaler.fit_transform(df_g[num_cols])

    class W:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[raw_cols]
            else:
                X = pd.DataFrame(X, columns=raw_cols)
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
            X = _align_columns(X, ref_cols)
            Xt = torch.tensor(X.values, dtype=torch.float32)
            with torch.no_grad():
                p = torch.sigmoid(model(Xt)).numpy().flatten()
            return np.column_stack([1 - p, p])

    cont = num_cols if num_cols else raw_cols
    d    = dice_ml.Data(dataframe=df_g, continuous_features=cont,
                        outcome_name="target")
    m    = dice_ml.Model(model=W(), backend="sklearn")
    exp  = Dice(d, m, method="random")

    _, df_test = train_test_split(df_g, test_size=0.2, random_state=seed,
                                   stratify=df_g["target"])
    bad = df_test[df_test["target"] == 1].reset_index(drop=True)
    vary = [f for f in raw_cols if f not in immutable]

    prox_list = []
    valid_count = 0
    N = min(n_eval, len(bad))

    for i in range(N):
        query = bad.iloc[[i]][raw_cols]
        try:
            cf    = exp.generate_counterfactuals(
                query, total_CFs=1,
                desired_class="opposite",
                features_to_vary=vary)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                valid_count += 1
                if cat_cols:
                    q_enc  = pd.get_dummies(query[raw_cols], columns=cat_cols,
                                            drop_first=False)
                    cf_enc = pd.get_dummies(cf_df[raw_cols], columns=cat_cols,
                                            drop_first=False)
                else:
                    q_enc  = query[raw_cols].copy()
                    cf_enc = cf_df[raw_cols].copy()
                q_enc  = _align_columns(q_enc,  ref_cols)
                cf_enc = _align_columns(cf_enc, ref_cols)
                dist = np.abs(cf_enc.values - q_enc.values).mean()
                prox_list.append(dist)
        except Exception:
            pass

    validity  = valid_count / N if N > 0 else 0
    proximity = np.mean(prox_list) if prox_list else float("nan")
    return validity, proximity


def run_fairness_audit():
    results = []
    SEED    = 42

    # ══════════════════════════════════════════════════════════
    # GERMAN CREDIT — personal_status (gender proxy)
    # A91=male:divorced, A92=female:married/div, A93=male:single
    # Group: female (A92) vs male (A91+A93+A94)
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FAIRNESS AUDIT — German Credit (gender proxy)")
    print("="*60)

    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn("german_credit", seed=SEED)

    from src.preprocessing.pipeline import _load_german
    X_raw, y_raw = _load_german()
    X_raw["target"] = y_raw.values

    female_idx = X_raw[X_raw["personal_status"] == "A92"].index
    male_idx   = X_raw[X_raw["personal_status"].isin(["A91","A93","A94"])].index

    print(f"  Female group (A92): n={len(female_idx)}")
    print(f"  Male group (A91/A93/A94): n={len(male_idx)}")

    from src.preprocessing.pipeline import prepare
    X_full, X_test_f, y_full, y_test_f, _ = prepare(
        "german_credit", "fnn", random_state=SEED)

    # RIS per group — kullan test setindeki indexleri
    for group_name, group_idx in [("Female", female_idx),
                                   ("Male",   male_idx)]:
        # Test setinde bu gruba ait örnekler
        common = X_raw.loc[group_idx].index
        mask   = [i for i, idx in enumerate(
            X_raw.index[:len(X_test_f)]) if idx in common]
        if len(mask) < 5:
            print(f"  {group_name}: too few samples, skip")
            continue
        X_grp_t = X_test_t[mask[:min(50, len(mask))]]
        ris = compute_ris_for_group(model, X_grp_t)
        print(f"  RIS [{group_name}]: {ris:.4f}")

        df_group = X_raw.loc[group_idx].copy()
        df_dice, cat_cols_d, num_cols_d, immutable_d = \
            _get_dataset_config("german_credit")
        raw_cols = [c for c in df_dice.columns if c != "target"]
        ref_cols = X_train.columns.tolist()

        val, prox = compute_dice_proximity_for_group(
            model, X_train, df_group, cat_cols_d, num_cols_d,
            immutable_d, raw_cols, ref_cols, n_eval=20, seed=SEED)
        print(f"  DiCE Validity [{group_name}]: {val:.2%}")
        print(f"  DiCE Proximity [{group_name}]: {prox:.4f}")

        results.append({
            "dataset": "german_credit",
            "group_attr": "personal_status",
            "group": group_name,
            "ris": round(ris, 4),
            "dice_validity": round(val, 4),
            "dice_proximity": round(prox, 4) if not np.isnan(prox) else None
        })

    # ══════════════════════════════════════════════════════════
    # ADULT INCOME — sex (Male vs Female)
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FAIRNESS AUDIT — Adult Income (sex)")
    print("="*60)

    model_a, X_train_a, X_test_a, y_train_a, y_test_a, X_train_t_a, X_test_t_a = \
        get_trained_fnn("adult", seed=SEED)

    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/adult/adult.data")
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]
    df_adult = pd.read_csv(url, header=None, names=col_names,
                            na_values=" ?", skipinitialspace=True).dropna()
    df_adult["target"] = (
        df_adult["income"].str.strip().str.startswith(">50K")).astype(int)

    for group_name, sex_val in [("Male", "Male"), ("Female", "Female")]:
        grp = df_adult[df_adult["sex"].str.strip() == sex_val]
        print(f"  {group_name} group: n={len(grp)}")

        df_dice_a, cat_cols_a, num_cols_a, immutable_a = \
            _get_dataset_config("adult")
        raw_cols_a = [c for c in df_dice_a.columns if c != "target"]
        ref_cols_a = X_train_a.columns.tolist()

        grp_dice = grp[raw_cols_a + ["target"]].copy() if "target" in grp.columns \
            else grp.copy()
        grp_dice["target"] = (
            grp_dice["income"].str.strip().str.startswith(">50K")).astype(int) \
            if "income" in grp_dice.columns else grp["target"].values

        if "income" in grp_dice.columns:
            grp_dice = grp_dice.drop(columns=["income"])

        val_a, prox_a = compute_dice_proximity_for_group(
            model_a, X_train_a, grp_dice, cat_cols_a, num_cols_a,
            immutable_a, raw_cols_a, ref_cols_a, n_eval=20, seed=SEED)
        print(f"  DiCE Validity [{group_name}]: {val_a:.2%}")
        print(f"  DiCE Proximity [{group_name}]: {prox_a:.4f}")

        results.append({
            "dataset": "adult",
            "group_attr": "sex",
            "group": group_name,
            "ris": None,
            "dice_validity": round(val_a, 4),
            "dice_proximity": round(prox_a, 4) if not np.isnan(prox_a) else None
        })

    # === CSV KAYDET ===
    os.makedirs("reports", exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv("reports/fairness_audit.csv", index=False)
    print(f"\nSaved → reports/fairness_audit.csv")
    print(df_results.to_string(index=False))


if __name__ == "__main__":
    run_fairness_audit()