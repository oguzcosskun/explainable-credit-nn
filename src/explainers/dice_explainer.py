import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import dice_ml
from dice_ml import Dice

UCI_LABELS = {
    "A11": "< 0 DM", "A12": "0-200 DM", "A13": "> 200 DM",
    "A14": "no checking account",
    "A30": "no credits taken", "A31": "all paid back duly",
    "A32": "existing credits paid duly", "A33": "delay in paying",
    "A34": "critical account",
    "A40": "car (new)", "A41": "car (used)", "A42": "furniture",
    "A43": "radio/TV", "A44": "domestic appliances", "A45": "repairs",
    "A46": "education", "A48": "retraining", "A49": "business",
    "A410": "others",
    "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
    "A64": "> 1000 DM", "A65": "unknown/no savings",
    "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
    "A74": "4-7 years", "A75": ">= 7 years",
    "A91": "male: divorced", "A92": "female: married/div",
    "A93": "male: single", "A94": "male: married",
    "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    "A121": "real estate", "A122": "building society savings",
    "A123": "car or other", "A124": "no property",
    "A141": "bank", "A142": "stores", "A143": "none",
    "A151": "rent", "A152": "own", "A153": "for free",
    "A171": "unemployed/unskilled", "A172": "unskilled",
    "A173": "skilled employee", "A174": "management",
    "A191": "no phone", "A192": "phone registered",
    "A201": "foreign worker: yes", "A202": "foreign worker: no",
}

def decode(val):
    return UCI_LABELS.get(str(val), str(val))


def _align_columns(df, reference_columns):
    """Reference column listesiyle uyumlu hale getir, eksikleri 0 yap."""
    missing = {col: 0.0 for col in reference_columns if col not in df.columns}
    if missing:
        df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1)
    return df[reference_columns].astype(float)


def _get_dataset_config(dataset):
    if dataset == "german_credit":
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/statlog/german/german.data")
        col_names = [
            "checking_account", "duration", "credit_history", "purpose",
            "credit_amount", "savings_account", "employment",
            "installment_rate", "personal_status", "other_debtors",
            "residence_since", "property", "age", "other_installments",
            "housing", "existing_credits", "job", "dependents",
            "telephone", "foreign_worker", "target"
        ]
        df = pd.read_csv(url, sep=" ", header=None, names=col_names)
        df["target"] = df["target"].map({1: 0, 2: 1})
        cat_cols = [
            "checking_account", "credit_history", "purpose",
            "savings_account", "employment", "personal_status",
            "other_debtors", "property", "other_installments",
            "housing", "job", "telephone", "foreign_worker"
        ]
        num_cols = [
            "duration", "credit_amount", "installment_rate",
            "residence_since", "age", "existing_credits", "dependents"
        ]
        immutable = ["age", "personal_status", "foreign_worker",
                     "property", "housing", "job", "telephone"]

    elif dataset == "heloc":
        df = pd.read_csv("data/raw/heloc_dataset_v1.csv")
        df.columns = [c.strip() for c in df.columns]
        df["target"] = (df["RiskPerformance"].str.strip() == "Bad").astype(int)
        df = df.drop(columns=["RiskPerformance"])
        df = df.replace([-7, -8, -9], np.nan).dropna()
        cat_cols = []
        num_cols = [c for c in df.columns if c != "target"]
        immutable = ["ExternalRiskEstimate"]

    elif dataset == "adult":
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
               "/adult/adult.data")
        col_names = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"
        ]
        df = pd.read_csv(url, header=None, names=col_names,
                         na_values=" ?", skipinitialspace=True).dropna()
        df["target"] = (df["income"].str.strip().str.startswith(">50K")).astype(int)
        df = df.drop(columns=["income"])
        cat_cols = ["workclass", "education", "marital_status", "occupation",
                    "relationship", "race", "sex", "native_country"]
        num_cols = ["age", "fnlwgt", "education_num", "capital_gain",
                    "capital_loss", "hours_per_week"]
        immutable = ["age", "race", "sex", "native_country"]

    elif dataset == "gmsc":
        df = pd.read_csv("data/raw/cs-training.csv", index_col=0)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"SeriousDlqin2yrs": "target"})
        for col in [c for c in df.columns if "NumberOfTime" in c]:
            df[col] = df[col].replace([96, 98], np.nan)
        df = df.dropna()
        cat_cols = []
        num_cols = [c for c in df.columns if c != "target"]
        immutable = ["age"]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return df, cat_cols, num_cols, immutable


def run_dice_explanation(dataset="german_credit", n_counterfactuals=3,
                          n_eval=50):
    print("=" * 60)
    print(f"DiCE COUNTERFACTUALS -- FNN on {dataset.upper()}")
    print("=" * 60)

    SEED = 42
    print("\n[1/3] Loading model...")
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn(dataset, seed=SEED)
    ref_cols = X_train.columns.tolist()
    print(f"  Features: {len(ref_cols)}")

    print("\n[2/3] Preparing data for DiCE...")
    df, cat_cols, num_cols, immutable = _get_dataset_config(dataset)

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    raw_feature_cols = [c for c in df.columns if c != "target"]

    class FNNWrapper:
        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[raw_feature_cols]
            else:
                X = pd.DataFrame(X, columns=raw_feature_cols)
            if cat_cols:
                X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)
            else:
                X_enc = X.copy()
            X_enc = _align_columns(X_enc, ref_cols)
            X_t   = torch.tensor(X_enc.values, dtype=torch.float32)
            with torch.no_grad():
                proba_bad = torch.sigmoid(model(X_t)).numpy().flatten()
            return np.column_stack([1 - proba_bad, proba_bad])

    wrapper = FNNWrapper()

    continuous_features = num_cols if num_cols else raw_feature_cols
    d   = dice_ml.Data(dataframe=df, continuous_features=continuous_features,
                       outcome_name="target")
    m   = dice_ml.Model(model=wrapper, backend="sklearn")
    exp = Dice(d, m, method="random")
    print("  DiCE explainer ready.")

    _, df_test = train_test_split(df, test_size=0.2, random_state=SEED,
                                   stratify=df["target"])
    df_test  = df_test.reset_index(drop=True)
    bad_test = df_test[df_test["target"] == 1].reset_index(drop=True)
    print(f"  Found {len(bad_test)} rejected applicants in test set.")

    vary_features = [f for f in raw_feature_cols if f not in immutable]

    print(f"\n[3/3] Computing Validity & Proximity on {n_eval} applicants...")
    validity_scores  = []
    proximity_scores = []
    N_eval = min(n_eval, len(bad_test))

    for i in range(N_eval):
        query = bad_test.iloc[[i]][raw_feature_cols]
        try:
            cf    = exp.generate_counterfactuals(
                query, total_CFs=n_counterfactuals,
                desired_class="opposite", features_to_vary=vary_features
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                validity_scores.append(1)
                if cat_cols:
                    q_enc  = pd.get_dummies(query[raw_feature_cols],
                                            columns=cat_cols, drop_first=False)
                    cf_enc = pd.get_dummies(cf_df[raw_feature_cols],
                                            columns=cat_cols, drop_first=False)
                else:
                    q_enc  = query[raw_feature_cols].copy()
                    cf_enc = cf_df[raw_feature_cols].copy()
                q_enc  = _align_columns(q_enc,  ref_cols)
                cf_enc = _align_columns(cf_enc, ref_cols)
                dist = np.abs(cf_enc.values - q_enc.values).mean(axis=1).mean()
                proximity_scores.append(dist)
            else:
                validity_scores.append(0)
        except Exception:
            validity_scores.append(0)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{N_eval} applicants...")

    val_rate  = np.mean(validity_scores)
    prox_mean = np.mean(proximity_scores) if proximity_scores else float("nan")

    print(f"\n  Validity  : {val_rate:.2%} ({sum(validity_scores)}/{N_eval})")
    print(f"  Proximity : {prox_mean:.4f}")

    print(f"\n--- Example Counterfactuals (first 3) ---")
    for i in range(min(3, len(bad_test))):
        instance = bad_test.iloc[[i]][raw_feature_cols]
        prob     = wrapper.predict_proba(instance)[0][1]
        print(f"\n  Applicant #{i+1} (default prob: {prob:.2f})")

        if dataset == "german_credit":
            inv = scaler.inverse_transform(instance[num_cols].values)[0]
            print(f"    Age: {int(inv[num_cols.index('age')])} yrs | "
                  f"Duration: {int(inv[num_cols.index('duration')])} mo | "
                  f"Amount: {int(inv[num_cols.index('credit_amount')])} DM")
            print(f"    Checking: {decode(instance['checking_account'].values[0])}")
        else:
            for col in num_cols[:3]:
                idx_n  = num_cols.index(col)
                d_orig = np.zeros(len(num_cols))
                d_orig[idx_n] = float(instance[col].values[0])
                real_val = scaler.inverse_transform([d_orig])[0][idx_n]
                print(f"    {col}: {real_val:.2f}")

        try:
            cf    = exp.generate_counterfactuals(
                instance, total_CFs=n_counterfactuals,
                desired_class="opposite", features_to_vary=vary_features
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                print(f"  Suggestions to get APPROVED:")
                for j, (_, row) in enumerate(cf_df.iterrows()):
                    changes = []
                    for feat in raw_feature_cols:
                        orig_val = instance[feat].values[0]
                        new_val  = row[feat]
                        if str(orig_val) != str(new_val):
                            if feat in num_cols:
                                idx_n      = num_cols.index(feat)
                                d_o        = np.zeros(len(num_cols))
                                d_n        = np.zeros(len(num_cols))
                                d_o[idx_n] = float(orig_val)
                                d_n[idx_n] = float(new_val)
                                r_o = scaler.inverse_transform([d_o])[0][idx_n]
                                r_n = scaler.inverse_transform([d_n])[0][idx_n]
                                if abs(r_o - r_n) >= 0.01:
                                    changes.append(
                                        f"{feat}: {r_o:.2f} -> {r_n:.2f}")
                            else:
                                if str(orig_val).strip() != str(new_val).strip():
                                    if dataset == "german_credit":
                                        changes.append(
                                            f"{feat}: {decode(orig_val)} -> {decode(new_val)}")
                                    else:
                                        changes.append(
                                            f"{feat}: {orig_val} -> {new_val}")
                    if changes:
                        print(f"    Scenario {j+1}: {' | '.join(changes)}")
        except Exception as e:
            print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print("DiCE SUMMARY")
    print("=" * 60)
    print(f"  Dataset:    {dataset}")
    print(f"  Evaluated:  {N_eval} applicants")
    print(f"  Validity:   {val_rate:.2%}")
    print(f"  Proximity:  {prox_mean:.4f}")
    print(f"  Immutable:  {immutable}")
    print(f"  Method:     Random sampling")
    print("=" * 60)

    os.makedirs("reports", exist_ok=True)
    row = {
        "dataset":   dataset,
        "validity":  round(val_rate, 4),
        "proximity": round(prox_mean, 4) if not np.isnan(prox_mean) else None,
        "n_eval":    N_eval,
    }
    csv_path = "reports/dice_results.csv"
    if os.path.exists(csv_path):
        df_csv = pd.read_csv(csv_path)
        df_csv = df_csv[df_csv["dataset"] != dataset]
        df_csv = pd.concat([df_csv, pd.DataFrame([row])], ignore_index=True)
    else:
        df_csv = pd.DataFrame([row])
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")


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
            run_dice_explanation(ds, n_counterfactuals=3, n_eval=50)
        except Exception as e:
            print(f"\nERROR on {ds}: {e}")
            traceback.print_exc()