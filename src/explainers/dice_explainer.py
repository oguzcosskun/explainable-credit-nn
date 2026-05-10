import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.abspath("."))
from src.models.train_utils import get_trained_fnn

import torch
import dice_ml
from dice_ml import Dice

# UCI kod -> anlamli isim sozlugu
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


def run_dice_explanation(n_counterfactuals=3):
    print("=" * 60)
    print("DiCE COUNTERFACTUALS -- FNN on German Credit (UCI)")
    print("=" * 60)

    # === 1. MODEL ===
    print("\n[1/3] Training FNN model...")
    SEED = 42
    model, X_train, X_test, y_train, y_test, X_train_t, X_test_t = \
        get_trained_fnn("german_credit", seed=SEED)
    print(f"  Features: {len(X_train.columns)}")

    # === 2. DiCE ICIN HAM VERIYI HAZIRLA ===
    print("\n[2/3] Preparing raw data for DiCE...")

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

    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    raw_feature_cols = [c for c in df.columns if c != "target"]

    class FNNWrapper:
        def __init__(self, pytorch_model):
            self.model = pytorch_model

        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[raw_feature_cols]
            else:
                X = pd.DataFrame(X, columns=raw_feature_cols)
            X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            for col in X_train.columns:
                if col not in X_enc.columns:
                    X_enc[col] = 0.0
            X_enc = X_enc[X_train.columns].astype(float)
            X_t   = torch.tensor(X_enc.values, dtype=torch.float32)
            with torch.no_grad():
                proba_bad = torch.sigmoid(model(X_t)).numpy().flatten()
            return np.column_stack([1 - proba_bad, proba_bad])

    wrapper = FNNWrapper(model)

    d   = dice_ml.Data(dataframe=df, continuous_features=num_cols,
                       outcome_name="target")
    m   = dice_ml.Model(model=wrapper, backend="sklearn")
    exp = Dice(d, m, method="random")
    print("  DiCE explainer ready.")

    # === 3. COUNTERFACTUAL URET ===
    print("\n[3/3] Generating counterfactuals...")

    _, df_test = train_test_split(df, test_size=0.2, random_state=SEED,
                                   stratify=df["target"])
    df_test  = df_test.reset_index(drop=True)
    bad_test = df_test[df_test["target"] == 1].reset_index(drop=True)
    print(f"  Found {len(bad_test)} rejected applicants in test set.")

    immutable     = ["age", "personal_status", "foreign_worker"]
    vary_features = [f for f in raw_feature_cols if f not in immutable]
    n_show        = min(3, len(bad_test))

    for i in range(n_show):
        instance = bad_test.iloc[[i]][raw_feature_cols]
        prob     = wrapper.predict_proba(instance)[0][1]

        inv         = scaler.inverse_transform(instance[num_cols].values)[0]
        age_real    = int(inv[num_cols.index("age")])
        dur_real    = int(inv[num_cols.index("duration")])
        credit_real = int(inv[num_cols.index("credit_amount")])

        print(f"\n  --- Applicant #{i+1} (default prob: {prob:.2f}) ---")
        print(f"  Original profile:")
        print(f"    Age:              {age_real} years")
        print(f"    Duration:         {dur_real} months")
        print(f"    Credit amount:    {credit_real} DM")
        print(f"    Checking account: {decode(instance['checking_account'].values[0])}")
        print(f"    Savings account:  {decode(instance['savings_account'].values[0])}")

        try:
            cf    = exp.generate_counterfactuals(
                instance, total_CFs=n_counterfactuals,
                desired_class="opposite", features_to_vary=vary_features
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and len(cf_df) > 0:
                print(f"  Counterfactual suggestions (to get APPROVED):")
                for j, (_, row) in enumerate(cf_df.iterrows()):
                    changes = []
                    for feat in raw_feature_cols:
                        orig_val = instance[feat].values[0]
                        new_val  = row[feat]
                        if str(orig_val) != str(new_val):
                            if feat in num_cols:
                                idx_n         = num_cols.index(feat)
                                d_orig        = np.zeros(len(num_cols))
                                d_new         = np.zeros(len(num_cols))
                                d_orig[idx_n] = float(orig_val)
                                d_new[idx_n]  = float(new_val)
                                real_orig = scaler.inverse_transform([d_orig])[0][idx_n]
                                real_new  = scaler.inverse_transform([d_new])[0][idx_n]
                                if abs(real_orig - real_new) >= 1:
                                    changes.append(
                                        f"{feat}: {real_orig:.0f} -> {real_new:.0f}")
                            else:
                                if str(orig_val).strip() != str(new_val).strip():
                                    changes.append(
                                        f"{feat}: {decode(orig_val)} -> {decode(new_val)}")
                    if changes:
                        print(f"\n    Scenario {j+1}:")
                        for c in changes:
                            print(f"      {c}")
            else:
                print("    No valid counterfactuals found.")
        except Exception as e:
            print(f"    Error: {e}")

    print("\n" + "=" * 60)
    print("DiCE SUMMARY")
    print("=" * 60)
    print(f"  Applicants analyzed: {n_show}")
    print(f"  Counterfactuals per applicant: {n_counterfactuals}")
    print(f"  Immutable features: {immutable}")
    print(f"  Method: Random sampling")
    print("=" * 60)


if __name__ == "__main__":
    import traceback
    try:
        run_dice_explanation(n_counterfactuals=3)
    except BaseException as e:
        print("\n!!! ERROR CAUGHT !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        traceback.print_exc()