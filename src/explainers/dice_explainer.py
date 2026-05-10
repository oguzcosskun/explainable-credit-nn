import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import dice_ml
from dice_ml import Dice
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath("."))
from src.preprocessing.pipeline import prepare
from src.models.fnn import FNN


def run_dice_explanation(n_counterfactuals=3):
    print("=" * 60)
    print("DiCE COUNTERFACTUALS -- FNN on German Credit")
    print("=" * 60)

    print("\n[1/3] Training FNN model...")
    SEED = 21
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_train, X_test, y_train, y_test, class_weights = prepare(
        "german_credit", "fnn", random_state=SEED
    )

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
    print("  Model trained (seed=21).")

    print("\n[2/3] Preparing raw data for DiCE...")

    df = pd.read_csv("data/raw/german_credit_data.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")

    df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

    numeric_cols = ["Age", "Job", "Credit amount", "Duration"]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    feature_cols     = [c for c in df.columns if c != "Risk"]
    categorical_cols = ["Sex", "Housing", "Saving accounts",
                        "Checking account", "Purpose"]

    class FNNWrapper:
        def __init__(self, pytorch_model, feature_cols, cat_cols):
            self.model        = pytorch_model
            self.feature_cols = feature_cols
            self.cat_cols     = cat_cols

        def predict_proba(self, X):
            if isinstance(X, pd.DataFrame):
                X = X[self.feature_cols]
            else:
                X = pd.DataFrame(X, columns=self.feature_cols)

            X_enc = pd.get_dummies(X, columns=self.cat_cols, drop_first=True)

            for col in X_train.columns:
                if col not in X_enc.columns:
                    X_enc[col] = 0.0

            X_enc = X_enc[X_train.columns].astype(float)
            X_t   = torch.tensor(X_enc.values, dtype=torch.float32)

            with torch.no_grad():
                proba_bad  = self.model(X_t).numpy().flatten()
            proba_good = 1 - proba_bad
            return np.column_stack([proba_good, proba_bad])

    wrapper = FNNWrapper(model, feature_cols, categorical_cols)

    d = dice_ml.Data(
        dataframe=df,
        continuous_features=numeric_cols,
        outcome_name="Risk"
    )
    m   = dice_ml.Model(model=wrapper, backend="sklearn")
    exp = Dice(d, m, method="random")
    print("  DiCE explainer ready.")

    print("\n[3/3] Generating counterfactuals...")

    _, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["Risk"]
    )
    df_test  = df_test.reset_index(drop=True)
    bad_test = df_test[df_test["Risk"] == 1].reset_index(drop=True)
    print(f"  Found {len(bad_test)} rejected applicants in test set.")

    immutable     = ["Age", "Sex"]
    vary_features = [f for f in feature_cols if f not in immutable]
    n_show        = min(3, len(bad_test))

    for i in range(n_show):
        instance = bad_test.iloc[[i]][feature_cols]
        prob     = wrapper.predict_proba(instance)[0][1]

        inv         = scaler.inverse_transform(instance[numeric_cols].values)[0]
        age_real    = int(inv[numeric_cols.index("Age")])
        dur_real    = int(inv[numeric_cols.index("Duration")])
        credit_real = int(inv[numeric_cols.index("Credit amount")])

        print(f"\n  --- Applicant #{i+1} (default prob: {prob:.2f}) ---")
        print(f"  Original profile:")
        print(f"    Age:              {age_real} years")
        print(f"    Duration:         {dur_real} months")
        print(f"    Credit amount:    {credit_real} DM")
        print(f"    Checking account: {instance['Checking account'].values[0]}")
        print(f"    Saving accounts:  {instance['Saving accounts'].values[0]}")

        try:
            cf    = exp.generate_counterfactuals(
                instance,
                total_CFs=n_counterfactuals,
                desired_class="opposite",
                features_to_vary=vary_features
            )
            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and len(cf_df) > 0:
                print(f"  Counterfactual suggestions (changes to get APPROVED):")
                for j, (_, row) in enumerate(cf_df.iterrows()):
                    changes = []
                    for feat in feature_cols:
                        orig_val = instance[feat].values[0]
                        new_val  = row[feat]
                        if str(orig_val) != str(new_val):
                            if feat in numeric_cols:
                                idx_n         = numeric_cols.index(feat)
                                d_orig        = np.zeros(len(numeric_cols))
                                d_new         = np.zeros(len(numeric_cols))
                                d_orig[idx_n] = float(orig_val)
                                d_new[idx_n]  = float(new_val)
                                real_orig = scaler.inverse_transform([d_orig])[0][idx_n]
                                real_new  = scaler.inverse_transform([d_new])[0][idx_n]
                                # Cok kucuk degisiklikleri atla (< 1 birim)
                                if abs(real_orig - real_new) >= 1:
                                    changes.append(
                                        f"{feat}: {real_orig:.0f} -> {real_new:.0f}"
                                    )
                            else:
                                # Kategorik: sadece gercekten degismisse ekle
                                if str(orig_val).strip() != str(new_val).strip():
                                    changes.append(
                                        f"{feat}: {orig_val} -> {new_val}"
                                    )

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
        print("\nFull traceback:")
        traceback.print_exc()