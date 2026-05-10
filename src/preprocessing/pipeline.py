import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def prepare(dataset_name="german_credit", model_family="fnn",
            test_size=0.2, random_state=42):
    """
    Unified preprocessing pipeline.
    dataset_name : "german_credit"
    model_family : "fnn" (one-hot) | "tabnet" (label encoding)
    Returns: X_train, X_test, y_train, y_test, class_weights
    """

    if dataset_name == "german_credit":
        X, y = _load_german()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported yet.")

    # === ENCODING ===
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if model_family == "fnn":
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    elif model_family == "tabnet":
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))
    else:
        raise ValueError(f"model_family '{model_family}' not supported.")

    # === SCALING ===
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)

    # === SPLIT ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # === CLASS WEIGHTS ===
    n = len(y_train)
    n_bad  = y_train.sum()
    n_good = n - n_bad
    class_weights = {
        0: n / (2 * n_good),
        1: n / (2 * n_bad)
    }

    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test, class_weights


def _load_german():
    """
    Orijinal UCI German Credit dataset.
    URL: archive.ics.uci.edu — ham .data dosyası, sütun adı yok.
    20 feature, binary target (1=good→0, 2=bad→1).
    """
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

    # Target: 1=good → 0, 2=bad → 1
    df["target"] = df["target"].map({1: 0, 2: 1})

    # Kategorik sütunlar — UCI'da A11, A12 gibi kodlar var, anlamlı isimlere çevir
    cat_cols = [
        "checking_account", "credit_history", "purpose",
        "savings_account", "employment", "personal_status",
        "other_debtors", "property", "other_installments",
        "housing", "job", "telephone", "foreign_worker"
    ]

    X = df.drop(columns=["target"])
    y = df["target"]

    print(f"[German Credit UCI] shape={df.shape}, "
          f"default_rate={y.mean()*100:.2f}%")
    print(f"  Categorical cols ({len(cat_cols)}): {cat_cols}")
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"  Numerical   cols ({len(num_cols)}): {num_cols}")

    return X, y


# === TEST ===
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, weights = prepare(
        "german_credit", "fnn"
    )
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train dist:\n{y_train.value_counts()}")
    print(f"Class weights: {weights}")
    print(f"\nFeatures ({X_train.shape[1]}):")
    print(X_train.columns.tolist())