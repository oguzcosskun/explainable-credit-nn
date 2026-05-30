import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def prepare(dataset_name="german_credit", model_family="fnn",
            test_size=0.2, random_state=42):
    """
    Unified preprocessing pipeline.
    dataset_name : "german_credit", "heloc", "adult", "gmsc"
    model_family : "fnn" (one-hot) | "tabnet" (label encoding)
    Returns: X_train, X_test, y_train, y_test, class_weights
    """

    loaders = {
        "german_credit": _load_german,
        "heloc":         _load_heloc,
        "adult":         _load_adult,
        "gmsc":          _load_gmsc,
    }

    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {list(loaders.keys())}")

    X, y = loaders[dataset_name]()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # === SPLIT ÖNCE (leakage önlemek için) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # === IMPUTATION (sadece train istatistikleri) ===
    num_medians = {col: X_train[col].median()
                   for col in num_cols if X_train[col].isnull().any()}
    cat_modes   = {col: X_train[col].mode()[0]
                   for col in cat_cols if X_train[col].isnull().any()}

    for col, val in num_medians.items():
        X_train[col] = X_train[col].fillna(val)
        X_test[col]  = X_test[col].fillna(val)
    for col, val in cat_modes.items():
        X_train[col] = X_train[col].fillna(val)
        X_test[col]  = X_test[col].fillna(val)

    # === ENCODING ===
    if model_family == "fnn":
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
        X_test  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=False)
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0
        X_test = X_test[X_train.columns]
    elif model_family == "tabnet":
        le = LabelEncoder()
        for col in cat_cols:
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col]  = le.transform(X_test[col].astype(str))
    else:
        raise ValueError(f"model_family '{model_family}' not supported.")

    # === SCALING (sadece train'e fit, test'e transform) ===
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns]   = scaler.transform(X_test)

    # === CLASS WEIGHTS ===
    n      = len(y_train)
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
    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"[German Credit UCI] shape={df.shape}, "
          f"default_rate={y.mean()*100:.2f}%")
    cat_cols = [
        "checking_account", "credit_history", "purpose",
        "savings_account", "employment", "personal_status",
        "other_debtors", "property", "other_installments",
        "housing", "job", "telephone", "foreign_worker"
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"  Categorical cols ({len(cat_cols)}): {cat_cols}")
    print(f"  Numerical   cols ({len(num_cols)}): {num_cols}")
    return X, y


def _load_heloc():
    """
    HELOC (Home Equity Line of Credit) dataset.
    Target: 1 = Bad (RiskPerformance == Bad), 0 = Good
    Sentinel values -7, -8, -9 treated as missing.
    """
    path = "data/raw/heloc_dataset_v1.csv"
    print("[HELOC] Loading from local file...")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["target"] = (df["RiskPerformance"].str.strip() == "Bad").astype(int)
    X = df.drop(columns=["RiskPerformance", "target"])
    X = X.replace([-7, -8, -9], np.nan)
    y = df["target"]
    print(f"[HELOC] shape={df.shape}, default_rate={y.mean()*100:.2f}%")
    return X, y


def _load_adult():
    """
    UCI Adult Income dataset.
    Target: 1 = >50K, 0 = <=50K
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases"
           "/adult/adult.data")
    col_names = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]
    print("[Adult Income] Loading from UCI...")
    df = pd.read_csv(url, header=None, names=col_names,
                     na_values=" ?", skipinitialspace=True)
    df["target"] = (df["income"].str.strip().str.startswith(">50K")).astype(int)
    X = df.drop(columns=["income", "target"])
    y = df["target"]
    print(f"[Adult Income] shape={df.shape}, default_rate={y.mean()*100:.2f}%")
    return X, y


def _load_gmsc():
    """
    Give Me Some Credit dataset.
    Target: 1 = SeriousDlqin2yrs (default)
    Sentinel values 96, 98 in NumberOfTime columns treated as missing.
    """
    path = "data/raw/cs-training.csv"
    print("[GMSC] Loading from local file...")
    df = pd.read_csv(path, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    y = df["SeriousDlqin2yrs"].astype(int)
    X = df.drop(columns=["SeriousDlqin2yrs"])
    for col in [c for c in X.columns if "NumberOfTime" in c]:
        X[col] = X[col].replace([96, 98], np.nan)
    print(f"[GMSC] shape={df.shape}, default_rate={y.mean()*100:.2f}%")
    return X, y


# === TEST ===
if __name__ == "__main__":
    for dataset in ["german_credit", "heloc", "adult", "gmsc"]:
        print(f"\n{'='*50}")
        print(f"Testing: {dataset.upper()}")
        print(f"{'='*50}")
        try:
            X_train, X_test, y_train, y_test, weights = prepare(
                dataset, "fnn"
            )
            print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
            print(f"Class weights: {weights}")
            print(f"OK ✓")
        except Exception as e:
            print(f"ERROR: {e}")