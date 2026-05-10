import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def prepare(dataset_name="german_credit", model_family="fnn", test_size=0.2, random_state=42):
    """
    Unified preprocessing pipeline.
    
    Parameters:
        dataset_name: "german_credit" (diğer datasetler ileride eklenecek)
        model_family: "fnn" (one-hot encoding) veya "tabnet" (label encoding)
        test_size: test split oranı
        random_state: reproducibility için seed
    
    Returns:
        X_train, X_test, y_train, y_test, class_weights
    """
    
    # === 1. VERİYİ OKU ===
    if dataset_name == "german_credit":
        df = pd.read_csv("data/raw/german_credit_data.csv")
        
        # Unnamed index sütununu at
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        # Target: "good" -> 0, "bad" -> 1 (1 = default = tahmin etmek istediğimiz şey)
        df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})
        
        target_col = "Risk"
    else:
        raise ValueError(f"Dataset '{dataset_name}' henüz desteklenmiyor.")
    
    # === 2. FEATURE VE TARGET'I AYIR ===
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # === 3. MISSING VALUES ===
    # Numerik sütunlar: median ile doldur
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    
    # Kategorik sütunlar: mode ile doldur
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown")
    
    # === 4. ENCODING ===
    if model_family == "fnn":
        # One-Hot Encoding — kategorik sütunları dummy'lere çevir
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    elif model_family == "tabnet":
        # Label Encoding — her kategorik sütunu sayıya çevir
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
    
    else:
        raise ValueError(f"model_family '{model_family}' desteklenmiyor. 'fnn' veya 'tabnet' kullanın.")
    
    # === 5. SCALING ===
    # Tüm sütunları 0-1 arasına normalize et
    scaler = MinMaxScaler()
    X[X.columns] = scaler.fit_transform(X)
    
    # === 6. TRAIN/TEST SPLIT ===
    # Stratified: orijinal class dağılımını korur
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # === 7. CLASS WEIGHTS ===
    # Weighted Cross-Entropy için: azınlık sınıfına daha fazla ağırlık ver
    n_samples = len(y_train)
    n_bad = y_train.sum()       # bad = 1
    n_good = n_samples - n_bad  # good = 0
    
    # weight = n_samples / (2 * n_class)
    weight_good = n_samples / (2 * n_good)
    weight_bad = n_samples / (2 * n_bad)
    
    class_weights = {0: weight_good, 1: weight_bad}
    
    # Index'leri sıfırla (PyTorch tensor'a çevirirken sorun çıkmasın)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test, class_weights


# === DOĞRUDAN ÇALIŞTIRMA TESTİ ===
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, weights = prepare("german_credit", "fnn")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
    print(f"Class weights: {weights}")
    print(f"\nFeature columns ({X_train.shape[1]} total):")
    print(X_train.columns.tolist())