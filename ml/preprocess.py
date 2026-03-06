# ml/preprocess.py
"""
Preprocess per Telco Customer Churn (IBM).
Produce data/processed/train.csv e data/processed/test.csv
Salva anche preprocessor (ColumnTransformer) se utile.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def load_raw(path=RAW):
    df = pd.read_csv(path)
    return df

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    # standard rename
    df = df.copy()
    # strip whitespace in column names
    df.columns = [c.strip() for c in df.columns]

    # typical column names in Telco dataset: 'customerID', 'TotalCharges', 'MonthlyCharges', 'tenure', 'Churn', ...
    # drop entirely empty columns (if any)
    df = df.dropna(axis=1, how="all")

    # convert TotalCharges to numeric: sometimes it's blank -> coerce to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # duplicates
    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])
    else:
        df = df.drop_duplicates()

    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # if TotalCharges missing, estimate as MonthlyCharges * tenure when possible
    if {"TotalCharges", "MonthlyCharges", "tenure"}.issubset(df.columns):
        mask = df["TotalCharges"].isna() & df["MonthlyCharges"].notna() & df["tenure"].notna()
        df.loc[mask, "TotalCharges"] = (df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"])

    # create churn target numeric
    if "Churn" in df.columns:
        df["churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    else:
        # if already present as churn
        if "churn" not in df.columns:
            raise ValueError("Nessuna colonna 'Churn' trovata nel dataset raw.")

    # cast SeniorCitizen to categorical (0/1 -> str) to one-hot encode as category
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # drop columns not used as features but keep customerID for output
    return df

def build_preprocessor(df: pd.DataFrame):
    # separate feature types
    # choose a safe default set of features (common in Telco dataset)
    numeric_features = [
        c for c in ["tenure", "MonthlyCharges", "TotalCharges", "NumServices"]  # NumServices placeholder
        if c in df.columns
    ]
    # fallback: if TotalCharges exists, include it
    numeric_features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]

    # categorical features: all object/string columns except customerID and Churn
    categorical_features = [c for c in df.select_dtypes(include=["object", "category"]).columns
                            if c not in {"customerID", "Churn"}]

    # Imputers and encoders
    numeric_transformer = (
        ColumnTransformer([( "num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)], remainder="drop")
    )

    # but simpler: use ColumnTransformer with numeric and categorical
    from sklearn.pipeline import Pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features)
        ],
        remainder="drop", verbose_feature_names_out=False
    )

    return preprocessor, numeric_features, categorical_features

def main():
    df = load_raw()
    df = clean_basic(df)
    df = feature_engineer(df)

    # Save cleaned raw for traceability
    df.to_csv(PROC / "cleaned_raw.csv", index=False)

    preprocessor, num_feats, cat_feats = build_preprocessor(df)

    # Prepare X, y
    keep_cols = num_feats + cat_feats
    X = df[keep_cols].copy()
    y = df["churn"].copy()
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # For reproducibility, fit preprocessor on train and save
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS / "preprocessor_v1.joblib")

    # transform and save processed CSVs (transformed arrays -> we'll save with column names)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # build column names from transformer
    # get feature names for numeric
    num_names = num_feats
    # for categorical: retrieve encoder categories
    try:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(cat_feats).tolist()
    except Exception:
        # fallback generic names
        cat_names = [f"cat_{i}" for i in range(X_train_t.shape[1] - len(num_feats))]

    columns = num_names + cat_names

    df_train = pd.DataFrame(X_train_t, columns=columns, index=X_train.index)
    df_train["churn"] = y_train.values
    df_test = pd.DataFrame(X_test_t, columns=columns, index=X_test.index)
    df_test["churn"] = y_test.values

    df_train.to_csv(PROC / "train.csv", index=False)
    df_test.to_csv(PROC / "test.csv", index=False)

    print("Preprocessing completato. File in data/processed/ e preprocessor salvato in models/")

if __name__ == "__main__":
    main()