from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Standard colonne (senza spazi) usato da training/predizione/API.
TARGET_COL = "ChurnValue"

# Mapping utile per trasformare label di churn in 0/1.
CHURN_MAP = {"Yes": 1, "No": 0, "Churned": 1, "Stayed": 0, 1: 1, 0: 0}

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def clean_raw(
    df: pd.DataFrame,
    include_log_totalcharges: bool = False,
    keep_customer_id: bool = False,
) -> pd.DataFrame:
    """Pulisce il dataset e genera feature usando uno schema senza spazi."""
    df = df.copy()

    # Normalizza nomi colonna rimuovendo spazi: es. "Total Charges" -> "TotalCharges".
    df.columns = df.columns.str.strip().str.replace(" ", "", regex=False)

    cols_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "LatLong",
        "ZipCode",
        "ChurnReason",
        "ChurnScore",
        "ChurnLabel",
        "CLTV",
        "Latitude",
        "Longitude",
    ]
    if keep_customer_id and "CustomerID" in cols_to_drop:
        cols_to_drop.remove("CustomerID")

    # Target -> 0/1
    target = next((c for c in df.columns if "churn" in c.lower()), TARGET_COL)
    if target in df.columns and target != TARGET_COL:
        df.rename(columns={target: TARGET_COL}, inplace=True)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].replace(CHURN_MAP)

    # SeniorCitizen: il raw spesso e una stringa Yes/No, ma la UI usa 0/1.
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({"Yes": 1, "No": 0, "0": 0, "1": 1})
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")

    # Numeriche
    for col in ["TotalCharges", "TenureMonths", "MonthlyCharges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    if "TotalCharges" in df.columns and "TenureMonths" in df.columns:
        df["AvgMonthlySpend"] = df["TotalCharges"] / df["TenureMonths"].replace(0, 1)

    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PhoneService",
    ]

    available_services = [c for c in service_cols if c in df.columns]
    service_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}

    if available_services:
        service_numeric = (
            df[available_services]
            .replace(service_map)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )
        df["NumServices"] = service_numeric.sum(axis=1)

        streaming_cols = [c for c in ["StreamingTV", "StreamingMovies"] if c in service_numeric.columns]
        if streaming_cols:
            df["StreamingBundleCount"] = service_numeric[streaming_cols].sum(axis=1)

        phone_cols = [c for c in ["PhoneService", "MultipleLines"] if c in service_numeric.columns]
        if phone_cols:
            df["PhoneBundleCount"] = service_numeric[phone_cols].sum(axis=1)

    if "MonthlyCharges" in df.columns and "NumServices" in df.columns:
        df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    if "InternetService" in df.columns:
        df["HasInternet"] = (df["InternetService"].astype(str).str.strip().str.lower() != "no").astype(int)

    if "PaymentMethod" in df.columns:
        df["Is_Electronic_Check"] = (
            df["PaymentMethod"].astype(str).str.strip().str.lower() == "electronic check"
        ).astype(int)

    if include_log_totalcharges and "TotalCharges" in df.columns:
        df["Log_TotalCharges"] = np.log1p(pd.to_numeric(df["TotalCharges"], errors="coerce"))

    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df.drop_duplicates()


def split_save(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found after preprocessing")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pd.concat([X_train, y_train], axis=1).to_csv(PROC_DIR / "train_raw.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROC_DIR / "test_raw.csv", index=False)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df_processed = clean_raw(df)
    split_save(df_processed)


if __name__ == "__main__":
    main()
