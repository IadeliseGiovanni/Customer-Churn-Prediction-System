from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIGURAZIONE PERCORSI ---
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Churn Value"
CHURN_MAP = {"Yes": 1, "No": 0, "Churned": 1, "Stayed": 0, 1: 1, 0: 0}


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rende compatibili varianti raw/processed con e senza spazi."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "ChurnValue": "Churn Value",
        "TenureMonths": "Tenure Months",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "SeniorCitizen": "Senior Citizen",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "InternetService": "Internet Service",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaperlessBilling": "Paperless Billing",
        "PaymentMethod": "Payment Method",
        "LatLong": "Lat Long",
        "ZipCode": "Zip Code",
        "CustomerId": "CustomerID",
    }
    applicable = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if applicable:
        df = df.rename(columns=applicable)

    return df


def clean_raw(
    df: pd.DataFrame,
    include_log_totalcharges: bool = False,
    use_gender: bool = True,
    use_total_charges: bool = True,
    keep_customer_id: bool = False,
) -> pd.DataFrame:
    """Pulisce il dataset e genera feature mantenendo compatibilità con training corrente."""
    df = _normalize_schema(df)

    cols_to_drop = [
        "CustomerID",
        "Count",
        "Country",
        "State",
        "City",
        "Lat Long",
        "Zip Code",
        "Churn Reason",
        "Churn Score",
        "Churn Label",
        "CLTV",
        "Latitude",
        "Longitude",
    ]
    if keep_customer_id and "CustomerID" in cols_to_drop:
        cols_to_drop.remove("CustomerID")

    # Target -> 0/1 con fallback robusto su possibili varianti
    target_candidates = [
        "Churn Value",
        "ChurnValue",
        "Churn",
        "churn",
    ]
    target = next((c for c in target_candidates if c in df.columns), None)
    if target and target != TARGET_COL:
        df.rename(columns={target: TARGET_COL}, inplace=True)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].replace(CHURN_MAP)

    # Gender opzionale
    if "Gender" in df.columns:
        if use_gender:
            df["Gender"] = df["Gender"].replace({"Female": 1, "Male": 0})
            df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")
        else:
            cols_to_drop.append("Gender")

    # Senior Citizen robusto (Yes/No o 0/1)
    if "Senior Citizen" in df.columns:
        df["Senior Citizen"] = df["Senior Citizen"].replace({"Yes": 1, "No": 0, "0": 0, "1": 1})
        df["Senior Citizen"] = pd.to_numeric(df["Senior Citizen"], errors="coerce")

    # Conversione numeriche coerente
    for col in ["Total Charges", "Tenure Months", "Monthly Charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature engineering
    if {"Total Charges", "Tenure Months"}.issubset(df.columns):
        tenure_nonzero = df["Tenure Months"].replace(0, np.nan)
        df["AvgMonthlySpend"] = df["Total Charges"] / tenure_nonzero

    service_cols = [
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Phone Service",
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

        streaming_cols = [c for c in ["Streaming TV", "Streaming Movies"] if c in service_numeric.columns]
        if streaming_cols:
            df["StreamingBundleCount"] = service_numeric[streaming_cols].sum(axis=1)
            # manteniamo solo aggregata streaming per coerenza con richieste precedenti
            cols_to_drop.extend(streaming_cols)

        phone_cols = [c for c in ["Phone Service", "Multiple Lines"] if c in service_numeric.columns]
        if phone_cols:
            df["PhoneBundleCount"] = service_numeric[phone_cols].sum(axis=1)

    if "Monthly Charges" in df.columns and "NumServices" in df.columns:
        df["ChargesPerService"] = df["Monthly Charges"] / (df["NumServices"] + 1)

    if "Internet Service" in df.columns:
        df["HasInternet"] = (df["Internet Service"].astype(str).str.strip().str.lower() != "no").astype(int)

    if "Payment Method" in df.columns:
        df["Is_Electronic_Check"] = (
            df["Payment Method"].astype(str).str.strip().str.lower() == "electronic check"
        ).astype(int)

    if include_log_totalcharges and "Total Charges" in df.columns:
        df["Log_TotalCharges"] = np.log1p(pd.to_numeric(df["Total Charges"], errors="coerce"))

    if not use_total_charges and "Total Charges" in df.columns:
        cols_to_drop.append("Total Charges")

    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    return df.drop_duplicates()


def split_save(df: pd.DataFrame) -> None:
    """Gestisce lo split dei dati e il salvataggio fisico."""
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

    # Config toggles
    use_gender = False
    use_total_charges = True

    df = pd.read_csv(RAW_FILE)
    df_processed = clean_raw(
        df,
        include_log_totalcharges=False,
        use_gender=use_gender,
        use_total_charges=use_total_charges,
    )
    split_save(df_processed)

    print(
        f"✅ DATASET PRONTO: {df_processed.shape[0]} righe, {df_processed.shape[1]} colonne | "
        f"use_gender={use_gender}, use_total_charges={use_total_charges}"
    )


if __name__ == "__main__":
    main()