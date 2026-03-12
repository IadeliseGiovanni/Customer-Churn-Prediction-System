from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"

# Carico una sola volta la pipeline (utile per performance nel backend)
pipeline = joblib.load(MODEL)
print(f"[predict] Loaded model: {MODEL}")


def _expected_cols_from_pipeline() -> list[str] | None:
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "transformers"):
        return None

    expected_cols: list[str] = []
    for name, _, cols in preprocessor.transformers:
        if cols is None or cols in ("drop", "passthrough"):
            continue
        expected_cols.extend(list(cols))

    return expected_cols


def _align_df_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalizzazione nomi (alcuni input UI/API usano nomi diversi dal training)
    rename_map = {
        "tenure": "TenureMonths",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges",
        "Internet Service": "InternetService",
        "Payment Method": "PaymentMethod",
        "Paperless Billing": "PaperlessBilling",
        "Senior Citizen": "SeniorCitizen",
        "Phone Service": "PhoneService",
        "Multiple Lines": "MultipleLines",
        "Online Security": "OnlineSecurity",
        "Online Backup": "OnlineBackup",
        "Device Protection": "DeviceProtection",
        "Tech Support": "TechSupport",
        "Streaming TV": "StreamingTV",
        "Streaming Movies": "StreamingMovies",
        "Avg Monthly Spend": "AvgMonthlySpend",
        "Charges per Service": "ChargesPerService",
    }

    rename_cols = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if rename_cols:
        df = df.rename(columns=rename_cols)

    expected_cols = _expected_cols_from_pipeline()
    if not expected_cols:
        return df

    # Bridge: se il modello e stato addestrato con colonne con spazi e l'input e senza spazi (o viceversa),
    # crea alias usando la chiave senza spazi.
    df_cols_by_nospace = {str(c).replace(" ", ""): c for c in df.columns}
    for exp in expected_cols:
        if exp in df.columns:
            continue
        key = str(exp).replace(" ", "")
        src = df_cols_by_nospace.get(key)
        if src is not None:
            df[exp] = df[src]

    # Colonne mancanti -> NaN (imputers nel preprocessing le gestiranno)
    missing_cols = [c for c in expected_cols if c not in df.columns]
    for col in missing_cols:
        df[col] = np.nan

    # Feature derivata: se il modello si aspetta AvgMonthlySpend e possiamo calcolarla.
    if "AvgMonthlySpend" in expected_cols and "AvgMonthlySpend" not in df.columns:
        if "TotalCharges" in df.columns and "TenureMonths" in df.columns:
            total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
            tenure = pd.to_numeric(df["TenureMonths"], errors="coerce").replace(0, 1)
            df["AvgMonthlySpend"] = total_charges / tenure

    # Conversione numeriche (basata sui transformer del preprocessor)
    preprocessor = pipeline.named_steps.get("preprocessor")
    num_cols: list[str] = []
    if preprocessor is not None and hasattr(preprocessor, "transformers"):
        for name, _, cols in preprocessor.transformers:
            if name == "num":
                num_cols = list(cols)
                break

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[expected_cols]


def predict_record(record: dict, threshold: float | None = None) -> dict:
    df = pd.DataFrame([record])
    df_aligned = _align_df_to_expected(df)

    proba = float(pipeline.predict_proba(df_aligned)[0][1])

    if threshold is None:
        pred = int(pipeline.predict(df_aligned)[0])
    else:
        thr = float(threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= thr)

    return {"churn_probability": proba, "prediction": pred}


def predict_dataframe(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Predice su un DataFrame e ritorna un DataFrame con proba + pred."""
    df_aligned = _align_df_to_expected(df)

    proba = pipeline.predict_proba(df_aligned)[:, 1]
    pred = (proba >= float(threshold)).astype(int)

    out = df.copy()
    out["churn_probability"] = proba
    out["prediction"] = pred
    return out
