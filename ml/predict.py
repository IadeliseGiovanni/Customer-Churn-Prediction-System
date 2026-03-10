"""Prediction helper.

Questo modulo viene usato dal backend FastAPI (`backend/api.py`).
Obiettivo: accettare un record (dict) anche parziale e trasformarlo in un DataFrame
compatibile con la Pipeline addestrata.

Punti chiave:
- Le colonne attese vengono ricavate dal `preprocessor` dentro la pipeline.
- Se mancano colonne, vengono create come NA (saranno gestite dagli imputers).
- Alcuni nomi vengono normalizzati (rename_map) per compatibilità con input UI/API.
- `threshold` è opzionale: utile se vuoi una soglia diversa da 0.5 (es. per massimizzare F1).
"""

from pathlib import Path

import os

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"

# Pipeline completa: preprocessing + modello
pipeline = joblib.load(MODEL)

# Log opzionali (utile in debug locale): `set CHURN_DEBUG_PREDICT=1`
DEBUG = os.getenv("CHURN_DEBUG_PREDICT", "") == "1"
if DEBUG:
    print(f"[predict] Loaded model: {MODEL}")


def predict_record(record: dict, threshold: float | None = None) -> dict:
    # 1) Dict -> DataFrame (una riga)
    df = pd.DataFrame([record]).copy()

    if DEBUG:
        print(f"[predict] input_cols={list(df.columns)}")

    # 2) Normalizzazione nomi colonna (UI/API vs training dataset)
    rename_map = {
        "tenure": "TenureMonths",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})

    # 3) Allineamento alle colonne attese dalla pipeline
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers"):
        expected_cols = []
        for name, _, cols in preprocessor.transformers:
            if cols is None or cols in ("drop", "passthrough"):
                continue
            expected_cols.extend(list(cols))

        # Crea colonne mancanti: gli imputers del preprocessing le gestiranno.
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

        # Feature derivata usata nel training: calcolo solo se posso.
        if "Avg Monthly Spend" in expected_cols and "Avg Monthly Spend" not in record:
            if "TotalCharges" in df.columns and "TenureMonths" in df.columns:
                total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
                tenure = pd.to_numeric(df["TenureMonths"], errors="coerce").replace(0, 1)
                df["Avg Monthly Spend"] = total_charges / tenure

        # Conversione numeriche: evita che numeri arrivino come stringhe (es. "42").
        num_cols = []
        for name, _, cols in preprocessor.transformers:
            if name == "num":
                num_cols = list(cols)
                break
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ordine colonne coerente con training
        df = df[expected_cols]

    # 4) Probabilità churn (classe positiva = 1)
    proba = float(pipeline.predict_proba(df)[0][1])

    # 5) Decisione di classe
    # - threshold None: usa la logica del modello/pipeline (tipicamente soglia 0.5).
    # - threshold esplicita: utile per ottimizzare F1 o recall/precision tradeoff.
    if threshold is None:
        pred = int(pipeline.predict(df)[0])
    else:
        if not (0.0 < float(threshold) < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= float(threshold))

    if DEBUG:
        print(f"[predict] output prob={proba:.4f} pred={pred}")

    return {
        "churn_probability": proba,
        "prediction": pred,
    }
