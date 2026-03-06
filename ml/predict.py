# ml/predict.py
"""
Interfaccia di prediction: carica pipeline salvata e fornisce predict/proba per singoli input o batch DataFrame.
Scrive risultati in results/predictions.csv
"""

from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v1.0"

_pipeline = None

def load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODELS / "churn_pipeline_v1.joblib")
    return _pipeline

def predict_record(record: dict) -> dict:
    """
    record: dict with raw feature names (as in Telco dataset)
    returns dict with churn_probability, prediction, model_version, timestamp
    """
    pipeline = load_pipeline()
    df = pd.DataFrame([record])

    # Ensure customerID present for logging
    customer_id = record.get("customerID") or record.get("customerId") or record.get("customer_id")

    proba = pipeline.predict_proba(df)[:, 1][0]
    pred = int(pipeline.predict(df)[0])

    timestamp = datetime.utcnow().isoformat()
    out = {
        "customer_id": customer_id,
        "churn_probability": float(proba),
        "prediction": pred,
        "model_version": MODEL_VERSION,
        "timestamp": timestamp
    }

    # append to results CSV (save features optionally)
    # flatten features - take a subset of record keys for traceability
    flat = {
        **{"customer_id": customer_id},
        **{k: v for k, v in record.items() if k != "customerID"},
        **{"churn_probability": float(proba), "prediction": pred, "model_version": MODEL_VERSION, "timestamp": timestamp}
    }

    # load existing
    out_path = RESULTS / "predictions.csv"
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        df_existing = pd.concat([df_existing, pd.DataFrame([flat])], ignore_index=True)
        df_existing.to_csv(out_path, index=False)
    else:
        pd.DataFrame([flat]).to_csv(out_path, index=False)

    return out

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    pipeline = load_pipeline()
    proba = pipeline.predict_proba(df)[:, 1]
    pred = pipeline.predict(df)
    ts = datetime.utcnow().isoformat()
    df_out = df.copy()
    df_out["churn_probability"] = proba
    df_out["prediction"] = pred
    df_out["model_version"] = MODEL_VERSION
    df_out["timestamp"] = ts
    # append or save
    out_path = RESULTS / "predictions.csv"
    if out_path.exists():
        df_existing = pd.read_csv(out_path)
        df_existing = pd.concat([df_existing, df_out], ignore_index=True)
        df_existing.to_csv(out_path, index=False)
    else:
        df_out.to_csv(out_path, index=False)
    return df_out