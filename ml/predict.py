import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"

pipeline = joblib.load(MODEL)

def predict_record(record: dict):
    df = pd.DataFrame([record])
    proba = pipeline.predict_proba(df)[0][1]
    pred = pipeline.predict(df)[0]

    return {
        "churn_probability": float(proba),
        "prediction": int(pred)
    }