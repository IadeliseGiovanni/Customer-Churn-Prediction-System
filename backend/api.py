# backend/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import ml.predict as predictor
import pandas as pd
from pathlib import Path

app = FastAPI(title="Churn Prediction API")

class CustomerIn(BaseModel):
    customerID: Optional[str] = Field(None, description="Customer ID (optional)")
    gender: Optional[str]
    SeniorCitizen: Optional[str]
    Partner: Optional[str]
    Dependents: Optional[str]
    tenure: Optional[float]
    PhoneService: Optional[str]
    MultipleLines: Optional[str]
    InternetService: Optional[str]
    OnlineSecurity: Optional[str]
    OnlineBackup: Optional[str]
    DeviceProtection: Optional[str]
    TechSupport: Optional[str]
    StreamingTV: Optional[str]
    StreamingMovies: Optional[str]
    Contract: Optional[str]
    PaperlessBilling: Optional[str]
    PaymentMethod: Optional[str]
    MonthlyCharges: Optional[float]
    TotalCharges: Optional[float]

class BatchIn(BaseModel):
    data: List[CustomerIn]

@app.get("/")
def root():
    return {"ok": True, "info": "Churn Prediction API"}

@app.post("/predict")
def predict_single(payload: CustomerIn):
    record = payload.dict()
    # map Pydantic keys to the pipeline expectation if needed
    try:
        out = predictor.predict_record(record)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(payload: BatchIn):
    records = [r.dict() for r in payload.data]
    df = pd.DataFrame(records)
    try:
        out_df = predictor.predict_batch(df)
        # return small summary
        return {"n": len(out_df), "saved_to": str(Path("results/predictions.csv").resolve())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_results():
    p = Path("results/predictions.csv")
    if not p.exists():
        return {"results": []}
    df = pd.read_csv(p)
    # return head for quick check
    return {"n": len(df), "sample": df.head(10).to_dict(orient="records")}