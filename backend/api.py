from fastapi import FastAPI
from pydantic import BaseModel
import ml.predict as predictor

app = FastAPI()

class Customer(BaseModel):
    class Config:
        extra = "allow"

@app.get("/")
def root():
    return {"status":"API running"}

@app.post("/predict")
def predict(data: Customer):
    result = predictor.predict_record(data.dict())
    return result