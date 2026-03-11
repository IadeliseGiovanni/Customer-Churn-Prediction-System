from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import ml.predict as predictor

app = FastAPI(title="Telco Churn API")

# --- CONFIGURAZIONE CORS ---
# Necessario per far comunicare il file HTML locale con l'API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette tutte le origini (ok per sviluppo locale)
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELLO DATI ---
# Definiamo esattamente cosa ci aspettiamo dal form HTML
class CustomerData(BaseModel):
    Gender: Literal["Male", "Female"]
    SeniorCitizen: int
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., description="Mesi di permanenza")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    # Convertiamo il modello Pydantic in dizionario per la funzione di predizione
    record = data.model_dump()
    result = predictor.predict_record(record)
    return result