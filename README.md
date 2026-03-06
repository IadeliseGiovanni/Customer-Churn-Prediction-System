# Customer Churn Prediction System

## Project Overview

Questo progetto implementa un sistema completo di **Customer Churn Prediction** che utilizza Machine Learning per stimare la probabilità che un cliente abbandoni un servizio.

L'obiettivo è dimostrare competenze in:

* Machine Learning
* Data pipeline
* Backend API
* Frontend interattivo
* Data export e reporting

Il sistema prende dati cliente in input, genera una **probabilità di churn** e salva i risultati in formato CSV.

---

# Tech Stack

**Python Version**

Python 3.12

**Machine Learning**

* scikit-learn
* pandas
* numpy
* joblib

**Backend**

* FastAPI
* Uvicorn

**Frontend**

* Streamlit

**Data Visualization**

* matplotlib
* seaborn

---

# Project Structure

```
churn-prediction-project

data/
raw/
processed/

ml/
preprocess.py
train.py
predict.py

backend/
api.py

frontend/
app.py

models/
churn_model.pkl

results/
predictions.csv

reports/
model_metrics.csv
```


# ⚙️ Data & ML Pipeline

## 1 Data Preprocessing

Script:

```
ml/preprocess.py
```

Funzioni:

* pulizia dataset
* encoding feature categoriche
* normalizzazione
* split train/test

Output:

```
data/processed/train.csv
data/processed/test.csv
```

---

## 2 Model Training

Script:

```
ml/train.py
```

Operazioni:

* training modello
* valutazione performance
* salvataggio modello

Output:

```
models/churn_model.pkl
reports/model_metrics.csv
```

---

## 3 Prediction

Script:

```
ml/predict.py
```

Input:

* dati cliente
* modello salvato

Output:

```
results/predictions.csv
```


## 🌐 Backend API

File:

```
backend/api.py
```

Endpoint principali:

POST `/predict`

Input:

```
{
 "feature1": value,
 "feature2": value
}
```

Output:

```
{
 "churn_probability": 0.73,
 "prediction": 1
}
```

Avvio server:

```
uvicorn backend.api:app --reload
```

---

## 5 Frontend

File:

```
frontend/app.py
```

Funzioni:

* inserimento dati cliente
* chiamata API
* visualizzazione probabilità churn
* esportazione risultati CSV

Avvio:

```
streamlit run frontend/app.py
```

---

# Librerie utilizzate

```
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
streamlit
matplotlib
seaborn
```

---

# Suddivisione dei ruoli

## Data Engineer

Responsabilità:

* gestione dataset
* preprocessing dati
* pipeline di trasformazione

Script principali:

```
ml/preprocess.py
```

---

## Machine Learning Engineer

Responsabilità:

* training modello
* tuning
* valutazione performance

Script principali:

```
ml/train.py
ml/predict.py
```

---

## Backend Developer

Responsabilità:

* sviluppo API
* integrazione modello ML
* gestione endpoint

Script principali:

```
backend/api.py
```

---

## Frontend Developer

Responsabilità:

* sviluppo interfaccia utente
* integrazione con API
* visualizzazione risultati

Script principali:

```
frontend/app.py
```

---

# Setup del progetto

Creare ambiente virtuale:

```
python3.12 -m venv venv
```

Attivare ambiente:

Mac/Linux

```
source venv/bin/activate
```

Windows

```
venv\Scripts\activate
```

Installare dipendenze:

```
pip install -r requirements.txt
```

---

# Avvio del sistema

1 Avviare backend

```
uvicorn backend.api:app --reload
```

2 Avviare frontend

```
streamlit run frontend/app.py
```

---

# Output del sistema

I risultati delle predizioni vengono salvati in:

```
results/predictions.csv
```

Formato:

```
customer_id,feature1,feature2,churn_probability,prediction,timestamp
```

---

# Obiettivo del progetto

Dimostrare capacità di costruire un sistema completo di:

* Machine Learning
* Backend API
* Frontend interattivo
* Data pipeline
* Export dati

simulando un caso reale utilizzato dalle aziende per **ridurre la perdita di clienti (customer churn)**.
