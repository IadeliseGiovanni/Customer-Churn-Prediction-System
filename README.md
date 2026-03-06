# Customer Churn ML Project

Pipeline:
1. Preprocessing dataset
2. Training ML model
3. Evaluation
4. API backend
5. Streamlit dashboard

Run order:

python ml/preprocessing.py
python ml/train_model.py
python ml/evaluate.py

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

streamlit run frontend/dashboard.py