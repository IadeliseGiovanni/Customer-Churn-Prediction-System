# 🚀 Customer Churn Prediction API

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost_Pipeline-orange?logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-blueviolet)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Frontend](https://img.shields.io/badge/Frontend-HTML5_JS-orange?logo=html5&logoColor=white)

## Progetto
<img width="1328" height="1238" alt="image" src="https://github.com/user-attachments/assets/8b7f60de-2480-4d75-bbd0-b7f739234630" />
## 🎯 Obiettivo del Progetto
Sviluppare un sistema **API-First** per la previsione dell'abbandono clienti (Churn) nel settore Telco.
Il servizio espone endpoint REST per l'inferenza real-time, progettati per massimizzare il valore di business intercettando il maggior numero di clienti a rischio (**Recall Focus**).

Il progetto include:
1.  **Pipeline ML Robusta**: Preprocessing, Feature Engineering avanzato e Tuning con Optuna.
2.  **Quality Gates**: Controlli automatici su Recall, F1 e AUC per prevenire regressioni del modello.
3.  **Deployment**: API performanti documentate automaticamente (Swagger UI).

---

## 📂 Struttura del Progetto

L'organizzazione del codice separa la pipeline di Machine Learning (training/valutazione) dalla logica di analisi e inferenza.

```plaintext
Customer-Churn-Prediction-System/
│
├── 📂 data/                   # Dataset Raw e Processati
├── 📂 ml/                     # Pipeline ML
│   ├── preprocessing.py       # Pulizia, Split Train/Test, Feature Engineering
│   ├── train_model.py         # Training XGBoost, Tuning (Optuna), Diagnostica
│   ├── evaluate.py            # Calcolo metriche, Quality Gates, Report
│   └── predict.py             # Entrypoint per inferenza (usato da API)
├── 📂 analysis/               # Analisi Esplorativa
│   └── plots.py               # Generazione batch di grafici business & insight
├── 📂 backend/                # API Service
│   └── api.py                 # Endpoint FastAPI
├── 📂 frontend/               # Dashboard Web
│   ├── index.html             # UI (HTML5)
│   ├── style.css              # Stili (CSS3)
│   └── script.js              # Logica Client (JS)
├── 📂 models/                 # Artifacts
│   └── churn_pipeline_v1.joblib
└── 📂 outputs/                # Report e Immagini
    ├── metrics.csv
    └── plots/                 # Grafici generati (EDA, Confusion Matrix, ecc.)
```

---

## ⚙️ Pipeline ML

La pipeline è orchestrata per trasformare i dati grezzi in insight azionabili.

### 1. Preprocessing (`ml/preprocessing.py`)
-   **Cleaning**: Gestione valori mancanti e conversione tipi.
-   **Feature Engineering**:
    -   `NumServices`: Indice di "stickiness" basato sui servizi attivi.
    -   `AvgMonthlySpend`: Spesa media normalizzata sulla tenure.
    -   Encoding variabili categoriche e target.

### 2. Training & Tuning (`ml/train_model.py`)
-   **Modello**: XGBoost Classifier.
-   **Ottimizzazione**: Utilizzo di **Optuna** per massimizzare la metrica **F2-Score** (privilegiando la Recall per non perdere clienti churn).
-   **Diagnostica**: Generazione automatica di curve di apprendimento e grafici overfitting.

### 3. Valutazione (`ml/evaluate.py`)
-   **Metriche**: Accuracy, Precision, Recall, F1, ROC-AUC.
-   **Quality Gate**: Il modello viene accettato solo se:
    -   Recall >= 0.78 🛡️
    -   ROC AUC >= 0.84 🛡️

---

##  Quick Start

### 1. Installazione
Clona la repository e installa le dipendenze:

```bash
# Crea virtual env (Python 3.10+)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Installa requisiti
pip install -r requirements.txt
```

### 2. Esecuzione Pipeline ML
Esegui i moduli in sequenza per rigenerare il modello:

```bash
# 1. Preprocessing e Feature Engineering
python ml/preprocessing.py

# 2. Training e Ottimizzazione Iperparametri
python ml/train_model.py

# 3. Valutazione e Report
python ml/evaluate.py
```

### 3. Generazione Report Grafici
Crea i grafici per l'analisi business in `outputs/plots`:

```bash
python analysis/plots.py
```

### 4. Avvio Applicazione Web

Lancia il Backend API e apri la Dashboard Frontend:

**Terminale 1 - Backend API:**
```bash
uvicorn backend.api:app --reload
```


```

---

## 📊 Expected Outputs

Al termine dell'esecuzione, troverai nella cartella `outputs/`:

-   **`metrics.csv`**: Le performance dettagliate del modello.
-   **`plots/confusion_matrix_xgb.png`**: Per visualizzare i Falsi Negativi (Clienti persi).
-   **`plots/feature_importance.png`**: Le variabili che impattano di più sul churn (es. Contratto mensile, Fibra Ottica).
-   **`plots/overfitting_report.csv`**: Verifica della stabilità tra Train e Test set.

---

## 🛠️ Tecnologie Utilizzate
*   **Core:** Python 3.10+, Pandas, NumPy
*   **ML:** Scikit-learn, XGBoost, Optuna, Joblib
*   **Vis:** Matplotlib, Seaborn
*   **Web:** FastAPI, HTML, CSS JS, Pydantic
*   

Team Roles,
Giovanni --- Data Engineer,
Files,
utils/data_loader.py\
ml/preprocessing.py

Responsibilities,
Load the raw dataset,
Clean missing values,
Convert column formats,
Remove duplicates,
Split dataset into train/test,

Output,
data/processed/train_raw.csv\
data/processed/test_raw.csv

------------------------------------------------------------------------

Davide --- Machine Learning Engineer,
Files,
ml/train_model.py\
ml/evaluate.py\
ml/predict.py

Responsibilities,
Build ML pipeline,
Train XGBoost model tuned with Optuna (F2-oriented),
Evaluate model performance,
Save trained model,
Implement prediction functions,

Output,
models/churn_pipeline_v1.joblib\
outputs/metrics.csv\
outputs/classification_report.txt

------------------------------------------------------------------------

Gabriele --- Backend + Frontend Developer,
Backend,
backend/api.py

Endpoints GET /\
POST /predict\
POST /preprocess\
POST /train\
POST /evaluate\
POST /generate-plots

Frontend,
frontend/dashboard.py

Features - Single prediction via API - Visualization support for churn probability - ML pipeline trigger from backend endpoints

------------------------------------------------------------------------

Elisabetta --- Data Visualization & Analysis,
Files,
analysis/eda.py\
analysis/plots.py

Responsibilities,
Exploratory Data Analysis,
Create data visualizations,
Provide insights about churn patterns,

Charts,
outputs/plots/

