# Customer Churn Prediction --- Machine Learning Project

## Project Goal

Build an end‑to‑end **Customer Churn Prediction system** that
demonstrates skills in:

-   Data Engineering
-   Machine Learning
-   Backend API development
-   Frontend data visualization

The system predicts whether a customer is likely to **churn** using the
**Telco Customer Churn dataset** and exposes predictions through an API
and a dashboard.

The project is designed to be **completed in 5 days by a team of 4
people**.

------------------------------------------------------------------------

# Project Architecture

    customer-churn-ml/

    data/
       raw/
          telco_churn.csv
       processed/
          train_raw.csv
          test_raw.csv

    ml/
       preprocessing.py
       train_model.py
       evaluate.py
       predict.py

    analysis/
       eda.py
       plots.py

    backend/
       api.py

    frontend/
       dashboard.py

    utils/
       data_loader.py

    models/
       churn_pipeline_v1.joblib

    outputs/
       metrics.csv
       predictions.csv
       plots/
          churn_distribution.png
          tenure_churn.png
          monthly_charges_churn.png
          correlation_matrix.png
          feature_importance.png

    requirements.txt
    README.md

------------------------------------------------------------------------

# Pipeline

The project follows a strict machine learning workflow.

    DATASET
       │
       ▼
    EDA + VISUALIZATION
    (analysis/)
       │
       ▼
    DATA CLEANING
    (ml/preprocessing.py)
       │
       ▼
    FEATURE ENGINEERING
    (ml/preprocessing.py)
       │
       ▼
    MODEL TRAINING
    (ml/train_model.py)
       │
       ▼
    MODEL EVALUATION
    (ml/evaluate.py)
       │
       ▼
    PREDICTIONS CSV
    (ml/predict.py)
       │
       ▼
    BACKEND API
    (backend/api.py)
       │
       ▼
    FRONTEND DASHBOARD
    (frontend/dashboard.py)

------------------------------------------------------------------------

# Team Roles

## Giovanni --- Data Engineer

### Files

utils/data_loader.py\
ml/preprocessing.py

### Responsibilities

-   Load the raw dataset
-   Clean missing values
-   Convert column formats
-   Remove duplicates
-   Split dataset into train/test

### Output

data/processed/train_raw.csv\
data/processed/test_raw.csv

------------------------------------------------------------------------

## Davide --- Machine Learning Engineer

### Files

ml/train_model.py\
ml/evaluate.py\
ml/predict.py

### Responsibilities

-   Build ML pipeline
-   Train RandomForest model
-   Evaluate model performance
-   Save trained model
-   Implement prediction functions

### Output

models/churn_pipeline_v1.joblib\
outputs/metrics.csv\
outputs/predictions.csv

------------------------------------------------------------------------

## Gabriele --- Backend + Frontend Developer

### Backend

backend/api.py

Endpoints POST /predict\
POST /predict_batch\
GET /results\
GET /download_results

### Frontend

frontend/dashboard.py

Features - Single prediction - Batch prediction via CSV - Visualization
of churn probability - Download predictions

------------------------------------------------------------------------

## Elisabetta --- Data Visualization & Analysis

### Files

analysis/eda.py\
analysis/plots.py

### Responsibilities

-   Exploratory Data Analysis
-   Create data visualizations
-   Provide insights about churn patterns

### Charts

outputs/plots/

------------------------------------------------------------------------

# Setup

Create environment

python3.12 -m venv venv\
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

------------------------------------------------------------------------

# Run Pipeline

Preprocessing

python ml/preprocessing.py

Training

python ml/train_model.py

Evaluation

python ml/evaluate.py

Generate plots

python analysis/plots.py

------------------------------------------------------------------------

# Run Application

Start backend

uvicorn backend.api:app --reload

Start dashboard

streamlit run frontend/dashboard.py

------------------------------------------------------------------------

# Expected Outputs

models/ churn_pipeline_v1.joblib

outputs/ metrics.csv\
predictions.csv\
plots/

------------------------------------------------------------------------

# Technologies

Python 3.12\
Pandas\
Scikit-learn\
FastAPI\
Streamlit\
Matplotlib\
Seaborn

------------------------------------------------------------------------

# Project Objective

This repository demonstrates a **complete machine learning system**
including:

-   Data preparation
-   Model training
-   Model deployment
-   Data visualization

The goal is to create a project suitable for **portfolio presentation to
companies**.
