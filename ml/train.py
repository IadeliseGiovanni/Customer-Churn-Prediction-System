# ml/train.py
"""
Train di un modello con pipeline: Preprocessor (ColumnTransformer) + Classifier.
Salva models/churn_pipeline_v1.joblib e reports/model_metrics.csv
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    # load processed train/test (these are already transformed arrays saved by preprocess)
    train = pd.read_csv(PROC / "train.csv")
    test = pd.read_csv(PROC / "test.csv")

    X_train = train.drop("churn", axis=1)
    y_train = train["churn"]
    X_test = test.drop("churn", axis=1)
    y_test = test["churn"]

    # Build pipeline: here preprocessor is already applied (we saved transformed CSV),
    # but for clarity we'll build a pipeline that expects already numeric features.
    # If you prefer an end-to-end pipeline (raw -> preprocess -> model), re-load preprocessor_v1 and include it.
    preprocessor = joblib.load(MODELS / "preprocessor_v1.joblib")

    # end-to-end pipeline: preprocessor + classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])

    # grid search small (fast)
    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_

    # evaluate on test
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    # save model pipeline
    joblib.dump(best, MODELS / "churn_pipeline_v1.joblib")

    # save metrics
    dfm = pd.DataFrame([metrics])
    dfm.to_csv(REPORTS / "model_metrics.csv", index=False)

    print("Training completato. Modello salvato in models/churn_pipeline_v1.joblib")
    print("Metriche:", metrics)

if __name__ == "__main__":
    main()