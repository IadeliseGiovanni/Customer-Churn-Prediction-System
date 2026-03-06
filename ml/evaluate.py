from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = joblib.load(MODELS_DIR / "churn_pipeline_v1.joblib")

df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
X_test = df_test.drop(columns=["Churn"])
y_test = df_test["Churn"]

preds = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:,1]

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "roc_auc": roc_auc_score(y_test, proba)
}

pd.DataFrame([metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)