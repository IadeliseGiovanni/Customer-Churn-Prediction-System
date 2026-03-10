"""Evaluation script.

Cosa fa questo file:
- Carica la Pipeline salvata in `models/`.
- Legge il test set già processato (output di `ml/preprocessing.py`).
- Calcola metriche principali (accuracy, precision, recall, f1, roc_auc).
- Salva `metrics.csv` e `classification_report.txt` in `outputs/`.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Carico la pipeline completa: stesso preprocessing del training + modello
pipeline = joblib.load(MODELS_DIR / "churn_pipeline_v1.joblib")

# Test set già pulito/ingegnerizzato
df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
print(f"Test shape: {df_test.shape}")

target_col = "Churn Value"
X_test = df_test.drop(columns=[target_col])
y_test = df_test[target_col]

# Predizioni (classe) e probabilità (serve per ROC-AUC)
preds = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1": f1_score(y_test, preds),
    "roc_auc": roc_auc_score(y_test, proba),
}

# Print rapidi per controllo manuale
print(f"Recall: {metrics['recall']:.2f}")
print(f"Accuracy: {metrics['accuracy']:.2f}")
print(f"Precision: {metrics['precision']:.2f}")
print(f"F1-score: {metrics['f1']:.2f}")
print(f"ROC-AUC: {metrics['roc_auc']:.2f}")

# Persistenza delle metriche in CSV
pd.DataFrame([metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)
print(f"Saved metrics to: {OUT_DIR / 'metrics.csv'}")

# Report testuale (per classe): precision/recall/f1/support
report = classification_report(y_test, preds)
(OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

# Matrice di confusione: visualizza errori (FP/FN)
ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("Matrice di Confusione - Errori del modello")
plt.show()
