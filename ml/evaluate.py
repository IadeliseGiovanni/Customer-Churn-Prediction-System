from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = joblib.load(MODELS_DIR / "churn_pipeline_v1.joblib")

df_test = pd.read_csv(PROC_DIR / "Telco_customer_churn.csv")
df_test['Total Charges'] = pd.to_numeric(df_test['Total Charges'], errors='coerce')

target_col = "Churn Value"

# Rimozione colonne non predittive (devono essere le stesse rimosse in training)
drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
             'Latitude', 'Longitude', 'Churn Label', 'Churn Value', 'Churn Reason']

X_test = df_test.drop(columns=drop_cols)
y_test = df_test[target_col]

preds = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:,1]

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "roc_auc": roc_auc_score(y_test, proba)
}

print(f"Recall: {metrics['recall']:.2f}") # Indica quanti dei 'churnati' reali abbiamo preso

pd.DataFrame([metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)

# Opzionale: Creazione di un report di classificazione testuale
from sklearn.metrics import classification_report
report = classification_report(y_test, preds)
with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)
    
# Aggiungi questo in evaluate.py
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("Matrice di Confusione - Errori del modello")
plt.show()