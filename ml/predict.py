"""Prediction helper.

Scopo del modulo
----------------
Questo modulo viene importato dal backend FastAPI (`backend/api.py`) e serve a:
- prendere un singolo record (dict) inviato da API/UI
- trasformarlo in un DataFrame compatibile con la Pipeline
- restituire probabilità di churn + predizione binaria

Problema che risolve
--------------------
In produzione spesso arrivano record *parziali* (non tutte le feature).
La Pipeline però è stata addestrata su un set completo di colonne.
Qui:
- ricaviamo le colonne attese dal `preprocessor` salvato nella pipeline
- aggiungiamo le colonne mancanti come NA (gli imputers le gestiranno)

Soglia di decisione
-------------------
Il modello produce una probabilità `proba`.
- Se `threshold` è None: usiamo `pipeline.predict` (default del modello)
- Se `threshold` è un float (0-1): decidiamo churn se `proba >= threshold`

Print di verifica
----------------
Questo file stampa volutamente messaggi di verifica (sempre attivi) per:
- confermare il caricamento del modello
- mostrare lo schema in input e l'allineamento alle colonne attese
- mostrare probabilità, soglia e predizione finale

Nota: in un ambiente di produzione questi print possono essere rumorosi.
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "churn_pipeline_v1.joblib"

# Carico una sola volta la pipeline (utile per performance nel backend)
pipeline = joblib.load(MODEL)
print(f"[predict] Loaded model: {MODEL}")


def predict_record(record: dict, threshold: float | None = None) -> dict:
    """Esegue una predizione su un singolo record.

    Args:
        record: dizionario con (alcune) feature del cliente.
        threshold: soglia opzionale per convertire la probabilità in classe.

    Returns:
        dict con probabilità e predizione binaria.
    """

    # 1) Dict -> DataFrame (una sola riga)
    df = pd.DataFrame([record]).copy()
    print(f"[predict] input_cols={list(df.columns)}")

    # 2) Normalizzazione nomi (alcuni input UI/API usano nomi diversi dal training)
    rename_map = {
        "tenure": "TenureMonths",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges",
        "Total Charges": "TotalCharges",
    }
    rename_cols = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    if rename_cols:
        print(f"[predict] renamed_cols={rename_cols}")
    df = df.rename(columns=rename_cols)

    # 3) Allineamento schema: stesse colonne viste in training
    # Recupero l'elenco delle feature attese dal ColumnTransformer salvato in pipeline.
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is not None and hasattr(preprocessor, "transformers"):
        expected_cols: list[str] = []
        for name, _, cols in preprocessor.transformers:
            if cols is None or cols in ("drop", "passthrough"):
                continue
            expected_cols.extend(list(cols))

        missing_cols = [c for c in expected_cols if c not in df.columns]
        print(f"[predict] expected_cols={len(expected_cols)}")
        print(f"[predict] missing_cols={len(missing_cols)} sample={missing_cols[:10]}")

        # Aggiungo colonne mancanti come NA: gli imputers nel preprocessing le gestiranno.
        for col in missing_cols:
            df[col] = np.nan

        # Feature derivata: se non arriva dal client ma possiamo calcolarla, la calcoliamo.
        if "Avg Monthly Spend" in expected_cols and "Avg Monthly Spend" not in record:
            if "TotalCharges" in df.columns and "TenureMonths" in df.columns:
                total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
                tenure = pd.to_numeric(df["TenureMonths"], errors="coerce").replace(0, 1)
                df["Avg Monthly Spend"] = total_charges / tenure
                print("[predict] computed feature: Avg Monthly Spend")

        # Conversione numeriche: spesso UI/API inviano numeri come stringhe.
        num_cols: list[str] = []
        for name, _, cols in preprocessor.transformers:
            if name == "num":
                num_cols = list(cols)
                break
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Riordino: alcune pipeline si aspettano un ordine consistente.
        df = df[expected_cols]
        print(f"[predict] aligned_shape={df.shape}")
    else:
        # Caso raro: pipeline senza preprocessor (non atteso in questo progetto)
        print("[predict] WARNING: preprocessor not found; skipping schema alignment")

    # 4) Probabilità churn (classe positiva = 1)
    proba = float(pipeline.predict_proba(df)[0][1])
    print(f"[predict] proba={proba:.4f}")

    # 5) Classe binaria
    if threshold is None:
        # Usa la logica di default del modello (tipicamente soglia 0.5)
        pred = int(pipeline.predict(df)[0])
        print("[predict] threshold=None -> using pipeline.predict")
    else:
        thr = float(threshold)
        if not (0.0 < thr < 1.0):
            raise ValueError("threshold must be between 0 and 1")
        pred = int(proba >= thr)
        print(f"[predict] threshold={thr:.3f} -> pred=(proba>=threshold)")

    print(f"[predict] output pred={pred}")

    return {
        "churn_probability": proba,
        "prediction": pred,
    }

if __name__ == "__main__":
    # Esempio rapido: esegui `python ml/predict.py` per vedere i print di verifica.
    # Puoi modificare questo record con valori reali dal dataset.
    sample_record = {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "Total Charges": 840.0,
        "Contract": "Month-to-month",
        "Internet Service": "Fiber optic",
        "Payment Method": "Electronic check",
        "Paperless Billing": "Yes",
        "Phone Service": "Yes",
    }
    print("[predict] Running sample prediction...")
    result = predict_record(sample_record, threshold=0.6)
    print(f"[predict] result={result}")


