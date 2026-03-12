"""Evaluation script.

Scopo del modulo
----------------
Valuta il modello salvato da ml/train_model.py sul test set prodotto da ml/preprocessing.py.

Input
-----
- models/churn_pipeline_v1.joblib
- data/processed/test_raw.csv (colonna target "ChurnValue")

Output
------
- outputs/metrics.csv: metriche riassuntive
- outputs/classification_report.txt: report dettagliato per classe
- outputs/confusion_matrix.png
"""

from __future__ import annotations

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


def evaluate_model(
    model_path: Path | str,
    test_data_path: Path | str,
    out_dir: Path | str,
    plot_out_dir: Path | str,
    target_col: str = "ChurnValue",
) -> dict:
    model_path = Path(model_path)
    test_data_path = Path(test_data_path)
    out_dir = Path(out_dir)
    plot_out_dir = Path(plot_out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_path)
    df_test = pd.read_csv(test_data_path)

    if target_col not in df_test.columns:
        raise KeyError(f"target_col '{target_col}' not found in test data columns")

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    report = classification_report(y_test, preds)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("Matrice di Confusione - Errori del modello")
    plt.tight_layout()
    plt.savefig(plot_out_dir / "confusion_matrix.png")
    plt.close()

    return metrics


def main() -> None:
    evaluate_model(
        model_path=MODELS_DIR / "churn_pipeline_v1.joblib",
        test_data_path=PROC_DIR / "test_raw.csv",
        out_dir=OUT_DIR,
        plot_out_dir=OUT_DIR,
        target_col="ChurnValue",
    )


if __name__ == "__main__":
    main()
