"""Training script (XGBoost + Optuna).

Scopo del modulo
----------------
Questo file produce un artifact unico (una sklearn.Pipeline) che include:
- preprocessing (imputazione, scaling, one-hot encoding)
- modello (XGBoost)

Perche una Pipeline
-------------------
Salvare una Pipeline (preprocessor + modello) evita mismatch tra:
- training: feature engineering + trasformazioni
- inferenza: input (API/UI) che deve subire le stesse trasformazioni

Input attesi
------------
- data/processed/train_raw.csv generato da ml/preprocessing.py
- Colonna target: "Churn Value" (0/1) oppure "ChurnValue" (fallback)

Output prodotti
--------------
- models/churn_pipeline_v1.joblib: pipeline addestrata, usata da evaluate.py e predict.py

Concetti chiave
---------------
- Class imbalance: scale_pos_weight = neg/pos bilancia la classe positiva in XGBoost
- Tuning con Optuna: ottimizza F1 su cross-validation stratificata

Nota su F1 e soglia
-------------------
scoring="f1" usa predizioni binarie con soglia 0.5.
La soglia in produzione puo essere gestita in ml/predict.py (parametro threshold).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# ------------------------------
# Setup path di progetto
# ------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Costruisce il ColumnTransformer coerente con lo schema del training set."""
    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ]
    )


def tune_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    scale_pos_weight: float,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """Ottimizza gli iperparametri di XGBoost con Optuna."""

    import optuna

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        model = XGBClassifier(**{**base_params, **trial_params})
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = cross_val_score(pipeline, X, y, scoring="f1", cv=cv, n_jobs=-1)
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def run_training_pipeline(
    train_data_path: Path | str,
    models_dir: Path | str,
    target_col: str = "Churn Value",
    n_trials: int = 50,
) -> str:
    """Allena e salva la pipeline (preprocessing + modello) per l'uso da API."""

    train_data_path = Path(train_data_path)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(train_data_path)

    # fallback robusto tra schema con/senza spazi
    if target_col not in df_train.columns:
        fallback = "ChurnValue" if target_col == "Churn Value" else "Churn Value"
        if fallback in df_train.columns:
            target_col = fallback
        else:
            raise KeyError(f"target_col '{target_col}' not found in training data columns")

    print(f"Train shape: {df_train.shape}")

    X_raw = df_train.drop(columns=[target_col])
    y_raw = df_train[target_col]

    pos = int((y_raw == 1).sum())
    neg = int((y_raw == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    print(f"scale_pos_weight: {scale_pos_weight:.3f}")
    print(f"Churn rate (mean target): {y_raw.mean():.3f}")

    preprocessor = build_preprocessor(X_raw)
    print(
        f"Features -> numeric: {len(X_raw.select_dtypes(include=['number']).columns)} | "
        f"categorical: {len(X_raw.select_dtypes(include=['object']).columns)}"
    )

    best_params = tune_xgb_with_optuna(
        X_raw,
        y_raw,
        preprocessor,
        scale_pos_weight=scale_pos_weight,
        n_trials=n_trials,
    )
    print(f"Optuna best params: {best_params}")

    model_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    model_params.update(best_params)

    # Early stopping su split interno per stimare n_estimators finale
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )

    preprocessor_es = clone(preprocessor)
    X_tr_p = preprocessor_es.fit_transform(X_tr)
    X_val_p = preprocessor_es.transform(X_val)

    model_es = XGBClassifier(**{**model_params, "early_stopping_rounds": 30})
    model_es.fit(X_tr_p, y_tr, eval_set=[(X_val_p, y_val)], verbose=False)

    best_iteration = getattr(model_es, "best_iteration", None)
    if best_iteration is not None:
        model_params["n_estimators"] = int(best_iteration) + 1
        print(f"Early stopping -> n_estimators finale: {model_params['n_estimators']}")

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier(**model_params))])

    # Fit finale su tutto il dataset
    pipeline.fit(X_raw, y_raw)

    # ------------------------------
    # Diagnostica overfitting
    # ------------------------------
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
    )

    diag_pipeline = clone(pipeline)
    diag_pipeline.fit(X_tr2, y_tr2)

    pred_tr = diag_pipeline.predict(X_tr2)
    proba_tr = diag_pipeline.predict_proba(X_tr2)[:, 1]
    pred_te = diag_pipeline.predict(X_te2)
    proba_te = diag_pipeline.predict_proba(X_te2)[:, 1]

    m_tr = _metrics(y_tr2, pred_tr, proba_tr)
    m_te = _metrics(y_te2, pred_te, proba_te)

    labels = list(m_tr.keys())
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, [m_tr[k] for k in labels], width, label="train")
    plt.bar(x + width / 2, [m_te[k] for k in labels], width, label="test")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title("Diagnostica overfitting: metriche train vs test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(models_dir / "overfitting_train_vs_test.png")
    plt.close()

    train_sizes, train_scores, test_scores = learning_curve(
        clone(pipeline),
        X_raw,
        y_raw,
        cv=3,
        scoring="f1",
        train_sizes=np.linspace(0.2, 1.0, 6),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_mean, "o-", label="train F1")
    plt.plot(train_sizes, test_mean, "o-", label="cv F1")
    plt.xlabel("Numero esempi di training")
    plt.ylabel("F1-score")
    plt.title("Learning curve (F1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(models_dir / "learning_curve_f1.png")
    plt.close()

    # ------------------------------
    # Feature importance
    # ------------------------------
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values(by="importance", ascending=False).head(10)

    df_imp.plot(kind="barh", x="feature", y="importance")
    plt.title("Top 10 Feature per importanza nel Churn")
    plt.tight_layout()
    plt.savefig(models_dir / "feature_importance.png")
    plt.close()

    # ------------------------------
    # Salvataggio artifact
    # ------------------------------
    model_path = models_dir / "churn_pipeline_v1.joblib"
    joblib.dump(pipeline, model_path)
    print("Pipeline addestrata e salvata correttamente.")
    print(f"Model saved to: {model_path}")

    return str(model_path)


def main() -> None:
    run_training_pipeline(
        train_data_path=PROC_DIR / "train_raw.csv",
        models_dir=MODELS_DIR,
        target_col="Churn Value",
        n_trials=50,
    )


if __name__ == "__main__":
    main()