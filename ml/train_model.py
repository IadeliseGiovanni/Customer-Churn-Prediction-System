"""Training script.

Cosa fa questo file:
- Legge i dati già processati (output di `ml/preprocessing.py`).
- Separa feature (X) e target (y = "Churn Value").
- Costruisce un preprocessor (numeriche + categoriche) e lo incapsula in una Pipeline.
- Addestra un modello XGBoost e salva l'intera Pipeline in `models/churn_pipeline_v1.joblib`.

Note importanti:
- `scale_pos_weight` serve a gestire classi sbilanciate (pochi churn vs molti non-churn):
  viene calcolato come neg/pos e passato al modello.
- L'ottimizzazione Optuna (se abilitata) fa cross-validation e massimizza l'F1-score.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def tune_xgb_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    n_trials: int = 30,
    random_state: int = 42,
) -> dict:
    """Tuning iperparametri XGBoost con Optuna.

    - Valuta i parametri con cross-validation (StratifiedKFold).
    - Usa `scoring="f1"` per ottimizzare l'equilibrio tra precision e recall.

    Ritorna:
        dict con i migliori iperparametri trovati (es. n_estimators, max_depth, ...)
    """

    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Peso della classe positiva: utile quando il dataset è sbilanciato.
    # Regola in modo semplice l'importanza degli esempi churn (1) rispetto ai non-churn (0).
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    scale_pos_weight = (neg / pos) if pos else 1.0

    # Parametri "base" sempre presenti, indipendenti dal trial.
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    def objective(trial: optuna.Trial) -> float:
        # Parametri che Optuna esplora (spazio di ricerca).
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        # Merge: i parametri del trial sovrascrivono quelli base se necessario.
        model = XGBClassifier(**{**base_params, **trial_params})

        # Pipeline = preprocessor + modello: garantisce la stessa trasformazione in train/predict.
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        # F1-score: usa le predizioni binarie (soglia di default 0.5).
        scores = cross_val_score(pipeline, X, y, scoring="f1", cv=cv, n_jobs=-1)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# --- PATHS (struttura repo) ---
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --- LOAD DATA (già processato) ---
df_train = pd.read_csv(PROC_DIR / "train_raw.csv")
print(f"Train shape: {df_train.shape}")

# Target binario: 0 = non churn, 1 = churn
target_col = "Churn Value"
X_raw = df_train.drop(columns=[target_col])
y_raw = df_train[target_col]

# Class imbalance handling: neg/pos
pos = int((y_raw == 1).sum())
neg = int((y_raw == 0).sum())
scale_pos_weight = (neg / pos) if pos else 1.0
print(f"scale_pos_weight: {scale_pos_weight:.3f}")
print(f"Churn rate (mean target): {y_raw.mean():.3f}")

# Identificazione automatica delle feature per tipo.
num_features = X_raw.select_dtypes(include=["number"]).columns.tolist()
cat_features = X_raw.select_dtypes(include=["object"]).columns.tolist()
print(f"Features -> numeric: {len(num_features)} | categorical: {len(cat_features)}")

# Pipeline numerica: imputazione + standardizzazione
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Pipeline categorica: imputazione + one-hot encoding
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Preprocessor unico: applica pipeline diverse a colonne diverse
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features),
    ]
)

# --- TUNING (opzionale) ---
USE_OPTUNA = True
best_params = tune_xgb_with_optuna(X_raw, y_raw, preprocessor) if USE_OPTUNA else {}

if USE_OPTUNA:
    print(f"Optuna best params: {best_params}")
else:
    print("Optuna disabled (USE_OPTUNA=False)")

# Parametri finali del modello.
# - `best_params` sovrascrive i default se Optuna è abilitato.
model_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
}
model_params.update(best_params)

model = XGBClassifier(**model_params)

# Pipeline finale: preprocessor + modello
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("Training model...")
pipeline.fit(X_raw, y_raw)
print("Training completed.")

# Salva l'intero oggetto Pipeline: garantisce coerenza tra train/evaluate/predict
model_path = MODELS_DIR / "churn_pipeline_v1.joblib"
joblib.dump(pipeline, model_path)
print("Pipeline addestrata e salvata correttamente.")
print(f"Model saved to: {model_path}")

# --- Interpretabilità (quick check): feature importance ---
# Nota: con OneHotEncoder le feature diventano molte; qui mostriamo solo le top 10.
import matplotlib.pyplot as plt

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False).head(10)

feature_importance_df.plot(kind="barh", x="feature", y="importance")
plt.title("Top 10 Feature per importanza nel Churn")
plt.show()
