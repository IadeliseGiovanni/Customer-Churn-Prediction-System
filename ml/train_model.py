"""Training script (XGBoost + Optuna).

Scopo del modulo
----------------
Questo file produce un *artifact* unico (una `sklearn.Pipeline`) che contiene:
- il preprocessing (imputazione, scaling, one-hot encoding)
- il modello (XGBoost)

Perché una Pipeline?
-------------------
Salvare una Pipeline (preprocessor + modello) evita mismatch tra:
- training: feature engineering + trasformazioni
- inferenza: input (API/UI) che deve subire esattamente le stesse trasformazioni

Input attesi
------------
- `data/processed/train_raw.csv` generato da `ml/preprocessing.py`.
  Deve contenere la colonna target `Churn Value` (0/1) e le feature.

Output prodotti
--------------
- `models/churn_pipeline_v1.joblib`: pipeline addestrata e pronta per `evaluate.py` e `predict.py`.

Concetti chiave
---------------
- Class imbalance: nel churn è comune avere pochi "1" (churn) e molti "0".
  `scale_pos_weight = neg/pos` aumenta il peso della classe positiva nella loss di XGBoost.
- Tuning obbligatorio: Optuna viene sempre eseguito e ottimizza l'F1-score.
  L'F1 è una metrica di compromesso tra precision e recall.

Nota su F1 e soglia
-------------------
`scoring="f1"` usa le predizioni binarie del modello (soglia tipica 0.5).
Se vuoi cambiare soglia in produzione, vedi `ml/predict.py` (parametro `threshold`).
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
    scale_pos_weight: float,
    n_trials: int = 30,
    random_state: int = 42,
) -> dict:
    """Ottimizza gli iperparametri di XGBoost con Optuna.

    Come funziona (in breve)
    ------------------------
    - Definisco uno *spazio di ricerca* (trial_params).
    - Per ogni trial:
        1) costruisco un modello con i parametri proposti
        2) lo inserisco in una Pipeline insieme al preprocessor
        3) valuto via cross-validation stratificata (mantiene la proporzione 0/1 nei fold)
        4) ritorno la media dell'F1-score

    Perché cross-validation?
    ------------------------
    Riduce il rischio di scegliere parametri buoni solo per una singola split.

    Ritorna:
        dict con i migliori iperparametri trovati.
    """

    # Import locali: Optuna non è necessario per usare API/predict/evaluate,
    # ma è necessario per eseguire questo training.
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Parametri *sempre* presenti, comuni a tutti i trial.
    # Li separiamo dai parametri ottimizzati per rendere il codice più leggibile.
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    def objective(trial: optuna.Trial) -> float:
        # Parametri che Optuna esplora (spazio di ricerca).
        # Nota: puoi restringere gli intervalli se vuoi ridurre tempo di training.
        trial_params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        # Unisco parametri base + trial.
        # Il trial non contiene `scale_pos_weight`: è derivato dai dati (imbalance).
        model = XGBClassifier(**{**base_params, **trial_params})

        # Pipeline: così il CV valuta l'intero flusso (preprocessing incluso).
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

        # StratifiedKFold: preserva le proporzioni 0/1 in ogni fold.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        # scoring="f1": ottimizza la metrica obiettivo richiesta.
        scores = cross_val_score(pipeline, X, y, scoring="f1", cv=cv, n_jobs=-1)
        return float(scores.mean())

    # Lo studio cerca di massimizzare l'F1.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# ------------------------------
# 1) Setup path di progetto
# ------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------
# 2) Caricamento dati di training
# ------------------------------
df_train = pd.read_csv(PROC_DIR / "train_raw.csv")
print(f"Train shape: {df_train.shape}")

# Target binario (0/1): churn/no churn
target_col = "Churn Value"
X_raw = df_train.drop(columns=[target_col])
y_raw = df_train[target_col]


# ------------------------------
# 3) Class imbalance handling
# ------------------------------
# `scale_pos_weight` è una euristica comune in XGBoost.
# Esempio: se ho 900 non-churn e 100 churn, scale_pos_weight=9.
pos = int((y_raw == 1).sum())
neg = int((y_raw == 0).sum())
scale_pos_weight = (neg / pos) if pos else 1.0
print(f"scale_pos_weight: {scale_pos_weight:.3f}")
print(f"Churn rate (mean target): {y_raw.mean():.3f}")


# ------------------------------
# 4) Definizione preprocessing
# ------------------------------
# Identifico automaticamente le feature numeriche e categoriche.
# Nota: se a monte cambiano i tipi (es. numeri come stringhe),
# questo elenco cambierà: meglio mantenere `preprocessing.py` coerente.
num_features = X_raw.select_dtypes(include=["number"]).columns.tolist()
cat_features = X_raw.select_dtypes(include=["object"]).columns.tolist()
print(f"Features -> numeric: {len(num_features)} | categorical: {len(cat_features)}")

# Pipeline numerica:
# - imputazione: sostituisce NaN con mediana
# - scaling: standardizza per aiutare modelli sensibili alla scala (in generale buona pratica)
numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Pipeline categorica:
# - imputazione: sostituisce i missing con stringa "Missing"
# - one-hot: converte le categorie in colonne binarie
categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# ColumnTransformer: applica trasformazioni diverse a subset di colonne.
preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features),
    ]
)


# ------------------------------
# 5) Tuning Optuna (obbligatorio)
# ------------------------------
# Nota: la durata dipende da n_trials e dalla dimensione del dataset.
best_params = tune_xgb_with_optuna(X_raw, y_raw, preprocessor, scale_pos_weight=scale_pos_weight)
print(f"Optuna best params: {best_params}")


# ------------------------------
# 6) Training finale con best params
# ------------------------------
# Parametri del modello finale = base + best_params.
# (best_params sovrascrive i default se presenti)
model_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "scale_pos_weight": scale_pos_weight,
}
model_params.update(best_params)
model = XGBClassifier(**model_params)

# Pipeline finale da salvare
pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("Training model...")
pipeline.fit(X_raw, y_raw)
print("Training completed.")


# ------------------------------
# 7) Salvataggio artifact
# ------------------------------
model_path = MODELS_DIR / "churn_pipeline_v1.joblib"
joblib.dump(pipeline, model_path)
print("Pipeline addestrata e salvata correttamente.")
print(f"Model saved to: {model_path}")


# ------------------------------
# 8) Quick interpretability check
# ------------------------------
# Con OneHotEncoder le feature diventano molte; qui mostriamo solo le top 10.
import matplotlib.pyplot as plt

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False).head(10)

feature_importance_df.plot(kind="barh", x="feature", y="importance")
plt.title("Top 10 Feature per importanza nel Churn")
plt.tight_layout()
#plt.show()
plt.savefig(MODELS_DIR / "feature_importance.png")
