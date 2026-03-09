from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(PROC_DIR / "Telco_customer_churn.csv")
print(df_raw.head())
print(df_raw.info())

# Definiamo le colonne da scartare perché non predittive o disponibili solo post-churn
drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 
             'Latitude', 'Longitude', 'Churn Label', 'Churn Score' , 'Churn Value', 'Churn Reason']

X_raw = df_raw.drop(columns=drop_cols)
y_raw = df_raw['Churn Value'] # Utilizziamo il valore numerico come target

# Identificazione automatica basata sui tipi di dato
num_features = X_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X_raw.select_dtypes(include=['object']).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_raw, y_raw)

joblib.dump(pipeline, MODELS_DIR / "churn_pipeline_v1.joblib")
print("Pipeline addestrata e salvata correttamente.")

import matplotlib.pyplot as plt
import pandas as pd

# Estraiamo i nomi delle feature dopo la trasformazione
# Nota: get_feature_names_out è disponibile dalle versioni recenti di sklearn
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

# Estraiamo le importanze dal modello RandomForest
importances = pipeline.named_steps['model'].feature_importances_

# Creiamo un DataFrame per la visualizzazione
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

# Plot
feature_importance_df.plot(kind='barh', x='feature', y='importance')
plt.title("Top 10 Feature per importanza nel Churn")
plt.show()