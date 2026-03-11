from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIGURAZIONE PERCORSI ---
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Esegue la pulizia e il feature engineering ottimizzato."""
    df = df.copy()
    
    # 1. PULIZIA NOMI E RIMOZIONE COLONNE
    df.columns = df.columns.str.strip().str.replace(" ", "") 
    cols_to_drop = [
        "CustomerID", "Count", "Country", "State", "City", "LatLong",
        "ZipCode", "ChurnReason", "ChurnScore", "ChurnLabel", "CLTV"
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print("🗑️ #Step1: Pulizia intestazioni e rimozione colonne irrilevanti")

    # 2. DIGITALIZZAZIONE (MAPPING GLOBALE)
    # Identifichiamo dinamicamente il target
    target = next((c for c in df.columns if "churn" in c.lower()), "ChurnValue")
    df.rename(columns={target: "ChurnValue"}, inplace=True)

    # Mapping centralizzato: più veloce e pulito rispetto a multipli .map()
    binary_map = {
        "Yes": 1, "No": 0, "Churned": 1, "Stayed": 0, 
        "Female": 1, "Male": 0, "No internet service": 0, "No phone service": 0
    }
    df.replace(binary_map, inplace=True)
    print("🎯 #Step2: Digitalizzazione variabili (Target, Gender e Servizi)")

    # 3. CONVERSIONE NUMERICA COERENTE
    num_cols = ["TotalCharges", "TenureMonths", "MonthlyCharges"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. FEATURE ENGINEERING (LOGICA DI BUSINESS)
    print("🛠️ #Step3: Generazione nuove Feature")
    # Calcolo spesa media evitando divisione per zero
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["TenureMonths"].replace(0, 1)

    # Calcolo NumServices (più rapido dopo la digitalizzazione)
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies", "PhoneService"
    ]
    available_services = [c for c in service_cols if c in df.columns]
    
    # Poiché abbiamo già convertito Yes in 1 e No in 0, basta sommare
    df["NumServices"] = df[available_services].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)
    df["ChargesPerService"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    return df.drop_duplicates()

def split_save(df: pd.DataFrame):
    """Gestisce lo split dei dati e il salvataggio fisico."""
    X = df.drop(columns=["ChurnValue"])
    y = df["ChurnValue"]

    # Split stratificato per mantenere le proporzioni del Churn
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Salvataggio CSV
    pd.concat([X_train, y_train], axis=1).to_csv(PROC_DIR / "train_raw.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROC_DIR / "test_raw.csv", index=False)
    print(f"💾 #Step4: File salvati con successo in {PROC_DIR}")

def main():
    print(f"🚀 #START: Processing {RAW_FILE.name}")
    try:
        if not RAW_FILE.exists():
            print(f"❌ Errore: File non trovato!")
            return
            
        df = pd.read_csv(RAW_FILE)
        df_processed = clean_raw(df)
        split_save(df_processed)
        
        print("\n" + "="*30)
        print(f"✅ DATASET PRONTO: {df_processed.shape[0]} righe, {df_processed.shape[1]} colonne")
        print("="*30)
        
    except Exception as e:
        print(f"❌ #ERROR: {e}")

if __name__ == "__main__":
    main()