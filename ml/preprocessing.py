from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. SETUP DEI PERCORSI (STRUTTURA PROGETTO)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # PULIZIA: Rimuove spazi bianchi invisibili nei nomi delle colonne
    df.columns = [c.strip() for c in df.columns]
    
    # ELIMINAZIONE COLONNE: Rimosse colonne non utili (ID, Geografici, ecc.)
    cols_to_drop = [
        "CustomerID", "Count", "Country", "State", "City", 
        "Zip Code", "Churn Reason", "Churn Score", "Churn Label", "CLTV"
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"🗑️ #Step1: Rimosse colonne irrilevanti")

    # TARGET BINDING: Identificazione e rinomina del Churn
    target_col = next((c for c in df.columns if "churn" in c.lower()), None)
    if target_col:
        df = df.rename(columns={target_col: "Churn Value"})
        df["Churn Value"] = df["Churn Value"].map({
            "Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0, "Churned": 1, "Stayed": 0
        })
        print("🎯 #Step2: Target 'Churn Value' configurato")

        # #GENDER CONVERSION: Trasformiamo Male/Female in numeri
    if "Gender" in df.columns:
        # Decidiamo: Female = 1, Male = 0 (o viceversa, l'importante è la coerenza)
        df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})
        print("🚻 #Step2.1: Colonna 'Gender' convertita (Female=1, Male=0)")

    # CORREZIONE NOMI COLONNE CRITICHE: Gestione flessibile degli spazi
    mapping = {
        "Total Charges": "TotalCharges",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges"
    }
    for old_name, new_name in mapping.items():
        col_found = next((c for c in df.columns if c.lower() == old_name.lower()), None)
        if col_found:
            df = df.rename(columns={col_found: new_name})

    # #DATA FIX: Conversione forzata in numeri
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    df["TenureMonths"] = pd.to_numeric(df.get("TenureMonths", 0), errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df.get("MonthlyCharges", 0), errors="coerce").fillna(0)

    
    # FEATURE ENGINEERING (CREAZIONE NUOVI DATI)
    print("🛠️ #Step3: Feature Engineering in corso...")

    df["Avg Monthly Spend"] = df["TotalCharges"] / df["TenureMonths"].replace(0, 1)

    df["Tenure Group"] = pd.cut(
        df["TenureMonths"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
        include_lowest=True
    )

    service_cols = [
        "Multiple Lines", "Online Security", "Online Backup", 
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    available_service_cols = [c for c in service_cols if c in df.columns]
    df[available_service_cols] = df[available_service_cols].replace({
        "No internet service": "No",
        "No phone service": "No"
    })

    services = ["Phone Service"] + service_cols
    available_services = [c for c in services if c in df.columns]
    df["NumServices"] = (df[available_services] == "Yes").sum(axis=1)

    df["Charges per Service"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    df = df.drop_duplicates()
    return df

def split_save(df: pd.DataFrame):
    # #DATA SPLIT: Features vs Target
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train.to_csv(PROC_DIR / "train_raw.csv", index=False)
    test.to_csv(PROC_DIR / "test_raw.csv", index=False)
    print(f"💾 #Step4: File salvati in {PROC_DIR}")

# === QUESTA PARTE MANCAVA NEL TUO CODICE ===
def main():
    print(f"🚀 #START: Analisi per {RAW.name}")
    try:
        if not RAW.exists():
            print(f"❌ Errore: Il file {RAW} non esiste!")
            return
            
        df = pd.read_csv(RAW)
        df_clean = clean_raw(df)
        split_save(df_clean)
        
        print("\n" + "="*30)
        print("📊 ANTEPRIMA DATI FINALI:")
        print(df_clean.head())
        print(f"\n✅ Dataset pronto: {df_clean.shape[0]} righe e {df_clean.shape[1]} colonne")
        print("="*30)
        print("✨ #FINISH: Task completato con successo!")
        
    except Exception as e:
        print(f"❌ #ERROR: {e}")

if __name__ == "__main__":
    main()