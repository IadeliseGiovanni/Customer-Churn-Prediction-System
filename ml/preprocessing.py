from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. SETUP DEI PERCORSI (STRUTTURA PROGETTO)
# ROOT: Identifica la cartella principale del progetto
ROOT = Path(__file__).resolve().parents[1]
# RAW: Percorso dove si trova il file CSV originale
RAW = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
# PROC_DIR: Percorso della cartella dove salveremo i dati puliti
PROC_DIR = ROOT / "data" / "processed"

# MKDIR: Crea la cartella 'processed' se non esiste già
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    #COPY: Crea una copia del dataset per non modificare l'originale
    df = df.copy()
    
    # STRIP: Rimuove spazi vuoti accidentali dai nomi delle colonne
    df.columns = [c.strip() for c in df.columns]
    
    # COLS_TO_DROP: Lista delle 10 colonne inutili da eliminare
    cols_to_drop = [
        "CustomerID", "Count", "Country", "State", "City", "Lat Long" ,
        "Zip Code", "Churn Reason", "Churn Score", "Churn Label", "CLTV"
    ]
    # DROP: Rimuove fisicamente le colonne dal dataset
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    print(f"🗑️ #Step1: Rimosse colonne irrilevanti")

    # TARGET_COL: Cerca automaticamente la colonna che parla di 'churn'
    target_col = next((c for c in df.columns if "churn" in c.lower()), None)
    if target_col:
        # RENAME: Rinoma la colonna target in modo standard: 'Churn Value'
        df = df.rename(columns={target_col: "Churn Value"})
        # MAP: Converte i testi 'Yes/No' nei numeri 1 e 0 (Digitalizzazione)
        df["Churn Value"] = df["Churn Value"].map({
            "Yes": 1, "No": 0, "1": 1, "0": 0, 1: 1, 0: 0, "Churned": 1, "Stayed": 0
        })
        print("🎯 #Step2: Target 'Churn Value' configurato")

    # GENDER_CHECK: Se esiste la colonna Gender, la converte in numeri (0 e 1)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})
        print("🚻 #Step2.1: Colonna 'Gender' convertita (Female=1, Male=0)")

    # MAPPING: Gestisce i nomi delle colonne che hanno spazi (es. 'Total Charges')
    mapping = {
        "Total Charges": "TotalCharges",
        "Tenure Months": "TenureMonths",
        "Monthly Charges": "MonthlyCharges"
    }
    # RENAME_LOOP: Ciclo per rinominare le colonne togliendo gli spazi
    for old_name, new_name in mapping.items():
        col_found = next((c for c in df.columns if c.lower() == old_name.lower()), None)
        if col_found:
            df = df.rename(columns={col_found: new_name})

    # NUMERIC_CONVERSION: Trasforma le colonne da Testo a Numero e gestisce i valori vuoti
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", 0), errors="coerce").fillna(0)
    df["TenureMonths"] = pd.to_numeric(df.get("TenureMonths", 0), errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df.get("MonthlyCharges", 0), errors="coerce").fillna(0)


    # FEATURE ENGINEERING (CREAZIONE NUOVI DATI)
    print("🛠️ #Step3: Feature Engineering in corso...")

    # AVG_SPEND: Crea la variabile Spesa Media Mensile Storica
    df["Avg Monthly Spend"] = df["TotalCharges"] / df["TenureMonths"].replace(0, 1)

    # #LIST_SERVICES: Elenco di tutti i servizi tecnologici offerti
    service_cols = [
        "Multiple Lines", "Online Security", "Online Backup", 
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    # #CLEAN_SERVICE: Uniforma i valori "No internet" semplicemente in "No"
    available_service_cols = [c for c in service_cols if c in df.columns]
    df[available_service_cols] = df[available_service_cols].replace({
        "No internet service": "No",
        "No phone service": "No"
    })

    # SERVICES_CHECK: Aggiunge il servizio telefonico alla lista
    services = ["Phone Service"] + service_cols
    available_services = [c for c in services if c in df.columns]
    
    # NUM_SERVICES: QUESTA È LA TUA VARIABILE! Conta quanti 'Yes' ci sono per ogni cliente
    df["NumServices"] = (df[available_services] == "Yes").sum(axis=1)

    # UNIT_COST: Crea la variabile Costo per singolo Servizio attivo
    df["Charges per Service"] = df["MonthlyCharges"] / (df["NumServices"] + 1)

    # DUPLICATES: Rimuove eventuali righe identiche nel dataset
    df = df.drop_duplicates()
    return df

def split_save(df: pd.DataFrame):
    # X_Y_SPLIT: Separa i dati (X) dal risultato che vogliamo prevedere (y)
    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    # TRAIN_TEST_SPLIT: Divide i dati: 80% per allenare il modello, 20% per testarlo
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # CONCAT: Riunisce X e y per creare i file finali di training e test
    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # #TO_CSV: Salva fisicamente i file prodotti nella cartella processed
    train.to_csv(PROC_DIR / "train_raw.csv", index=False)
    test.to_csv(PROC_DIR / "test_raw.csv", index=False)
    print(f"💾 #Step4: File salvati in {PROC_DIR}")

def main():
    # LOG_START: Messaggio di inizio elaborazione
    print(f"🚀 #START: Analisi per {RAW.name}")
    try:
        # CHECK_FILE: Verifica se il file originale esiste prima di iniziare
        if not RAW.exists():
            print(f"❌ Errore: Il file {RAW} non esiste!")
            return
            
        # READ_CSV: Carica i dati dal file originale
        df = pd.read_csv(RAW)
        # PROCESS: Esegue tutta la pulizia e l'ingegneria dei dati
        df_clean = clean_raw(df)
        # SAVE: Esegue lo split e il salvataggio
        split_save(df_clean)
        
        # PREVIEW: Stampa a video i risultati finali per controllo
        print("\n" + "="*30)
        print("📊 ANTEPRIMA DATI FINALI:")
        print(df_clean.head())
        print(f"\n✅ Dataset pronto: {df_clean.shape[0]} righe e {df_clean.shape[1]} colonne")
        print("="*30)
        print("✨ #FINISH: Task completato con successo!")
        
    except Exception as e:
        # ERROR_LOG: Se succede un errore, lo stampa nel terminale
        print(f"❌ #ERROR: {e}")

if __name__ == "__main__":
    # EXECUTE: Avvia la funzione principale
    main()