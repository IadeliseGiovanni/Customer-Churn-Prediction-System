from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Percorsi dinamici basati sulla tua struttura (Desktop > Progetto)
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
PROC_DIR = ROOT / "data" / "processed"

# Crea la cartella processed se non esiste
PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Pulizia nomi colonne (rimuove spazi invisibili all'inizio/fine)
    df.columns = [c.strip() for c in df.columns]
    
    # 2. RICERCA SMART DELLA COLONNA TARGET (Churn)
    # Dato che il tuo file ha colonne come 'Count', 'Country', cerchiamo 'Churn' ovunque
    target_col = None
    for c in df.columns:
        if "churn" in c.lower(): # Cerca la parola churn ignorando maiuscole
            target_col = c
            break
    
    if target_col:
        print(f"🎯 Giovanni, ho trovato la colonna target: si chiama '{target_col}'")
        # La rinominiamo "Churn" per uniformità nel resto del progetto
        df = df.rename(columns={target_col: "Churn"})
        
        # Mapping flessibile: gestisce Yes/No, stringhe "1"/"0" e numeri 1/0
        df["Churn"] = df["Churn"].map({
            "Yes": 1, "No": 0, 
            "1": 1, "0": 0, 
            1: 1, 0: 0,
            "Churned": 1, "Stayed": 0
        })
        print("✅ Colonna Churn convertita correttamente in formato numerico.")
    else:
        print(f"❌ Colonne disponibili nel tuo file: {df.columns.tolist()}")
        raise KeyError("ERRORE: Non trovo nessuna colonna che contenga la parola 'Churn'!")

    # 3. Conversione TotalCharges (spesso letta come oggetto/testo)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        print("✅ Colonna TotalCharges convertita in numerica.")

    # 4. Gestione duplicati (usa customerID se esiste, altrimenti riga intera)
    id_col = next((c for c in df.columns if "id" in c.lower()), None)
    if id_col:
        df = df.drop_duplicates(subset=[id_col])
        print(f"✅ Duplicati rimossi basandosi sulla colonna '{id_col}'.")
    else:
        df = df.drop_duplicates()
        print("✅ Duplicati rimossi (riga intera).")

    return df

def split_save(df: pd.DataFrame):
    # Identifichiamo le feature (tutte tranne l'eventuale ID e il Churn)
    id_cols = [c for c in df.columns if "id" in c.lower()]
    feat_cols = [c for c in df.columns if c not in id_cols and c != "Churn"]
    
    X = df[feat_cols]
    y = df["Churn"]

    # Split 80% Training e 20% Test
    # Stratify assicura che la % di Churn sia uguale in entrambi i set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Ricreiamo i file CSV completi
    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    # Salvataggio finale
    train.to_csv(PROC_DIR / "train_raw.csv", index=False)
    test.to_csv(PROC_DIR / "test_raw.csv", index=False)
    print(f"💾 File generati in: {PROC_DIR}")

def main():
    print(f"🚀 Analisi file: {RAW.name}")
    try:
        # Carichiamo il file RAW dal tuo percorso data/raw/
        df = pd.read_csv(RAW)
        df_clean = clean_raw(df)
        split_save(df_clean)
        print("✨ Task completato! Ora puoi fare il Push su GitHub.")
    except Exception as e:
        print(f"❌ ERRORE DURANTE L'ESECUZIONE: {e}")

if __name__ == "__main__":
    main()