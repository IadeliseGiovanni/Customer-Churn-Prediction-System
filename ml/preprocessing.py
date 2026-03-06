from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
PROC_DIR = ROOT / "data" / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])
    else:
        df = df.drop_duplicates()

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

def split_save(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
    X = df[feat_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    train = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train.to_csv(PROC_DIR / "train_raw.csv", index=False)
    test.to_csv(PROC_DIR / "test_raw.csv", index=False)

def main():
    df = pd.read_csv(RAW)
    df_clean = clean_raw(df)
    split_save(df_clean)

if __name__ == "__main__":
    main()