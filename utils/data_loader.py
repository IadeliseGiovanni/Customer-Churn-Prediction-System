from pathlib import Path
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
   
   
   
   
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} non trovato.")
    return pd.read_csv(p)