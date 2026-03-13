import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "telco_churn.csv"
OUT = ROOT / "outputs" / "grafici"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED)

sns.countplot(x="Churn", data=df)
plt.savefig(OUT / "churn_distribution.png")
plt.close()


def generate_plot():
    """Carica i dati e genera il grafico sulla distribuzione del churn."""
    df = pd.read_csv(PROCESSED)
    
    sns.countplot(x="Churn", data=df)
    plt.savefig(OUT / "churn_distribution.png")
    plt.close()
    print(f"Grafico salvato in: {OUT / 'churn_distribution.png'}")

