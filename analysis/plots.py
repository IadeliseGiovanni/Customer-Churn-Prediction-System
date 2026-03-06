import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_churn.csv"
OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

sns.countplot(x="Churn", data=df)
plt.savefig(OUT / "churn_distribution.png")
plt.close()