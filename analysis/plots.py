import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.frameon': False
})

# Palette coerente: Verde per chi resta (0), Arancio/Rosso per chi se ne va (1)
# Usiamo stringhe "0" e "1" per evitare errori di mappatura
PALETTE = {"0": "#2a9d8f", "1": "#e76f51"} 

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco_customer_churn.csv"
OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Caricamento e normalizzazione dati
df = pd.read_csv(RAW)
# Convertiamo in stringa per matchare la palette e risolvere i FutureWarning
df["Churn Value"] = df["Churn Value"].astype(str)

# --- DISTRIBUZIONE CHURN CON PERCENTUALI ---
plt.figure(figsize=(8, 5))

# Creazione del plot
ax = sns.countplot(x="Churn Value", hue="Churn Value", data=df, palette=PALETTE, legend=False)

# Calcolo del totale per le percentuali
total = len(df["Churn Value"])

# Iterazione sulle barre per aggiungere le etichette
for p in ax.patches:
    # Calcolo della percentuale
    percentage = f'{100 * p.get_height() / total:.1f}%'
    
    # Posizionamento del testo
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    
    ax.annotate(percentage, 
                (x, y), 
                ha='center', 
                va='bottom', 
                fontsize=12, 
                weight='bold', 
                xytext=(0, 5), 
                textcoords='offset points')

plt.title("Distribuzione Generale del Churn (%)")
plt.ylabel("Numero di Clienti")
plt.savefig(OUT / "churn_distribution_percent.png", dpi=300, bbox_inches='tight')
plt.close()
'''
#### Boxplot: variabili numeriche vs churn
# Tenure vs Churn  chi churn ha tenure molto più bassa
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn Value", y="Tenure Months", hue="Churn Value", data=df, palette=PALETTE, width=0.5, legend=False)
plt.title("Tenure Months vs Churn")
plt.savefig(OUT / "TenureMonthsvsChurn.png", dpi=300)
plt.close()

# Monthly Charges vs Churn  chi paga di più tende a churnare di più
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn Value", y="Monthly Charges", hue="Churn Value", data=df, palette=PALETTE, width=0.5, legend=False)
plt.title("Monthly Charges vs Churn")
plt.savefig(OUT / "MonthlyChargesvsChurn.png", dpi=300)
plt.close()
'''

#### Analisi Distribuzione: variabili numeriche vs Churn
# Utilizziamo il KDE Plot per vedere la "densità" della probabilità

# Tenure vs Churn: chi churn ha tenure molto più bassa
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df[df["Churn Value"] == "0"], x="Tenure Months", fill=True, label="Stayed", color=PALETTE["0"], alpha=0.5)
sns.kdeplot(data=df[df["Churn Value"] == "1"], x="Tenure Months", fill=True, label="Churned", color=PALETTE["1"], alpha=0.5)
plt.title("Distribuzione della Tenure: Clienti Fedeli vs Churned")
plt.xlabel("Mesi di Tenure")
plt.ylabel("Densità")
plt.legend()
plt.savefig(OUT / "Tenure_KDE_Distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Monthly Charges vs Churn: chi paga di più tende a churnare di più
plt.figure(figsize=(10, 5))
sns.kdeplot(data=df[df["Churn Value"] == "0"], x="Monthly Charges", fill=True, label="Stayed", color=PALETTE["0"], alpha=0.5)
sns.kdeplot(data=df[df["Churn Value"] == "1"], x="Monthly Charges", fill=True, label="Churned", color=PALETTE["1"], alpha=0.5)
plt.title("Impatto dei Costi Mensili sul Churn")
plt.xlabel("Monthly Charges ($)")
plt.ylabel("Densità")
plt.legend()
plt.savefig(OUT / "MonthlyCharges_KDE_Distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Contract vs Churn
# Insight: Month-to-month = churn altissimo
plt.figure(figsize=(10, 6))
sns.countplot(x="Contract", hue="Churn Value", data=df, palette=PALETTE)
plt.title("Contract vs Churn")
plt.savefig(OUT / "ContractvsChurn.png", dpi=300)
plt.close()

# Calcolo churn rate per contratto
# Trasformiamo temporaneamente in numerico per la media
churn_rate_contract = df.copy()
churn_rate_contract["Churn Value"] = churn_rate_contract["Churn Value"].astype(int)
churn_rate_contract = churn_rate_contract.groupby("Contract")["Churn Value"].mean().reset_index()

plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Contract", y="Churn Value", data=churn_rate_contract, color="#264653")

# Aggiunta etichette percentuali sopra le barre (Professional touch)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.savefig(OUT / "Churn Rate by Contract Type.png", dpi=300)
plt.close()

# Internet Service vs Churn
# Insight: spesso Fiber optic ha churn più alto
plt.figure(figsize=(10, 6))
sns.countplot(x="Internet Service", hue="Churn Value", data=df, palette=PALETTE)
plt.title("Internet Service vs Churn")
plt.savefig(OUT / "InternetServicevsChurn.png", dpi=300)
plt.close()

# Payment Method vs Churn
# Insight: Electronic check → churn maggiore
plt.figure(figsize=(12, 6))
sns.countplot(x="Payment Method", hue="Churn Value", data=df, palette=PALETTE)
plt.xticks(rotation=15) # Rotazione leggera per pulizia
plt.title("Payment Method vs Churn")
plt.savefig(OUT / "PaymentMethodvsChurn.png", dpi=300, bbox_inches='tight')
plt.close()

# Churn Reason
# competitor, prezzi, costi extra
churned = df[df["Churn Value"] == "1"]
plt.figure(figsize=(10, 8))
sns.countplot(
    y="Churn Reason",
    data=churned,
    order=churned["Churn Reason"].value_counts().index,
    palette="flare"
)
plt.title("Principali Cause di Churn")
plt.xlabel("Conteggio")
plt.savefig(OUT / "ChurnReason.png", dpi=300, bbox_inches='tight')
plt.close()

#### Mappa Distribuzione
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x="Longitude",
    y="Latitude",
    hue="Churn Value",
    data=df,
    alpha=0.4,
    palette=PALETTE,
    s=15
)
plt.title("Customer Location and Churn")
plt.legend(title="Churned?", loc='upper right')
plt.savefig(OUT / "map_churn_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# Creazione gruppi Tenure
df["Tenure Group"] = pd.cut(
    df["Tenure Months"].astype(float),
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "12-24", "24-48", "48-72"]
)

# Calcolo churn rate: 
# churn molto alto nei primi 12 mesi, poi cala molto
tenure_churn = df.copy()
tenure_churn["Churn Value"] = tenure_churn["Churn Value"].astype(int)
tenure_churn = tenure_churn.groupby("Tenure Group", observed=True)["Churn Value"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(x="Tenure Group", y="Churn Value", data=tenure_churn, color="#457b9d")
plt.title("Churn Rate by Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.savefig(OUT / "churn_rate_grouped_by_tenuere_group.png", dpi=300)
plt.close()


# SCATTER PLOT AVANZATO: Tenure vs Monthly Charges
# Obiettivo: Vedere se i nuovi clienti (bassa tenure) che pagano molto sono quelli che scappano di più
plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df, 
    x="Tenure Months", 
    y="Monthly Charges", 
    hue="Churn Value", 
    palette=PALETTE, 
    alpha=0.4, 
    s=30,
    edgecolor=None
)

# Aggiungiamo linee di riferimento per i quadranti critici (es. primi 12 mesi e media costi)
plt.axvline(x=12, color='grey', linestyle='--', alpha=0.5)
plt.axhline(y=df["Monthly Charges"].mean(), color='grey', linestyle='--', alpha=0.5)

plt.title("Analisi Congiunta: Tenure vs Monthly Charges (Colorata per Churn)")
plt.xlabel("Tenure Months (Anzianità)")
plt.ylabel("Monthly Charges ($)")
plt.legend(title="Churn Status", labels=["Stayed (0)", "Churned (1)"], loc='upper right')

# Nota: I puntini arancioni concentrati in alto a sinistra indicano i clienti ad alto rischio
plt.savefig(OUT / "Scatter_Tenure_vs_Charges.png", dpi=300, bbox_inches='tight')
plt.close()


# ---- 6.4 Numero totale di servizi ----
# Definizione dei servizi per il calcolo
services = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies"
]

# Calcolo del numero di servizi (assicurandoci che i dati siano stringhe "Yes")
df["NumServices"] = (df[services] == "Yes").sum(axis=1)

# Calcolo del Churn Rate per ogni numero di servizi
# Trasformiamo Churn Value in intero per calcolare la media (rate)
service_churn = df.copy()
service_churn["Churn Value"] = service_churn["Churn Value"].astype(int)
service_analysis = service_churn.groupby("NumServices")["Churn Value"].mean().reset_index()

# Visualizzazione
plt.figure(figsize=(12, 6))

# Grafico a barre con gradazione di colore (più servizi = colore più scuro/stabile)
ax = sns.barplot(
    x="NumServices", 
    y="Churn Value", 
    data=service_analysis, 
    palette="viridis_r"
)

# Aggiunta di etichette percentuali sopra ogni barra
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1%}', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points', 
                weight='bold',
                color='#333333')

# Insight: Più servizi usa un cliente, meno tende a churnare
plt.title("Relazione tra Numero di Servizi Attivi e Churn Rate", pad=20)
plt.xlabel("Numero di Servizi Sottoscritti (0-8)")
plt.ylabel("Probabilità di Abbandono (Churn Rate)")
plt.ylim(0, service_analysis["Churn Value"].max() + 0.1) # Spazio per le etichette

plt.savefig(OUT / "Churn_Rate_by_NumServices.png", dpi=300, bbox_inches='tight')
plt.close()


# ---- Analisi Quantitativa: Numero di Servizi vs Churn ----
# Calcolo del numero di servizi (Phone, Security, Backup, ecc.)
services = [
    "Phone Service", "Multiple Lines", "Online Security", 
    "Online Backup", "Device Protection", "Tech Support", 
    "Streaming TV", "Streaming Movies"
]

# Creiamo la colonna NumServices se non esiste già
df["NumServices"] = (df[services] == "Yes").sum(axis=1)

plt.figure(figsize=(12, 7))

# Grafico a barre raggruppate per vedere i volumi assoluti
ax = sns.countplot(
    x="NumServices", 
    hue="Churn Value", 
    data=df, 
    palette=PALETTE
)

# Miglioramento estetico e legibilità
plt.title("Volume di Utenti per Numero di Servizi e Status Churn", pad=20)
plt.xlabel("Numero di Servizi Attivi (0-8)")
plt.ylabel("Numero di Clienti")
plt.legend(title="Churn Status", labels=["Stayed (0)", "Churned (1)"])

# Aggiungiamo i valori sopra le barre per precisione analitica
for p in ax.patches:
    if p.get_height() > 0: # Evitiamo etichette su barre vuote
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 7), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold')

plt.savefig(OUT / "Count_Churn_by_NumServices2.png", dpi=300, bbox_inches='tight')
plt.close()




import numpy as np

# ---- Preprocessing per Correlazione Totale ----
# Selezioniamo le colonne rilevanti (escludiamo ID o colonne con troppi valori unici come 'Churn Reason')
cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Churn Reason']
df_all = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Trasformiamo tutte le variabili categoriche in dummy variables (0 e 1)
# Questo espanderà il numero di colonne per includere ogni tipo di contratto, servizio, ecc.
df_encoded = pd.get_dummies(df_all, drop_first=True)

# Calcolo della matrice di Pearson su TUTTE le feature
corr_matrix_all = df_encoded.corr(method='pearson')

# Filtriamo per vedere solo la correlazione rispetto al Churn (ordinata)
# Questo è il grafico più utile per un analyst
churn_corr = corr_matrix_all['Churn Value_1'].sort_values(ascending=False).to_frame()

import numpy as np

# --- Matrice di Correlazione Normale (Tutte le feature) ---
plt.figure(figsize=(20, 16))

# Calcolo della matrice su tutte le feature numeriche e dummy
# (Assicurati di aver eseguito df_encoded = pd.get_dummies(...) prima)
corr_matrix_full = df_encoded.corr(method='pearson')

# Maschera per il triangolo superiore (opzionale, ma rende tutto più pulito)
mask = np.triu(np.ones_like(corr_matrix_full, dtype=bool))

# Generazione della Heatmap
sns.heatmap(
    corr_matrix_full, 
    mask=mask,
    annot=True,           # Inseriamo i numeri
    fmt=".2f",            # Due decimali
    cmap='coolwarm',      # Rosso (positivo), Blu (negativo)
    center=0,             # Lo zero è il bianco (neutro)
    annot_kws={"size": 8}, # Font piccolo per far stare i numeri
    linewidths=.5,
    cbar_kws={"shrink": .8}
)

plt.title("Matrice di Correlazione di Pearson (Dataset Completo)", fontsize=20, pad=20)

# Salvataggio con alta risoluzione per poter zoomare sui numeri
plt.savefig(OUT / "full_correlation_matrix_pearson.png", dpi=300, bbox_inches='tight')
plt.close()


'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco_customer_churn.csv"

OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

sns.countplot(x="Churn Value", data=df)
plt.savefig(OUT / "churn_distribution.png")
plt.close()



####Boxplot: variabili numeriche vs churn
#Tenure vs Churn  chi churn ha tenure molto più bassa
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn Value", y="Tenure Months", data=df)
plt.title("Tenure Months vs Churn")
plt.savefig(OUT / "TenureMonthsvsChurn.png")
plt.close()
#Monthly Charges vs Churn  chi paga di più tende a churnare di più
plt.figure(figsize=(6,4))
sns.boxplot(x="Churn Value", y="Monthly Charges", data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig(OUT / "MonthlyChargesvsChurn.png")
plt.close()


# Contract vs Churn
#Insight: Month-to-month = churn altissimo
sns.countplot(x="Contract", hue="Churn Value", data=df)
plt.savefig(OUT / "ContractvsChurn.png")
plt.close()

churn_rate_contract = df.groupby("Contract")["Churn Value"].mean().reset_index()

plt.figure(figsize=(6,4))

sns.barplot(x="Contract", y="Churn Value", data=churn_rate_contract)

plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.savefig(OUT / "Churn Rate by Contract Type.png")
plt.close()

#Internet Service vs Churn
#Insight:spesso Fiber optic ha churn più alto
sns.countplot(x="Internet Service", hue="Churn Value", data=df)
plt.savefig(OUT / "InternetServicevsChurn.png")
plt.close()

#Payment Method vs Churn
#Insight: Electronic check → churn maggiore
sns.countplot(x="Payment Method", hue="Churn Value", data=df)
plt.xticks(rotation=45)
plt.savefig(OUT / "PaymentMethodvsChurn.png")
plt.close()

# Churn Reason
#competitor, prezzi, costi extra
churned = df[df["Churn Value"] == 1]

sns.countplot(
    y="Churn Reason",
    data=churned,
    order=churned["Churn Reason"].value_counts().index
)
plt.savefig(OUT / "ChurnReason.png")
plt.close()


####
plt.figure(figsize=(8,6))

sns.scatterplot(
    x="Longitude",
    y="Latitude",
    hue="Churn Value",
    data=df,
    alpha=0.6
)

plt.title("Customer Location and Churn")
plt.savefig(OUT / "map_churn_distribution.png")
plt.close()


df["Tenure Group"] = pd.cut(
    df["Tenure Months"],
    bins=[0,12,24,48,72],
    labels=["0-12","12-24","24-48","48-72"]
)

#Calcolo churn rate: 
# churn molto alto nei primi 12 mesi, poi cala molto
tenure_churn = df.groupby("Tenure Group")["Churn Value"].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(x="Tenure Group", y="Churn Value", data=tenure_churn)
plt.title("Churn Rate by Tenure Group")
plt.ylabel("Churn Rate")
plt.savefig(OUT / "churn_rate_grouped_by_tenuere_group.png")
plt.close()
'''