import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# --- CONFIGURAZIONE GLOBALE ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'legend.frameon': False
})

PALETTE = {"0": "#2a9d8f", "1": "#e76f51"}

# --- FUNZIONI DI VISUALIZZAZIONE ---

def plot_churn_distribution(df, output_path):
    """
    Mostra la distribuzione percentuale del Churn.
    Insight: Permette di capire se il dataset è sbilanciato e qual è il tasso di abbandono base.
    """
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="Churn Value", hue="Churn Value", data=df, palette=PALETTE, legend=False)
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=12, weight='bold', xytext=(0, 5),
                    textcoords='offset points')
    plt.title("Distribuzione Generale del Churn (%)")
    plt.savefig(output_path / "churn_distribution_percent.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_tenure_kde(df, output_path):
    """
    Analisi densità Tenure vs Churn.
    Insight: Chi abbandona tende ad avere una tenure molto bassa (picco nei primi mesi).
    """
    plt.figure(figsize=(10, 5))
    for val, label in [("0", "Stayed"), ("1", "Churned")]:
        sns.kdeplot(data=df[df["Churn Value"] == val], x="TenureMonths", 
                    fill=True, label=label, color=PALETTE[val], alpha=0.5)
    plt.title("Distribuzione della Tenure: Clienti Fedeli vs Churned")
    plt.legend()
    plt.savefig(output_path / "Tenure_KDE_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_monthly_charges_kde(df, output_path):
    """
    Analisi densità Costi Mensili vs Churn.
    Insight: I clienti con costi mensili più elevati mostrano una densità di churn maggiore.
    """
    plt.figure(figsize=(10, 5))
    for val, label in [("0", "Stayed"), ("1", "Churned")]:
        sns.kdeplot(data=df[df["Churn Value"] == val], x="MonthlyCharges", 
                    fill=True, label=label, color=PALETTE[val], alpha=0.5)
    plt.title("Impatto dei Costi Mensili sul Churn")
    plt.legend()
    plt.savefig(output_path / "MonthlyCharges_KDE_Distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_contract_churn(df, output_path):
    """
    Confronto contratti e Churn Rate.
    Insight: Il contratto 'Month-to-month' è il principale predittore di churn.
    """
    # Countplot
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Contract", hue="Churn Value", data=df, palette=PALETTE)
    plt.title("Contract vs Churn")
    plt.savefig(output_path / "ContractvsChurn.png", dpi=300)
    plt.close()

    # Barplot Churn Rate
    temp_df = df.copy()
    temp_df["Churn Value"] = temp_df["Churn Value"].astype(int)
    churn_rate = temp_df.groupby("Contract")["Churn Value"].mean().reset_index()
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x="Contract", y="Churn Value", data=churn_rate, color="#264653")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')
    plt.title("Churn Rate by Contract Type")
    plt.savefig(output_path / "Churn_Rate_by_Contract_Type.png", dpi=300)
    plt.close()

def plot_services_and_payment(df, output_path):
    """
    Analisi Internet Service e Metodo di Pagamento.
    Insight: Fibra Ottica ed Electronic Check sono associati a churn più elevati.
    """
    # Internet Service
    plt.figure(figsize=(10, 6))
    sns.countplot(x="Internet Service", hue="Churn Value", data=df, palette=PALETTE)
    plt.title("Internet Service vs Churn")
    plt.savefig(output_path / "InternetServicevsChurn.png", dpi=300)
    plt.close()

    # Payment Method
    plt.figure(figsize=(12, 6))
    sns.countplot(x="Payment Method", hue="Churn Value", data=df, palette=PALETTE)
    plt.xticks(rotation=15)
    plt.title("Payment Method vs Churn")
    plt.savefig(output_path / "PaymentMethodvsChurn.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_map_distribution(df, output_path):
    """
    Mappa geografica del Churn.
    Insight: Identifica se ci sono cluster geografici di abbandono (hotspots).
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="Longitude", y="Latitude", hue="Churn Value", data=df, 
                    alpha=0.4, palette=PALETTE, s=15)
    plt.title("Customer Location and Churn")
    plt.savefig(output_path / "map_churn_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_tenure_group_rate(df, output_path):
    """
    Churn Rate raggruppato per fasce di anzianità.
    Insight: Il churn cala drasticamente dopo il primo anno (0-12 mesi).
    """
    df["Tenure Group"] = pd.cut(df["TenureMonths"].astype(float), bins=[0, 12, 24, 48, 72],
                                 labels=["0-12", "12-24", "24-48", "48-72"])
    temp_df = df.copy()
    temp_df["Churn Value"] = temp_df["Churn Value"].astype(int)
    tenure_churn = temp_df.groupby("Tenure Group", observed=True)["Churn Value"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Tenure Group", y="Churn Value", data=tenure_churn, color="#457b9d")
    plt.title("Churn Rate by Tenure Group")
    plt.savefig(output_path / "churn_rate_grouped_by_tenure_group.png", dpi=300)
    plt.close()

def plot_scatter_tenure_charges(df, output_path):
    """
    Scatter plot Tenure vs Monthly Charges.
    Insight: Identifica la 'Red Zone' (clienti nuovi ad alto costo).
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="TenureMonths", y="MonthlyCharges", hue="Churn Value", 
                    palette=PALETTE, alpha=0.4, s=30, edgecolor=None)
    plt.axvline(x=12, color='grey', linestyle='--', alpha=0.5)
    plt.axhline(y=df["MonthlyCharges"].mean(), color='grey', linestyle='--', alpha=0.5)
    plt.title("Analisi Congiunta: Tenure vs Monthly Charges")
    plt.savefig(output_path / "Scatter_Tenure_vs_Charges.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_num_services_count(df, output_path):
    """
    Volume di Churn in base al numero di servizi attivi.
    Insight: Più servizi = Maggiore 'stickiness' (meno churn).
    """
    services = ["Phone Service", "Multiple Lines", "Online Security", "Online Backup", 
                "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
    df["NumServices"] = (df[services] == "Yes").sum(axis=1)

    plt.figure(figsize=(12, 7))
    ax = sns.countplot(x="NumServices", hue="Churn Value", data=df, palette=PALETTE)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points', 
                        fontsize=10, fontweight='bold')
    plt.title("Volume di Utenti per Numero di Servizi e Status Churn")
    plt.savefig(output_path / "Count_Churn_by_NumServices.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_full_correlation_matrix(df, output_path):
    """
    Matrice di correlazione di Pearson su tutte le feature.
    Insight: Identifica quali variabili dummy (es. Contract_Two year) hanno l'impatto più forte.
    """
    cols_to_drop = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Churn Reason']
    df_all = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df_encoded = pd.get_dummies(df_all, drop_first=True)
    corr_matrix = df_encoded.corr(method='pearson')

    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, linewidths=.5)
    plt.title("Matrice di Correlazione di Pearson (Dataset Completo)")
    plt.savefig(output_path / "full_correlation_matrix_pearson.png", dpi=300, bbox_inches='tight')
    plt.close()

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Setup percorsi
    ROOT = Path(__file__).resolve().parents[1]
    PROC_DIR = ROOT / "data" / "processed"
    OUT_DIR = ROOT / "outputs" / "plots"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Caricamento dati
    df_train = pd.read_csv(PROC_DIR / "train_raw.csv")
    df_test = pd.read_csv(PROC_DIR / "test_raw.csv")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    # Pre-processing minimo per analisi
    df["Churn Value"] = df["Churn Value"].astype(str)

    # Chiamata a tutte le funzioni
    plot_churn_distribution(df, OUT_DIR)
    plot_tenure_kde(df, OUT_DIR)
    plot_monthly_charges_kde(df, OUT_DIR)
    plot_contract_churn(df, OUT_DIR)
    plot_services_and_payment(df, OUT_DIR)
    plot_map_distribution(df, OUT_DIR)
    plot_tenure_group_rate(df, OUT_DIR)
    plot_scatter_tenure_charges(df, OUT_DIR)
    plot_num_services_count(df, OUT_DIR)
    plot_full_correlation_matrix(df, OUT_DIR)
    
    print(f"Analisi completata. Grafici salvati in: {OUT_DIR}")