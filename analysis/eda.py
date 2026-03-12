import pandas as pd




def leggi_dati():
    df = pd.read_csv("data/raw/Telco_customer_churn.csv")

    print(df.head())
    print(df.info())
    print(df.describe())

    print(df.isnull().sum())
    