# frontend/app.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO

API_URL = "http://127.0.0.1:8000"

st.title("Churn Prediction — Demo")

st.markdown("Singola predizione")
with st.form("single"):
    customerID = st.text_input("customerID (opzionale)")
    gender = st.selectbox("gender", ["Male","Female",""] )
    SeniorCitizen = st.selectbox("SeniorCitizen", ["0","1",""])
    tenure = st.number_input("tenure (months)", min_value=0, max_value=100, value=12)
    MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=50.0)
    TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=600.0)
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year",""])
    submitted = st.form_submit_button("Predict single")
    if submitted:
        payload = {
            "customerID": customerID,
            "gender": gender if gender!="" else None,
            "SeniorCitizen": SeniorCitizen if SeniorCitizen!="" else None,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "Contract": Contract if Contract!="" else None
        }
        r = requests.post(API_URL + "/predict", json=payload)
        st.json(r.json())

st.markdown("---")
st.markdown("Batch prediction — upload CSV (raw Telco columns)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Anteprima dati:", df.head())
    if st.button("Predict batch"):
        payload = {"data": df.to_dict(orient="records")}
        r = requests.post(API_URL + "/predict_batch", json=payload, timeout=120)
        st.write(r.json())
        # show results head from API's results file
        res = requests.get(API_URL + "/results").json()
        st.write("Risultati salvati (sample):", res.get("sample", []))

st.markdown("---")
st.markdown("Scarica predictions registrate")
if st.button("Download saved predictions"):
    # simply show path; user can open from filesystem. For production implement file download endpoint.
    st.write("File salvato in: results/predictions.csv")