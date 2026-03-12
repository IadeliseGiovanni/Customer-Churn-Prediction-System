import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Customer Churn Predictor")

tenure = st.slider("Tenure Months", 0, 72)
monthly = st.number_input("Monthly Charges")

if st.button("Predict"):
    payload = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly
    }

    r = requests.post(API_URL+"/predict", json=payload)

    st.write(r.json())
