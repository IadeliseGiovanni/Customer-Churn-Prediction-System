# Customer Churn ML Project

Pipeline:
1. Preprocessing dataset
2. Training ML model
3. Evaluation
4. API backend
5. Streamlit dashboard

Run order:

python ml/preprocessing.py
python ml/train_model.py
python ml/evaluate.py

uvicorn backend.api:app --reload

streamlit run frontend/dashboard.py