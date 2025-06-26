import streamlit as st
import numpy as np
import joblib

model = joblib.load("pd_model_with_creditscore.pkl")
scaler = joblib.load("scaler_with_creditscore.pkl")

st.title("Enhanced PD Scoring with Credit Score")

age = st.slider("Age", 21, 65)
income = st.number_input("Monthly Income", min_value=500.0, value=3000.0)
profit = st.number_input("Monthly Profit", min_value=200.0, value=2000.0)
loan = st.number_input("Loan Amount", min_value=1000.0, value=20000.0)
tenure = st.slider("Tenure (months)", 6, 60)
interest = st.slider("Interest Rate (%)", 5.0, 35.0)
exchange = st.slider("Exchange Rate (GHS/USD)", 10.0, 17.0)
inflation = st.slider("Inflation (%)", 6.0, 20.0)
willingness = st.slider("Willingness Score (0 to 1)", 0.0, 1.0, 0.5)

repayment = st.selectbox("Repayment Structure", ["Equal Installment", "Bullet", "Interest Only"])
loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])
sector = st.selectbox("Sector", ["Agriculture", "Manufacturing", "Salaried", "SME", "Trade"])
eco_phase = st.selectbox("Economic Phase", ["Growth", "Recession", "Recovery"])

installment = loan / tenure
dti = installment / income
itp = installment / profit

rep_dict = {"Bullet": [1, 0], "Interest Only": [0, 1], "Equal Installment": [0, 0]}
loan_dict = {"Unsecured": 1, "Secured": 0}
sector_dict = {"Manufacturing": [1, 0, 0, 0], "Salaried": [0, 1, 0, 0],
               "SME": [0, 0, 1, 0], "Trade": [0, 0, 0, 1], "Agriculture": [0, 0, 0, 0]}
eco_dict = {"Recession": [1, 0], "Recovery": [0, 1], "Growth": [0, 0]}

features = np.array([[
    age, income, profit, loan, tenure, installment, interest,
    dti, itp, willingness, exchange, inflation,
    loan_dict[loan_type],
    *rep_dict[repayment],
    *sector_dict[sector],
    *eco_dict[eco_phase]
]])
features_scaled = scaler.transform(features)
pred_prob = model.predict_proba(features_scaled)[0][1]

st.subheader("Results")
st.metric(label="Predicted Probability of Default", value=f"{pred_prob:.2%}")