import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Churn Predictor", layout="centered")

# Load the saved model
try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Please run train_model.py first.")

st.title("🚀 Advanced Churn Analytics")
st.markdown("Predict if a customer will leave or stay based on their billing and contract info.")

# Sidebar Inputs
st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)
contract_type = st.sidebar.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])

# Encoding input for model
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
contract_val = contract_mapping[contract_type]

# Prediction Logic
if st.button("Predict Churn"):
    # Creating input array in the SAME order as training
    input_data = np.array([[tenure, monthly, total, contract_val]])
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.divider()
    if prediction[0] == 1:
        st.error(f"⚠️ **Result: Likely to Churn** (Confidence: {prediction_proba[0][1]:.2%})")
    else:
        st.success(f"✅ **Result: Likely to Stay** (Confidence: {prediction_proba[0][0]:.2%})")

# Visual context (Optional)
st.info("Note: Prediction is based on tenure, billing, and contract type.")