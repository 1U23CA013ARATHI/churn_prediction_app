import streamlit as st
import pickle
import pandas as pd

# Load new model
model = pickle.load(open('churn_model.pkl', 'rb'))

st.title("🚀 Advanced Churn Analytics")

# Sidebar for inputs
st.sidebar.header("Customer Information")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0, 200, 50)
total = st.sidebar.number_input("Total Charges ($)", 0, 8000, 500)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert contract to number for model
contract_val = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}[contract]

if st.button("Predict Churn"):
    prediction = model.predict([[tenure, monthly, total, contract_val]])
    if prediction[0] == 1:
        st.error("Indha customer poga vaaipu adhigam!")
    else:
        st.success("Indha customer safe-ah irupparu.")

st.divider()

# --- Visualizations ---
st.subheader("📊 Customer Trends")
# Dummy data for chart (Ungalukku puriya)
chart_data = pd.DataFrame({'Charges': [monthly, monthly*0.8, monthly*1.2], 
                           'Category': ['Current', 'Min', 'Max']})
st.bar_chart(chart_data.set_index('Category'))