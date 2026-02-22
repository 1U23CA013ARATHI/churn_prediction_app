import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="ChurnGuard AI", layout="wide", initial_sidebar_state="expanded")

# --- DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
    df = load_data()
except Exception as e:
    st.error(f"Error: {e}")

# --- NAVIGATION MENU ---
with st.sidebar:
    st.title("🌐 ChurnGuard AI")
    page = st.radio("Navigation", ["🏠 Home", "🔍 Predict Churn", "📊 Analytics Dashboard", "📌 About Project"])
    st.divider()
    st.info("Built with Machine Learning & Streamlit")

# --- 1. HOME PAGE (OVERVIEW) ---
if page == "🏠 Home":
    st.title("🎯 Customer Churn Prediction System")
    st.subheader("Keep your customers before they leave!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Why use this system?
        This system predicts whether a customer is likely to leave a service based on their 
        usage patterns and billing information.
        
        **Key Features:**
        - **Instant Prediction:** AI-driven churn analysis.
        - **Visual Dashboard:** Real-time data trends.
        - **Risk Assessment:** High/Low risk categorization.
        """)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3408/3408545.png", width=250)

# --- 2. PREDICTION PAGE ---
elif page == "🔍 Predict Churn":
    st.title("🔍 Customer Data Input")
    st.write("Enter customer details below to predict the churn risk.")
    
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Customer Tenure (Months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0)
        with c2:
            contract_type = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            total = st.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)
        
        predict_btn = st.form_submit_button("👉 Predict Churn")

    # --- 3. PREDICTION RESULT ---
    if predict_btn:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        input_data = np.array([[tenure, monthly, total, contract_map[contract_type]]])
        
        prediction = model.predict(input_data)
        prob = model.predict_proba(input_data)[0]
        
        st.divider()
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        if prediction[0] == 1:
            res_col1.error("❌ **Churn: YES**")
            res_col2.warning(f"⚠️ **Risk Level: High ({prob[1]:.2%})**")
            st.write("📢 **Message:** Customer is likely to leave the service. Immediate action recommended.")
        else:
            res_col1.success("✅ **Churn: NO**")
            res_col2.info(f"🟢 **Risk Level: Low ({prob[0]:.2%})**")
            st.write("📢 **Message:** Customer is likely to stay. Continue current engagement strategies.")

# --- 4. DASHBOARD PAGE (PROFESSIONAL LOOK) ---
elif page == "📊 Analytics Dashboard":
    st.title("📊 Customer Insights Dashboard")
    
    # Metrics
    total_cust = len(df)
    churned_cust = len(df[df['Churn'] == 'Yes'])
    churn_rate = (churned_cust / total_cust) * 100
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", f"{total_cust:,}")
    m2.metric("Churned Customers", f"{churned_cust:,}")
    m3.metric("Churn Percentage", f"{churn_rate:.2f}%")
    
    st.divider()
    
    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig1 = px.pie(df, names='Churn', title="Churn vs Non-Churn Distribution", hole=0.5, color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig1, use_container_width=True)
        
    with chart_col2:
        fig2 = px.bar(df, x='Contract', color='Churn', title="Churn by Contract Type", barmode='group')
        st.plotly_chart(fig2, use_container_width=True)

# --- ABOUT PAGE ---
elif page == "📌 About Project":
    st.title("📌 About the Project")
    st.markdown("""
    **Developer:** 1U23CA013ARATHI
    **Algorithm:** Random Forest Classifier
    **Dataset:** Telco Customer Churn (IBM Dataset)
    **Objective:** This project leverages Machine Learning to help telecommunication companies reduce customer attrition by identifying at-risk users early.
    """)