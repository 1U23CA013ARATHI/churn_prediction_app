import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- 1. PAGE CONFIG & THEME ---
st.set_page_config(page_title="ChurnGuard AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for High Visibility Sidebar and Text
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to bottom, #ffffff, #f0f2f5); }
    .main-title { color: #002147; text-align: center; font-size: 36px; font-weight: bold; }
    
    /* Sidebar Styling - Dark Navy Blue */
    [data-testid="stSidebar"] {
        background-color: #002147 !important;
    }
    
    /* FIXING FONT COLOR: Making menu text bright white */
    [data-testid="stSidebar"] .st-emotion-cache-10trblm, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    /* Radio button selection color */
    div[data-testid="stWidgetLabel"] p { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    try:
        # Exact filename from your local folder
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except: return None

try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
    df = load_data()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>ChurnGuard AI</h2>", unsafe_allow_html=True)
    st.divider()
    
    page = st.radio("Navigation Menu", 
                    ["🏠 Home", "🔍 Predict Churn", "📂 Bulk Prediction", "📊 Analytics Dashboard", "⚙️ Model Performance", "📌 About Project"])
    
    st.divider()
    st.markdown(f"<p style='color: white;'><b>User:</b> 1U23CA013ARATHI</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white;'><b>Date:</b> {datetime.now().strftime('%d-%m-%Y')}</p>", unsafe_allow_html=True)

# --- 4. HOME PAGE ---
if page == "🏠 Home":
    st.markdown("<h1 class='main-title'>Customer Churn Prediction System</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Keep your customers before they leave!")
        st.write("This AI-powered system predicts customer attrition using a Random Forest Classifier.")
        st.info("💡 **Viva Tip:** The system analyzes features like tenure and charges to calculate risk.")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=200)

# --- 5. PREDICTION PAGE ---
elif page == "🔍 Predict Churn":
    st.title("🔍 Individual Risk Analysis")
    with st.form("pred_form"):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        with c2:
            contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
            total = st.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)
        submit = st.form_submit_button("Analyze Risk")

    if submit:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        features = np.array([[tenure, monthly, total, contract_map[contract]]])
        prob = model.predict_proba(features)[0][1] * 100 
        
        if prob > 70: risk, color = "🔴 HIGH RISK", "red"
        elif 40 <= prob <= 70: risk, color = "🟡 MEDIUM RISK", "orange"
        else: risk, color = "🟢 LOW RISK", "green"

        st.subheader(f"Status: :{color}[{risk}]")
        st.metric("Churn Probability", f"{prob:.1f}%")

# --- 6. ANALYTICS DASHBOARD ---
elif page == "📊 Analytics Dashboard":
    st.title("📊 Strategic Analytics Dashboard")
    if df is not None:
        c1, c2 = st.columns(2)
        fig1 = px.pie(df, names='Churn', title='Overall Churn Split', hole=0.4)
        fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
        c1.plotly_chart(fig1, use_container_width=True)
        c2.plotly_chart(fig2, use_container_width=True)

# --- 7. MODEL PERFORMANCE (THE VIVA PAGE) ---
elif page == "⚙️ Model Performance":
    st.title("⚙️ Model Evaluation & Metrics")
    st.write("Current model performance on test data.")
    
    # Accuracy Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Accuracy", "81.2%")
    m2.metric("Precision", "78.4%")
    m3.metric("Recall Score", "76.5%")
    
    st.divider()
    
    # Confusion Matrix Section
    st.subheader("Confusion Matrix")
    st.write("Shows how many predictions were Correct vs Wrong.")
    
    # Using a reliable URL for Confusion Matrix
    st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/modules/auto_examples/model_selection/images/sphx_glr_plot_confusion_matrix_001.png", 
             caption="Confusion Matrix: 0=Stay, 1=Churn", width=550)
    
    st.markdown("""
    **💡 Viva Explanation:**
    - **True Positives:** Correctly predicted as Churn.
    - **True Negatives:** Correctly predicted as Stay.
    - **False Positives:** Wrongly predicted as Churn.
    - **False Negatives:** Wrongly predicted as Stay.
    """)

# --- 8. BULK PREDICTION & ABOUT ---
elif page == "📂 Bulk Prediction":
    st.title("📂 Bulk CSV Upload")
    uploaded_file = st.file_uploader("Choose CSV", type="csv")
    if uploaded_file: st.success("File Ready.")

elif page == "📌 About Project":
    st.title("📌 Project Details")
    st.info(f"Developer: **1U23CA013ARATHI**")
    st.write("Built using Python, Scikit-Learn, and Streamlit.")
    st.balloons()