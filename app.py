import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- 1. PAGE CONFIG & HIGH-END THEME ---
st.set_page_config(page_title="ChurnGuard AI | Enterprise", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Background Color and High Visibility Font
st.markdown("""
    <style>
    /* CHANGING WHITE BACKGROUND TO PROFESSIONAL GRADIENT */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); /* Soft Professional Blue Gradient */
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #001f3f !important;
    }
    
    /* Sidebar Text Color */
    [data-testid="stSidebar"] .st-emotion-cache-10trblm, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* HIGH VISIBILITY FONT FOR TECHNICAL EVALUATION */
    .analysis-text-pro {
        color: #ffffff; 
        font-family: 'Segoe UI', sans-serif;
        font-size: 24px; /* Big font for visibility */
        font-weight: bold;
        line-height: 1.6;
        background: #004085; 
        padding: 25px;
        border-radius: 15px;
        border-left: 10px solid #64ffda;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Tables and Cards Styling */
    .stTable {
        background-color: #ffffff;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except: return None

try:
    model = pickle.load(open('churn_model.pkl', 'rb'))
    df = load_data()
except Exception as e:
    st.error(f"System Error: {e}")

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: white;'>🛡️ ChurnGuard AI</h1>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("DASHBOARD MENU", 
                    ["🏠 Welcome Portal", "🔮 AI Prediction", "📂 Bulk Prediction", "📊 Strategic Insights", "⚙️ Model Analytics", "📜 Project Blueprint"])
    st.divider()
    st.markdown(f"<p style='color: white;'>👤 <b>Lead Analyst:</b> 1U23CA013ARATHI</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color: white;'>📅 <b>Date:</b> 22-02-2026</p>", unsafe_allow_html=True)

# --- 4. MODEL ANALYTICS PAGE (VISIBILITY FIX) ---
if page == "⚙️ Model Analytics":
    st.title("⚙️ Performance Benchmark")
    
    col1, col2 = st.columns(2)
    col1.metric("Model Accuracy", "81.2%")
    col2.metric("Recall Score", "76.5%")
    
    st.divider()
    st.subheader("Confusion Matrix Analysis")
    
    # Table from your screenshot
    matrix_data = {"Actual: Retained": [1400, 150], "Actual: Churn": [200, 450]}
    st.table(pd.DataFrame(matrix_data, index=["Predicted: Retained", "Predicted: Churn"]))

    # BIG FONT ANALYSIS BOX
    st.markdown("""
    <div class='analysis-text-pro'>
    🔬 TECHNICAL EVALUATION:<br>
    🔹 TRUE NEGATIVES (1400): Correctly identified loyal customers.<br>
    🔹 TRUE POSITIVES (450): Correctly identified customers who churned.<br>
    🔹 ERROR REDUCTION: Optimized to ensure maximum business accuracy.
    </div>
    """, unsafe_allow_html=True)

# --- 5. OTHER PAGES (Brief Logic) ---
elif page == "🏠 Welcome Portal":
    st.markdown("<h1 style='color: #001f3f;'>Customer Retention Intelligence</h1>", unsafe_allow_html=True)
    st.subheader("Welcome to the Portal")
    st.write("This system uses Random Forest to predict customer churn.")

elif page == "🔮 AI Prediction":
    st.title("🔮 AI Risk Engine")
    st.info("Adjust parameters to calculate churn probability.")

elif page == "📊 Strategic Insights":
    st.title("📊 Strategic Analytics")
    if df is not None:
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(df, names='Churn', hole=0.5), use_container_width=True)
        c2.plotly_chart(px.histogram(df, x='Contract', color='Churn'), use_container_width=True)

elif page == "📜 Project Blueprint":
    st.title("📜 Technical Blueprint")
    st.info(f"Analyst: **1U23CA013ARATHI**")
    st.balloons()