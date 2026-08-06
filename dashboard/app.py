"""
Diabetes Risk Prediction Dashboard

Mini Project 5

Features:
1. Individual patient risk prediction
2. Population diabetes analysis
3. Model explanation

Author:
Dr Suhirthakumar Puvanendran
"""
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

import streamlit as st

import pandas as pd
import numpy as np

import joblib

import plotly.express as px





# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(

    page_title="Diabetes Risk Prediction Platform",

    page_icon="🩺",

    layout="wide"

)



# =====================================================
# Healthcare Dashboard Styling
# =====================================================

st.markdown(

"""
<style>


/* ===============================
Main background
================================*/

.stApp {

    background-color:#f5f9fc;

}



/* ===============================
Global text colour
================================*/

.stApp * {

    color:#000000;

}



/* ===============================
Main headings
================================*/

h1 {

    color:#0b3954 !important;

    font-weight:700;

}


h2, h3 {

    color:#0b3954 !important;

}



/* ===============================
Sidebar
================================*/


section[data-testid="stSidebar"] {

    background-color:#e8f3f8;

}


section[data-testid="stSidebar"] * {

    color:#000000 !important;

}



/* ===============================
Cards
================================*/


.card {

    background:white;

    padding:25px;

    border-radius:15px;

    box-shadow:
    0px 4px 15px rgba(0,0,0,0.08);

    margin-bottom:20px;

}



/* ===============================
Metrics
================================*/


div[data-testid="metric-container"] {

    background:white;

    padding:20px;

    border-radius:15px;

    box-shadow:
    0px 3px 10px rgba(0,0,0,0.08);

}



div[data-testid="metric-container"] * {

    color:black !important;

}



/* ===============================
Input labels
================================*/


div[data-testid="stNumberInput"] label {

    color:#000000 !important;

    font-weight:600;

}



/* ===============================
Input number values
================================*/


div[data-testid="stNumberInput"] input {

    color:#FFD700 !important;

    font-weight:700;

    font-size:18px;

}



/* ===============================
Buttons
================================*/


.stButton button {

    background-color:#087e8b;

    color:white !important;

    border-radius:10px;

    height:45px;

    font-size:16px;

    font-weight:600;

}


.stButton button:hover {

    background-color:#0b3954;

}



/* ===============================
Footer
================================*/


.footer {

    text-align:center;

    color:#555555;

    margin-top:40px;

}



</style>

""",

unsafe_allow_html=True

)




# =====================================================
# File Paths
# =====================================================


MODEL_PATH = "models/random_forest.pkl"

SCALER_PATH = "models/scaler.pkl"

DATA_PATH = "data/processed/diabetes_features.csv"




# =====================================================
# Load Model
# =====================================================


@st.cache_resource

def load_model():


    model = joblib.load(

        MODEL_PATH

    )


    scaler = joblib.load(

        SCALER_PATH

    )


    return model, scaler




# =====================================================
# Load Dataset
# =====================================================


@st.cache_data

def load_data():


    return pd.read_csv(

        DATA_PATH

    )




try:


    model, scaler = load_model()

    df = load_data()



except Exception as e:


    st.error(

        f"Application loading error: {e}"

    )

    st.stop()




# =====================================================
# Sidebar
# =====================================================


st.sidebar.title(

    "🩺 Diabetes Analytics"

)



page = st.sidebar.radio(

    "Select Module",

    [

        "Patient Risk Prediction",

        "Population Analytics",

        "Clinical Risk Factors"

    ]

)



st.sidebar.markdown(

"""

---

### About

This dashboard uses machine learning
to estimate diabetes risk from clinical
measurements.


### Technology

- Python
- Scikit-learn
- Streamlit
- Machine Learning


---

"""

)




# =====================================================
# Dashboard Header
# =====================================================


st.markdown(

"""

<div class="card">


<h1>

🩺 Diabetes Risk Prediction Platform

</h1>


<p>

AI-powered clinical decision support dashboard
for diabetes risk assessment and population analytics.

</p>


</div>


""",

unsafe_allow_html=True

)