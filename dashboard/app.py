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
# Modern Healthcare Theme
# =====================================================

st.markdown(

"""
<style>


/* Main background */

.stApp {

    background-color:#eaf6ea;

}


/* All text */

.stApp * {

    color:#000000;

}


/* ==========================================
   Number Input Values
   Individual Patient Assessment
   ========================================== */


/* Input box text colour */

div[data-baseweb="input"] input {

    color: #FFD700 !important;

    font-weight:700;

    font-size:18px;

}



/* Input box background */

div[data-baseweb="input"] {

    background-color:white;

    border-radius:10px;

}



/* Input labels */

div[data-testid="stNumberInput"] label {

    color:black !important;

    font-weight:600;

}



</style>

""",

unsafe_allow_html=True

)



# =====================================================
# Paths
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
# Load Data
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
# Header
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




# =====================================================
# Patient Prediction
# =====================================================


if page == "Patient Risk Prediction":



    st.subheader(

        "Individual Patient Assessment"

    )



    col1, col2 = st.columns(2)



    with col1:


        pregnancies = st.number_input(

            "Pregnancies",

            min_value=0,

            value=2

        )


        glucose = st.number_input(

            "Glucose level (mg/dL)",

            min_value=0,

            value=120

        )


        blood_pressure = st.number_input(

            "Blood Pressure",

            min_value=0,

            value=80

        )


        skin = st.number_input(

            "Skin Thickness",

            min_value=0,

            value=20

        )



    with col2:


        insulin = st.number_input(

            "Insulin",

            min_value=0,

            value=100

        )


        bmi = st.number_input(

            "BMI",

            min_value=0.0,

            value=25.0

        )


        pedigree = st.number_input(

            "Diabetes Pedigree Function",

            min_value=0.0,

            value=0.5

        )


        age = st.number_input(

            "Age",

            min_value=1,

            value=45

        )



    risk_score = (

        int(glucose > 125)

        +

        int(bmi > 30)

        +

        int(age > 50)

        +

        int(blood_pressure > 80)

    )



    patient = pd.DataFrame({

        "Pregnancies":[pregnancies],

        "Glucose":[glucose],

        "BloodPressure":[blood_pressure],

        "SkinThickness":[skin],

        "Insulin":[insulin],

        "BMI":[bmi],

        "DiabetesPedigreeFunction":[pedigree],

        "Age":[age],

        "Risk_Score":[risk_score]

    })



    if st.button(

        "Generate Diabetes Risk Prediction"

    ):



        patient_scaled = scaler.transform(

            patient

        )


        probability = model.predict_proba(

            patient_scaled

        )[0][1]


        prediction = model.predict(

            patient_scaled

        )[0]



        st.markdown("---")



        col1, col2 = st.columns(2)



        with col1:


            st.metric(

                "Risk Probability",

                f"{probability:.1%}"

            )



        with col2:


            st.metric(

                "Clinical Risk Score",

                risk_score

            )



        st.markdown("---")



        if prediction == 1:



            st.markdown(

            """

            <div class="card"
            style="
            background:#ffecec;
            border-left:6px solid #d9534f;
            ">


            <h3>

            HIGH DIABETES RISK

            </h3>


            <p>

            Recommended clinical considerations:

            <br><br>

            ✓ HbA1c assessment

            <br>

            ✓ Lifestyle evaluation

            <br>

            ✓ Clinical follow-up

            <br>

            ✓ Cardiometabolic risk assessment

            </p>


            </div>

            """,

            unsafe_allow_html=True

            )



        else:


            st.markdown(

            """

            <div class="card"
            style="
            background:#eaf7ea;
            border-left:6px solid #28a745;
            ">


            <h3>

            LOW DIABETES RISK

            </h3>


            <p>

            Current measurements indicate
            lower predicted diabetes risk.

            </p>


            </div>


            """,

            unsafe_allow_html=True

            )




# =====================================================
# Population Analytics
# =====================================================


elif page == "Population Analytics":



    st.subheader(

        "Population Diabetes Overview"

    )


    outcome = (

        df["Outcome"]

        .value_counts()

        .reset_index()

    )


    outcome.columns=[

        "Outcome",

        "Patients"

    ]


    fig = px.bar(

        outcome,

        x="Outcome",

        y="Patients",

        title="Diabetes Outcome Distribution"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )



    fig2 = px.histogram(

        df,

        x="BMI",

        nbins=30,

        title="BMI Distribution"

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )





# =====================================================
# Clinical Risk Factors
# =====================================================


else:



    st.subheader(

        "Clinical Risk Factors"

    )



    correlation = (

        df.corr(numeric_only=True)

        ["Outcome"]

        .drop("Outcome")

        .sort_values()

    )



    fig = px.bar(

        correlation,

        title="Clinical Variable Association"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )




# =====================================================
# Footer
# =====================================================


st.markdown(

"""

<div class="footer">

Developed as a clinical machine learning portfolio project

<br>

Python | Machine Learning | Explainable AI | Streamlit

</div>

""",

unsafe_allow_html=True

)