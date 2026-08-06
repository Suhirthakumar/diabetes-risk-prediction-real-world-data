"""
Diabetes Risk Prediction Dashboard

Mini Project 5

Features:
1. Individual patient risk prediction
2. Population diabetes analysis
3. Model explanation

Author:
Suhirthakumar Puvanendran
"""


import streamlit as st

import pandas as pd
import numpy as np

import joblib

import plotly.express as px



# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(

    page_title="Diabetes Risk Prediction",

    page_icon="🩺",

    layout="wide"

)



# =====================================================
# Paths
# =====================================================

MODEL_PATH = "models/random_forest.pkl"

SCALER_PATH = "models/scaler.pkl"

DATA_PATH = "data/processed/diabetes_features.csv"



# =====================================================
# Load model
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



model, scaler = load_model()



# =====================================================
# Load dataset
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    return df



df = load_data()



# =====================================================
# Sidebar
# =====================================================

st.sidebar.title(
    "Diabetes Analytics"
)


page = st.sidebar.radio(

    "Select Analysis",

    [

        "Patient Risk Prediction",

        "Population Analysis",

        "Clinical Factors"

    ]

)



# =====================================================
# PAGE 1
# Patient prediction
# =====================================================

if page == "Patient Risk Prediction":


    st.title(
        "🩺 Individual Diabetes Risk Prediction"
    )


    st.write(

        """
        Enter patient clinical measurements
        to estimate diabetes risk.
        """

    )


    col1, col2 = st.columns(2)



    with col1:


        pregnancies = st.number_input(

            "Pregnancies",

            min_value=0,

            max_value=20,

            value=2

        )


        glucose = st.number_input(

            "Glucose",

            min_value=0,

            max_value=300,

            value=120

        )


        blood_pressure = st.number_input(

            "Blood Pressure",

            min_value=0,

            max_value=200,

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

            max_value=120,

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
        "Predict Diabetes Risk"
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



        st.subheader(
            "Prediction Result"
        )



        st.metric(

            "Risk Probability",

            f"{probability:.1%}"

        )



        if prediction == 1:


            st.error(

                "High Diabetes Risk"

            )


            st.write(

                """
                Suggested clinical actions:

                - HbA1c assessment
                - Lifestyle evaluation
                - Clinical follow-up

                """

            )


        else:


            st.success(

                "Low Diabetes Risk"

            )



# =====================================================
# PAGE 2
# Population analysis
# =====================================================


elif page == "Population Analysis":


    st.title(

        "Population Diabetes Analysis"

    )


    outcome_count = (

        df["Outcome"]

        .value_counts()

        .reset_index()

    )


    outcome_count.columns = [

        "Outcome",

        "Patients"

    ]



    fig = px.bar(

        outcome_count,

        x="Outcome",

        y="Patients",

        title="Diabetes Distribution"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )



    st.subheader(

        "BMI Distribution"

    )



    fig2 = px.histogram(

        df,

        x="BMI",

        nbins=30

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )



# =====================================================
# PAGE 3
# Clinical factors
# =====================================================


else:


    st.title(

        "Clinical Risk Factors"

    )


    correlation = (

        df.corr(numeric_only=True)

        ["Outcome"]

        .sort_values()

    )



    correlation = (

        correlation

        .drop("Outcome")

    )


    fig = px.bar(

        correlation,

        title="Feature Association With Diabetes Outcome"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )



    st.write(

        """

        Higher positive values indicate stronger
        association with diabetes outcome.

        """

    )