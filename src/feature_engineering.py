"""
Feature Engineering Pipeline

Creates clinically relevant diabetes risk variables

"""


import pandas as pd
import os



INPUT_FILE = (
    "data/processed/diabetes_cleaned.csv"
)


OUTPUT_FILE = (
    "data/processed/diabetes_features.csv"
)



# -------------------------------------
# Load cleaned data
# -------------------------------------

df = pd.read_csv(
    INPUT_FILE
)



# -------------------------------------
# BMI categories
# -------------------------------------

def create_bmi_category(df):


    df["BMI_Category"] = pd.cut(

        df["BMI"],

        bins=[
            0,
            18.5,
            25,
            30,
            100
        ],

        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ]

    )


    return df



# -------------------------------------
# Age groups
# -------------------------------------

def create_age_group(df):


    df["Age_Group"] = pd.cut(

        df["Age"],

        bins=[
            0,
            30,
            45,
            60,
            100
        ],

        labels=[
            "Young",
            "Middle",
            "Older",
            "Senior"
        ]

    )


    return df



# -------------------------------------
# Glucose risk category
# -------------------------------------

def create_glucose_risk(df):


    df["Glucose_Risk"] = pd.cut(

        df["Glucose"],

        bins=[
            0,
            100,
            125,
            300
        ],

        labels=[
            "Normal",
            "Prediabetes",
            "High"
        ]

    )


    return df



# -------------------------------------
# Diabetes risk score
# -------------------------------------

def create_risk_score(df):


    df["Risk_Score"] = (

        (df["BMI"] > 30).astype(int)

        +

        (df["Age"] > 50).astype(int)

        +

        (df["Glucose"] > 125).astype(int)

        +

        (df["BloodPressure"] > 80).astype(int)

    )


    return df



# -------------------------------------
# Run pipeline
# -------------------------------------

df = create_bmi_category(df)

df = create_age_group(df)

df = create_glucose_risk(df)

df = create_risk_score(df)



df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "Feature engineering completed"
)


print(
    df.head()
)