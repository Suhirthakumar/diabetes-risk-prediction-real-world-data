"""
Data Cleaning Pipeline

Diabetes Risk Prediction Project

Steps:
- Load raw diabetes dataset
- Check missing values
- Handle invalid clinical measurements
- Remove duplicates
- Save cleaned dataset
"""


import pandas as pd
import numpy as np
import os



# -----------------------------------------
# Paths
# -----------------------------------------

RAW_DATA = "data/raw/diabetes.csv"

PROCESSED_PATH = "data/processed"

OUTPUT_FILE = (
    f"{PROCESSED_PATH}/diabetes_cleaned.csv"
)


os.makedirs(
    PROCESSED_PATH,
    exist_ok=True
)



# -----------------------------------------
# Load dataset
# -----------------------------------------

def load_data():

    df = pd.read_csv(
        RAW_DATA
    )

    return df



# -----------------------------------------
# Data quality check
# -----------------------------------------

def quality_check(df):

    print("\nDataset shape:")
    print(df.shape)


    print("\nMissing values:")
    print(
        df.isnull().sum()
    )


    print("\nDuplicate records:")
    print(
        df.duplicated().sum()
    )



# -----------------------------------------
# Replace biologically impossible values
# -----------------------------------------

def replace_invalid_values(df):

    columns_with_zero = [

        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"

    ]


    for col in columns_with_zero:

        df[col] = df[col].replace(
            0,
            np.nan
        )


    return df



# -----------------------------------------
# Missing value treatment
# -----------------------------------------

def handle_missing_values(df):


    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns


    for col in numerical_columns:

        median = df[col].median()


        df[col] = df[col].fillna(
            median
        )


    return df



# -----------------------------------------
# Remove duplicates
# -----------------------------------------

def remove_duplicates(df):

    df = df.drop_duplicates()

    return df



# -----------------------------------------
# Main pipeline
# -----------------------------------------

if __name__ == "__main__":


    print(
        "Starting data cleaning..."
    )


    df = load_data()


    quality_check(df)


    df = replace_invalid_values(
        df
    )


    df = handle_missing_values(
        df
    )


    df = remove_duplicates(
        df
    )


    print(
        "\nCleaned dataset:"
    )

    print(
        df.shape
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print(
        f"\nSaved: {OUTPUT_FILE}"
    )