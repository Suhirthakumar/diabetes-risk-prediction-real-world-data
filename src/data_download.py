"""
Download Diabetes Dataset

Project:
Diabetes Risk Prediction and Treatment Analytics

Author:
Your Name
"""


import os
import requests
import pandas as pd


# -------------------------------------------------
# Dataset URL
# -------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "plotly/datasets/master/diabetes.csv"
)


# -------------------------------------------------
# Create directories
# -------------------------------------------------

RAW_DATA_PATH = "data/raw"

os.makedirs(
    RAW_DATA_PATH,
    exist_ok=True
)


# -------------------------------------------------
# Download function
# -------------------------------------------------

def download_dataset():

    file_path = os.path.join(
        RAW_DATA_PATH,
        "diabetes.csv"
    )


    if os.path.exists(file_path):

        print(
            "Dataset already exists"
        )

        return file_path


    print(
        "Downloading diabetes dataset..."
    )


    response = requests.get(
        DATA_URL
    )


    response.raise_for_status()


    with open(
        file_path,
        "wb"
    ) as file:

        file.write(
            response.content
        )


    print(
        f"Dataset saved: {file_path}"
    )


    return file_path



# -------------------------------------------------
# Load data check
# -------------------------------------------------

if __name__ == "__main__":


    path = download_dataset()


    df = pd.read_csv(path)


    print("\nDataset Shape:")
    print(df.shape)


    print("\nColumns:")
    print(df.columns.tolist())


    print("\nFirst five rows:")
    print(df.head())