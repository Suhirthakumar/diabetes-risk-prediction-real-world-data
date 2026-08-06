"""
Database Management Module

Creates and manages SQLite database
for diabetes analytics project.
"""


import sqlite3
import pandas as pd

import os



DATABASE_PATH = "database/diabetes.db"



def create_connection():

    os.makedirs(
        "database",
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection



def create_patient_table():

    connection = create_connection()


    query = """

    CREATE TABLE IF NOT EXISTS patients (

        Patient_ID INTEGER PRIMARY KEY,

        Pregnancies INTEGER,

        Glucose REAL,

        BloodPressure REAL,

        BMI REAL,

        Age INTEGER,

        Outcome INTEGER

    )

    """


    connection.execute(query)

    connection.commit()

    connection.close()



def load_patient_data(csv_file):


    connection = create_connection()


    df = pd.read_csv(
        csv_file
    )


    df.to_sql(

        "patients",

        connection,

        if_exists="replace",

        index=False

    )


    connection.close()



def get_patients():

    connection = create_connection()


    query = """

    SELECT *

    FROM patients

    """


    df = pd.read_sql(

        query,

        connection

    )


    connection.close()


    return df



if __name__ == "__main__":


    create_patient_table()


    load_patient_data(

        "data/processed/diabetes_features.csv"

    )


    print(

        get_patients().head()

    )