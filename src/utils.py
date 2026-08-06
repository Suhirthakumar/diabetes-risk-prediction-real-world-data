"""
Utility functions
"""

import os
import pandas as pd
import joblib



def load_data(path):

    return pd.read_csv(path)



def save_model(model, path):

    os.makedirs(

        os.path.dirname(path),

        exist_ok=True

    )


    joblib.dump(

        model,

        path

    )



def load_model(path):

    return joblib.load(path)



def create_directory(path):

    os.makedirs(

        path,

        exist_ok=True

    )