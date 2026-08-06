"""
Diabetes Risk Prediction Model Training

Models:
- Logistic Regression
- Random Forest
- XGBoost

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

"""


import os
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)



# -------------------------------------
# Paths
# -------------------------------------

DATA_PATH = (
    "data/processed/diabetes_features.csv"
)


MODEL_PATH = "models"


os.makedirs(
    MODEL_PATH,
    exist_ok=True
)



# -------------------------------------
# Load data
# -------------------------------------

df = pd.read_csv(
    DATA_PATH
)



# -------------------------------------
# Prepare features
# -------------------------------------

# Remove categorical engineered columns
df = df.drop(
    [
        "BMI_Category",
        "Age_Group",
        "Glucose_Risk"
    ],
    axis=1
)



X = df.drop(
    "Outcome",
    axis=1
)


y = df["Outcome"]



# -------------------------------------
# Train test split
# -------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# -------------------------------------
# Scaling
# -------------------------------------

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


joblib.dump(
    scaler,
    f"{MODEL_PATH}/scaler.pkl"
)



# -------------------------------------
# Define models
# -------------------------------------

models = {


    "logistic_regression":
        LogisticRegression(
            max_iter=1000
        ),


    "random_forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),


    "xgboost":
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

}



# -------------------------------------
# Train and evaluate
# -------------------------------------

results = []



for name, model in models.items():


    print(
        f"\nTraining {name}"
    )


    model.fit(
        X_train_scaled,
        y_train
    )


    predictions = model.predict(
        X_test_scaled
    )


    probabilities = model.predict_proba(
        X_test_scaled
    )[:,1]



    results.append({

        "Model": name,

        "Accuracy":
        accuracy_score(
            y_test,
            predictions
        ),

        "Precision":
        precision_score(
            y_test,
            predictions
        ),

        "Recall":
        recall_score(
            y_test,
            predictions
        ),

        "F1":
        f1_score(
            y_test,
            predictions
        ),

        "ROC_AUC":
        roc_auc_score(
            y_test,
            probabilities
        )

    })



    from utils import save_model


save_model(

    model,

    "models/random_forest.pkl"

)


# -------------------------------------
# Save results
# -------------------------------------

results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "reports/model_results.csv",
    index=False
)



print("\nModel Performance")

print(
    results_df
)