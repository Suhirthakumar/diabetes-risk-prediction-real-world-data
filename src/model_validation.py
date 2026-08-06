"""
Clinical Model Validation

Evaluation:
- ROC Curve
- Precision Recall Curve
- Calibration
- Confusion Matrix

"""


import pandas as pd
import joblib
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn.metrics import (

    RocCurveDisplay,

    PrecisionRecallDisplay,

    confusion_matrix,

    ConfusionMatrixDisplay,

    classification_report

)


from sklearn.calibration import CalibrationDisplay



# ----------------------------------
# Load data
# ----------------------------------

df = pd.read_csv(
    "data/processed/diabetes_features.csv"
)



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



X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# Load best model

model = joblib.load(
    "models/xgboost.pkl"
)



# Prediction

probabilities = model.predict_proba(
    X_test
)[:,1]


predictions = model.predict(
    X_test
)



# ----------------------------------
# ROC curve
# ----------------------------------

RocCurveDisplay.from_predictions(

    y_test,

    probabilities

)


plt.title(
    "ROC Curve"
)


plt.savefig(
    "figures/roc_curve.png",
    dpi=300
)


plt.close()



# ----------------------------------
# Precision Recall
# ----------------------------------

PrecisionRecallDisplay.from_predictions(

    y_test,

    probabilities

)


plt.title(
    "Precision Recall Curve"
)


plt.savefig(
    "figures/precision_recall.png",
    dpi=300
)


plt.close()



# ----------------------------------
# Confusion matrix
# ----------------------------------

cm = confusion_matrix(

    y_test,

    predictions

)


ConfusionMatrixDisplay(

    cm

).plot()


plt.savefig(
    "figures/confusion_matrix.png",
    dpi=300
)


plt.close()



# ----------------------------------
# Calibration
# ----------------------------------

CalibrationDisplay.from_predictions(

    y_test,

    probabilities

)


plt.title(
    "Calibration Curve"
)


plt.savefig(
    "figures/calibration_curve.png",
    dpi=300
)


plt.close()



print(
    classification_report(
        y_test,
        predictions
    )
)


print(
    "Validation completed"
)