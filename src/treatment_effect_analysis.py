"""
Treatment Effect Analysis

Mini Project 5
Diabetes Risk Prediction Using Real-World Healthcare Data

This script demonstrates a simplified observational treatment-effect
analysis using a simulated treatment variable.

Author: Suhirthakumar Puvanendran
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------
# Create folders
# -----------------------------------------------------

os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------

df = pd.read_csv("data/processed/diabetes_features.csv")

# -----------------------------------------------------
# Simulate treatment
#
# In real-world research this would be:
# GLP-1 receptor agonist
# Metformin
# Insulin
# SGLT2 inhibitor
#
# Here we simulate treatment using risk score.
# -----------------------------------------------------

np.random.seed(42)

probability = np.where(
    df["Risk_Score"] >= 3,
    0.70,
    0.25
)

df["Treatment"] = np.random.binomial(
    1,
    probability
)

# -----------------------------------------------------
# Outcome rates
# -----------------------------------------------------

summary = (

    df.groupby("Treatment")["Outcome"]

      .agg(
          Patients="count",
          Events="sum",
          Mean_Risk="mean"
      )

)

print(summary)

summary.to_csv(
    "reports/treatment_summary.csv"
)

# -----------------------------------------------------
# Relative Risk
# -----------------------------------------------------

treated = summary.loc[1, "Mean_Risk"]

control = summary.loc[0, "Mean_Risk"]

relative_risk = treated / control

risk_difference = treated - control

# -----------------------------------------------------
# Odds Ratio
# -----------------------------------------------------

contingency = pd.crosstab(
    df["Treatment"],
    df["Outcome"]
)

a = contingency.loc[1,1]
b = contingency.loc[1,0]
c = contingency.loc[0,1]
d = contingency.loc[0,0]

odds_ratio = (a*d)/(b*c)

chi2, p_value, _, _ = chi2_contingency(
    contingency
)

# -----------------------------------------------------
# Logistic Regression Adjustment
# -----------------------------------------------------

X = df[
    [
        "Treatment",
        "Age",
        "BMI",
        "Glucose",
        "BloodPressure",
        "Risk_Score"
    ]
]

y = df["Outcome"]

model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)

coefficients = pd.DataFrame({

    "Variable": X.columns,

    "Coefficient": model.coef_[0]

})

coefficients.to_csv(
    "reports/logistic_coefficients.csv",
    index=False
)

# -----------------------------------------------------
# Plot outcome rates
# -----------------------------------------------------

plt.figure(figsize=(6,5))

summary["Mean_Risk"].plot(
    kind="bar"
)

plt.ylabel("Diabetes Prevalence")

plt.title("Outcome Rate by Treatment Group")

plt.tight_layout()

plt.savefig(
    "figures/treatment_effect.png",
    dpi=300
)

plt.close()

# -----------------------------------------------------
# Save report
# -----------------------------------------------------

report = pd.DataFrame({

    "Metric":[
        "Relative Risk",
        "Risk Difference",
        "Odds Ratio",
        "Chi-square p-value"
    ],

    "Value":[
        relative_risk,
        risk_difference,
        odds_ratio,
        p_value
    ]

})

report.to_csv(
    "reports/treatment_effect_results.csv",
    index=False
)

# -----------------------------------------------------
# Console Output
# -----------------------------------------------------

print("\nTreatment Effect Analysis")

print("-"*50)

print(f"Relative Risk : {relative_risk:.3f}")

print(f"Risk Difference : {risk_difference:.3f}")

print(f"Odds Ratio : {odds_ratio:.3f}")

print(f"Chi-square p-value : {p_value:.4f}")

print("\nLogistic Regression")

print(coefficients)

print("\nResults saved to reports/")