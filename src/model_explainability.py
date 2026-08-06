"""
Explainable AI using SHAP
"""

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# Load data
df = pd.read_csv("data/processed/diabetes_features.csv")

# Remove categorical engineered variables
df = df.drop(
    [
        "BMI_Category",
        "Age_Group",
        "Glucose_Risk"
    ],
    axis=1
)

X = df.drop("Outcome", axis=1)

# Load scaler
scaler = joblib.load("models/scaler.pkl")
X_scaled = scaler.transform(X)

# Load best model
model = joblib.load("models/random_forest.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_scaled)

# Summary plot
shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.savefig(
    "figures/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Bar plot
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.savefig(
    "figures/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.plot.bar(
    x="Feature",
    y="Importance",
    figsize=(8,5)
)

plt.tight_layout()

plt.savefig(
    "figures/feature_importance.png",
    dpi=300
)

plt.close()

print("Explainability completed.")