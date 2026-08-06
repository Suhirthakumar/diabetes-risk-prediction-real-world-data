"""
Exploratory Data Analysis

Diabetes Risk Prediction Project

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# ----------------------------------
# Paths
# ----------------------------------

DATA_PATH = (
    "data/raw/diabetes.csv"
)


FIGURE_PATH = "figures"


os.makedirs(
    FIGURE_PATH,
    exist_ok=True
)



# ----------------------------------
# Load dataset
# ----------------------------------

df = pd.read_csv(
    DATA_PATH
)


print(
    "Dataset size:"
)

print(
    df.shape
)



# ----------------------------------
# Basic information
# ----------------------------------

print(
    "\nDataset Information"
)

print(
    df.info()
)



print(
    "\nMissing Values"
)

print(
    df.isnull().sum()
)



# ----------------------------------
# Statistical summary
# ----------------------------------

print(
    "\nSummary Statistics"
)


print(
    df.describe()
)



# ----------------------------------
# Outcome distribution
# ----------------------------------

plt.figure(
    figsize=(6,4)
)


sns.countplot(
    data=df,
    x="Outcome"
)


plt.title(
    "Diabetes Outcome Distribution"
)


plt.savefig(
    f"{FIGURE_PATH}/outcome_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ----------------------------------
# Age distribution
# ----------------------------------

plt.figure(
    figsize=(7,4)
)


sns.histplot(
    data=df,
    x="Age",
    bins=20,
    kde=True
)


plt.title(
    "Age Distribution"
)


plt.savefig(
    f"{FIGURE_PATH}/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ----------------------------------
# BMI vs Diabetes
# ----------------------------------

plt.figure(
    figsize=(7,5)
)


sns.boxplot(
    data=df,
    x="Outcome",
    y="BMI"
)


plt.title(
    "BMI Distribution by Diabetes Status"
)


plt.savefig(
    f"{FIGURE_PATH}/bmi_diabetes.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ----------------------------------
# Correlation matrix
# ----------------------------------

plt.figure(
    figsize=(10,8)
)


corr = df.corr()


sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm"
)


plt.title(
    "Feature Correlation Matrix"
)


plt.savefig(
    f"{FIGURE_PATH}/correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print(
    "\nEDA completed successfully"
)