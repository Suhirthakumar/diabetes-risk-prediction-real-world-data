"""
Automated Research Report Generator

Diabetes Risk Prediction Project

Creates a PDF report containing:
- Dataset information
- Machine learning performance
- Clinical interpretation
- Treatment effect analysis

"""


import os
import pandas as pd

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet



# --------------------------------------------------
# Paths
# --------------------------------------------------

REPORT_PATH = (
    "reports/diabetes_prediction_report.pdf"
)


os.makedirs(
    "reports",
    exist_ok=True
)



# --------------------------------------------------
# Load results
# --------------------------------------------------

model_results = pd.read_csv(
    "reports/model_results.csv"
)


treatment_results = pd.read_csv(
    "reports/treatment_effect_results.csv"
)


feature_importance = pd.read_csv(
    "reports/logistic_coefficients.csv"
)



# --------------------------------------------------
# Create PDF
# --------------------------------------------------

doc = SimpleDocTemplate(
    REPORT_PATH
)


styles = getSampleStyleSheet()


content = []



# --------------------------------------------------
# Title
# --------------------------------------------------

content.append(

    Paragraph(
        "Diabetes Risk Prediction Using Machine Learning",
        styles["Title"]
    )

)


content.append(
    Spacer(1,20)
)



# --------------------------------------------------
# Introduction
# --------------------------------------------------

text = """

This report summarises an end-to-end computational
health analytics pipeline for diabetes risk prediction.

The workflow includes:

- Data preprocessing
- Feature engineering
- Machine learning modelling
- Clinical validation
- Explainable AI
- Treatment effect analysis

"""

content.append(

    Paragraph(
        text,
        styles["BodyText"]
    )

)



content.append(
    Spacer(1,20)
)



# --------------------------------------------------
# Model Results
# --------------------------------------------------

content.append(

    Paragraph(
        "Machine Learning Model Performance",
        styles["Heading2"]
    )

)



model_table = [

    list(model_results.columns)

]


for row in model_results.values.tolist():

    model_table.append(row)



table = Table(
    model_table
)


table.setStyle(

    TableStyle(
        [
            ("GRID",(0,0),(-1,-1),0.5,None)
        ]
    )

)


content.append(table)


content.append(
    Spacer(1,20)
)



# --------------------------------------------------
# Treatment effect
# --------------------------------------------------

content.append(

    Paragraph(
        "Treatment Effect Analysis",
        styles["Heading2"]
    )

)



treatment_table = [

    list(treatment_results.columns)

]


for row in treatment_results.values.tolist():

    treatment_table.append(row)



table2 = Table(
    treatment_table
)


table2.setStyle(

    TableStyle(
        [
            ("GRID",(0,0),(-1,-1),0.5,None)
        ]
    )

)


content.append(table2)



content.append(
    Spacer(1,20)
)



# --------------------------------------------------
# Feature importance
# --------------------------------------------------

content.append(

    Paragraph(
        "Important Predictive Variables",
        styles["Heading2"]
    )

)



feature_text = """

The predictive modelling analysis identified
clinical variables associated with diabetes risk.

Important variables included:

"""

content.append(

    Paragraph(
        feature_text,
        styles["BodyText"]
    )

)


for item in feature_importance.head(5).values:

    content.append(

        Paragraph(

            f"{item[0]} : {item[1]:.3f}",

            styles["BodyText"]

        )

    )



content.append(
    Spacer(1,20)
)



# --------------------------------------------------
# Add figures
# --------------------------------------------------

content.append(

    Paragraph(
        "Model Validation Figures",
        styles["Heading2"]
    )

)


figures = [

    "figures/roc_curve.png",

    "figures/calibration_curve.png",

    "figures/shap_summary.png"

]


for fig in figures:


    if os.path.exists(fig):

        content.append(

            Image(
                fig,
                width=350,
                height=250
            )

        )

        content.append(
            Spacer(1,15)
        )



# --------------------------------------------------
# Build report
# --------------------------------------------------

doc.build(
    content
)



print(
    "Research report generated:"
)

print(
    REPORT_PATH
)