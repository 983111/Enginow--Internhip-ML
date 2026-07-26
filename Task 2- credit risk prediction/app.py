# app.py
# Streamlit web application for the Credit Risk Prediction System.
# Loads the trained pipeline (preprocessing + best model) saved as best_model.pkl
# and exposes a simple form-based interface for predicting default risk.

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Prediction System",
    page_icon=":bar_chart:",
    layout="centered"
)

# -----------------------------------------------------------------------
# Load the trained model pipeline
# -----------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str = "best_model.pkl"):
    """Load the saved scikit-learn pipeline (preprocessing + classifier)."""
    return joblib.load(model_path)


model_pipeline = load_model()

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------
st.title("Credit Risk Prediction System")
st.write(
    "Enter applicant details below to estimate the probability of loan default "
    "and receive a High Risk / Low Risk classification."
)

st.divider()

# -----------------------------------------------------------------------
# Input form
# -----------------------------------------------------------------------
with st.form("applicant_form"):
    st.subheader("Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("Income (annual)", min_value=0.0, value=50000.0, step=1000.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        employment_type = st.selectbox(
            "Employment Type",
            options=["Salaried", "Self-Employed", "Business", "Unemployed"]
        )
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    with col2:
        loan_duration_months = st.selectbox(
            "Loan Duration (months)",
            options=[12, 24, 36, 48, 60, 84, 120]
        )
        previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=20, value=0)
        collateral = st.selectbox("Collateral", options=["Yes", "No"])
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("Predict")

# -----------------------------------------------------------------------
# Feature engineering to mirror the notebook pipeline exactly
# -----------------------------------------------------------------------
def build_feature_row(income, age, employment_type, loan_amount, credit_score,
                       loan_duration_months, previous_defaults, collateral, dependents):
    """Recreate the engineered features used during model training."""
    debt_to_income_ratio = loan_amount / income if income > 0 else 0.0
    loan_burden_ratio = (loan_amount / loan_duration_months) / (income / 12) if income > 0 else 0.0
    total_obligation_ratio = (dependents + previous_defaults * 2) / (income / 10000) if income > 0 else 0.0

    if age <= 25:
        age_group = "18-25"
    elif age <= 35:
        age_group = "26-35"
    elif age <= 45:
        age_group = "36-45"
    elif age <= 55:
        age_group = "46-55"
    elif age <= 65:
        age_group = "56-65"
    else:
        age_group = "66+"

    row = pd.DataFrame([{
        "employment_type": employment_type,
        "collateral": collateral,
        "age_group": age_group,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_duration_months": loan_duration_months,
        "dependents": dependents,
        "previous_defaults": previous_defaults,
        "debt_to_income_ratio": debt_to_income_ratio,
        "loan_burden_ratio": loan_burden_ratio,
        "total_obligation_ratio": total_obligation_ratio
    }])
    return row


# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------
if submitted:
    input_row = build_feature_row(
        income, age, employment_type, loan_amount, credit_score,
        loan_duration_months, previous_defaults, collateral, dependents
    )

    prediction = model_pipeline.predict(input_row)[0]
    probability = model_pipeline.predict_proba(input_row)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High Risk of Default")
    else:
        st.success(f"Low Risk of Default")

    st.metric(label="Predicted Default Probability", value=f"{probability * 100:.2f}%")
    st.progress(float(min(max(probability, 0.0), 1.0)))

    with st.expander("View input features used for this prediction"):
        st.dataframe(input_row)

st.divider()
st.caption(
    "This tool provides a data-driven estimate of default risk and should be used "
    "as a decision-support aid alongside, not a replacement for, professional underwriting judgment."
)
