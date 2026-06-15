import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load("models/loan_model.pkl")

# Page Config
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Smart Loan Approval Prediction System")

st.markdown(
    "Predict whether a loan application is likely to be approved."
)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    Machine Learning Loan Approval System

    Model: Random Forest
    Accuracy: 80.49%
    """
)

# Input Fields

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["0", "1", "2", "3+"]
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)

co_income = st.number_input(
    "Coapplicant Income",
    min_value=0,
    value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=150
)

loan_term = st.number_input(
    "Loan Term (Months)",
    min_value=0,
    value=360
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

# Prediction

if st.button("Predict"):

    total_income = income + co_income

    loan_income_ratio = (
        loan_amount /
        (total_income + 1)
    )

    log_income = np.log1p(
        total_income
    )

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "TotalIncome": total_income,
        "LoanAmount": loan_amount,
        "LoanIncomeRatio": loan_income_ratio,
        "LogIncome": log_income,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    prediction = model.predict(
        input_data
    )[0]

    probabilities = model.predict_proba(
        input_data
    )[0]

    classes = model.classes_

    approved_index = list(classes).index("Y")

    approval_probability = (
        probabilities[approved_index]
    )

    st.subheader("Prediction Result")

    if prediction == "Y":
        st.success(
            "✅ Loan Approved"
        )
    else:
        st.error(
            "❌ Loan Rejected"
        )

    st.metric(
        "Approval Probability",
        f"{approval_probability*100:.2f}%"
    )

    st.subheader("Applicant Summary")

    st.write(
        f"Total Income: ₹{total_income:,.0f}"
    )

    st.write(
        f"Loan Amount: ₹{loan_amount:,.0f}"
    )

    st.write(
        f"Credit History: {credit_history}"
    )