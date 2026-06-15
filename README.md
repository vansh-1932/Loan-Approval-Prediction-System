# Loan Approval Prediction System

## Overview

Machine Learning project that predicts whether a loan application will be approved.

## Features

- Data Cleaning
- Feature Engineering
- Logistic Regression
- Random Forest
- Streamlit Web App

## Model Performance

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 78.86% |
| Random Forest | 80.49% |

## Engineered Features

- TotalIncome
- LoanIncomeRatio
- LogIncome

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit

## Run Locally

```bash
pip install -r requirements.txt

python src/train.py

streamlit run frontend/app.py
```