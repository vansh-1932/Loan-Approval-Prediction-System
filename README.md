# 🏦 Loan Approval Prediction System

## 📌 Overview

This project is an end-to-end Machine Learning application that predicts whether a loan application is likely to be approved based on applicant details such as income, education, employment status, credit history, and loan amount.

The project includes data preprocessing, feature engineering, model training, model evaluation, and a Streamlit web application for real-time predictions.

---

## 🚀 Features

* Data Cleaning and Preprocessing
* Missing Value Handling
* Feature Engineering
* Logistic Regression Model
* Random Forest Model
* Model Comparison
* Interactive Streamlit Web Application
* Real-Time Loan Approval Prediction

---

## 📂 Project Structure

LoanApprovalSystem/

├── api/

├── data/

│ ├── train.csv

│ └── test.csv

├── frontend/

│ └── app.py

├── models/

│ └── loan_model.pkl

├── notebooks/

│ └── EDA.ipynb

├── reports/

│ ├── project_report.md

│ ├── training_result.png

│ ├── approved.png

│ └── rejected.png

├── src/

│ └── train.py

├── README.md

└── requirements.txt

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

## 📊 Feature Engineering

The following custom features were created to improve model performance:

* TotalIncome = ApplicantIncome + CoapplicantIncome
* LoanIncomeRatio = LoanAmount / TotalIncome
* LogIncome = log(TotalIncome)

---

## 🤖 Models Trained

### Logistic Regression

Accuracy: 78.86%

### Random Forest

Accuracy: 80.49%

Best Model Selected: Random Forest

---

## 📈 Results

The Random Forest model achieved the highest accuracy of **80.49%** on the test dataset.

---

## 🖥️ Running the Project

### Install Dependencies

pip install -r requirements.txt

### Train the Model

python src/train.py

### Run the Streamlit App

streamlit run frontend/app.py

---

## 📸 Screenshots

### Training Result

See: reports/training_result.png

### Approved Prediction

See: reports/approved.png

### Rejected Prediction

See: reports/rejected.png

---

## 🎯 Future Improvements

* XGBoost Integration
* Hyperparameter Tuning
* SHAP Explainability
* Loan Risk Scoring
* Cloud Deployment

---

## 👨‍💻 Author

Vansh

Machine Learning & Data Science Enthusiast
