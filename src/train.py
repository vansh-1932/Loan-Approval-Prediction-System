import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/train.csv")

# =========================
# Feature Engineering
# =========================

df["TotalIncome"] = (
    df["ApplicantIncome"]
    + df["CoapplicantIncome"]
)

df["LoanIncomeRatio"] = (
    df["LoanAmount"]
    / (df["TotalIncome"] + 1)
)

df["LogIncome"] = np.log1p(
    df["TotalIncome"]
)

# =========================
# Remove Loan_ID
# =========================

df.drop("Loan_ID", axis=1, inplace=True)

# =========================
# Features & Target
# =========================

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# =========================
# Column Types
# =========================

categorical_cols = X.select_dtypes(
    include="object"
).columns

numerical_cols = X.select_dtypes(
    exclude="object"
).columns

# =========================
# Preprocessing
# =========================

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

numerical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "cat",
        categorical_transformer,
        categorical_cols
    ),
    (
        "num",
        numerical_transformer,
        numerical_cols
    )
])

# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Models
# =========================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=2000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
}

best_model = None
best_accuracy = 0

# =========================
# Training Loop
# =========================

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline

# =========================
# Save Model
# =========================

joblib.dump(
    best_model,
    "models/loan_model.pkl"
)

print("\n======================")
print("Best Accuracy:", best_accuracy)
print("Model Saved Successfully!")
print("======================")