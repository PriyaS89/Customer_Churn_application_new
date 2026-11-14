"""Backend: data preparation, Logistic Regression training and prediction."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges"]
CATEGORICAL = ["Contract"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"
DATA_NAMES = ["Telco-Customer-Churn.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv"]

def load_data():
    folder = Path(__file__).resolve().parent
    path = next((folder / name for name in DATA_NAMES if (folder / name).exists()), None)
    if path is None:
        raise FileNotFoundError("Upload the Telco churn CSV beside app.py. Accepted names: " + ", ".join(DATA_NAMES))
    df = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    data = df[required].copy()
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET] = data[TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1})
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)
    if data[TARGET].nunique() < 2:
        raise ValueError("The Churn column must contain both Yes and No.")
    return data

def train_model():
    data = load_data()
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    category_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", category_pipe, CATEGORICAL)
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, model.score(X_test, y_test), confusion_matrix(y_test, predictions)

def predict_customer(model, tenure, monthly_charges, contract):
    customer = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract
    }], columns=FEATURES)
    prediction = int(model.predict(customer)[0])
    probability = float(model.predict_proba(customer)[0][1])
    return prediction, probability
