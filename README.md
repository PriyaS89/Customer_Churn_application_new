# Customer Churn Prediction

A beginner-friendly machine learning application using Python, Pandas, Scikit-learn and Streamlit. It uses the Telco Customer Churn dataset to estimate whether a customer may leave the telecom service.

## Project files

Keep these files together in the root of your GitHub repository:

```text
customer-churn-prediction/
├── app.py
├── after_integration_app.py
├── requirements.txt
├── README.md
└── Telco-Customer-Churn.csv
```

The dataset may also retain its original filename: `WA_Fn-UseC_-Telco-Customer-Churn.csv`. The code supports either filename.

## Dataset columns used

| Column | Meaning |
|---|---|
| `tenure` | Number of months the customer has stayed |
| `MonthlyCharges` | Customer's monthly charges |
| `Contract` | Contract type |
| `Churn` | Target column: Yes or No |

The model uses `tenure`, `MonthlyCharges` and `Contract` to predict `Churn`.

## Machine learning workflow

1. Load the CSV and select the required columns.
2. Convert numeric columns to numeric values.
3. Fill missing values using imputers.
4. Scale numeric features and encode contract categories.
5. Split data into 80% training and 20% testing sets.
6. Train a Logistic Regression model.
7. Display test accuracy and a confusion matrix.
8. Accept customer details and show a churn prediction.

## Run locally

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Upload `app.py`, `after_integration_app.py`, `requirements.txt`, `README.md` and the Telco CSV to your GitHub repository.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Select **Create app** and choose your repository.
4. Select the `main` branch.
5. Enter `app.py` as the main file path.
6. Select **Deploy** and wait for the application to launch.

## Important

- Keep the CSV in the same repository folder as `app.py`.
- Do not upload passwords, API keys, access tokens or confidential customer information.
- The model trains when the application starts, so the first load may take a little time.
- Predictions are estimates for learning purposes and are not guaranteed outcomes.
