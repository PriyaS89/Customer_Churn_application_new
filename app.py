"""Streamlit user interface for the Customer Churn Prediction project."""
import streamlit as st
from after_integration_app import train_model, predict_customer

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to estimate whether the customer may leave the telecom service.")

st.sidebar.header("Customer Details")
customer_id = st.sidebar.text_input("Customer ID", "C001")
tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=80.0, step=5.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
predict_button = st.sidebar.button("Predict Customer Churn", type="primary")

@st.cache_resource
def get_model():
    return train_model()

try:
    with st.spinner("Loading data and training the model..."):
        model, accuracy, matrix = get_model()
    st.success("The machine learning model is ready.")
    st.metric("Model Test Accuracy", f"{accuracy * 100:.2f}%")
    st.divider()

    if predict_button:
        prediction, probability = predict_customer(model, tenure, monthly_charges, contract)
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("The model predicts that this customer is likely to churn.")
        else:
            st.success("The model predicts that this customer is not likely to churn.")
        st.metric("Estimated Churn Probability", f"{probability * 100:.2f}%")
        st.write("**Customer details**")
        st.write(f"- Customer ID: {customer_id}")
        st.write(f"- Tenure: {tenure} months")
        st.write(f"- Monthly Charges: {monthly_charges}")
        st.write(f"- Contract: {contract}")
    else:
        st.info("Enter customer details in the sidebar and click Predict Customer Churn.")

    with st.expander("View Model Evaluation"):
        st.write("Confusion Matrix")
        st.write(matrix)

except Exception as error:
    st.error("The application could not be started. Check the dataset and uploaded files.")
    st.code(str(error))

st.caption("Educational demonstration only. Predictions are estimates, not guaranteed outcomes.")
