# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Load the trained model ---
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="Telco Customer Churn Predictor", layout="centered")

# --- App Title ---
st.title("Telco Customer Churn Prediction AI Agent")
st.write(
    "Predict whether a customer is likely to churn based on their profile and account data."
)

# --- Sidebar for user input ---
st.sidebar.header("Enter Customer Details")

def user_input_features():
    # You can adjust these fields to match your model's features exactly
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    senior_citizen = st.sidebar.selectbox("Senior Citizen", (0, 1))
    partner = st.sidebar.selectbox("Partner", ("Yes", "No"))
    dependents = st.sidebar.selectbox("Dependents", ("Yes", "No"))
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.sidebar.selectbox("Phone Service", ("Yes", "No"))
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ("Yes", "No", "No phone service"))
    internet_service = st.sidebar.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    online_security = st.sidebar.selectbox("Online Security", ("Yes", "No", "No internet service"))
    online_backup = st.sidebar.selectbox("Online Backup", ("Yes", "No", "No internet service"))
    device_protection = st.sidebar.selectbox("Device Protection", ("Yes", "No", "No internet service"))
    tech_support = st.sidebar.selectbox("Tech Support", ("Yes", "No", "No internet service"))
    streaming_tv = st.sidebar.selectbox("Streaming TV", ("Yes", "No", "No internet service"))
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ("Yes", "No", "No internet service"))
    contract = st.sidebar.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ("Yes", "No"))
    payment_method = st.sidebar.selectbox("Payment Method", (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ))
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Encode input the same way as model training ---
# For simplicity, we assume all categorical variables were label-encoded
from sklearn.preprocessing import LabelEncoder

for col in input_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    input_df[col] = le.fit_transform(input_df[col])

# --- Prediction ---
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
churn_label = "Yes" if prediction[0] == 1 else "No"
st.write(f"Customer likely to churn? **{churn_label}**")

st.subheader("Prediction Probability")
st.write(f"Probability of not churning: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Probability of churning: {prediction_proba[0][1]*100:.2f}%")
