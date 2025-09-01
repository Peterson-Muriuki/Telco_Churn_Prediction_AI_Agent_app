# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime

# --- Load the trained model ---
model = joblib.load("churn_model.pkl")

# --- Streamlit Page Config ---
st.set_page_config(page_title="🌟 Telco Customer Churn Predictor", layout="centered")

# --- App Title ---
st.title("🌟 Telco Customer Churn Prediction AI Agent")
st.write("Predict whether a customer is likely to churn based on their profile and account data.")

# --- Path for commit log ---
log_file = "prediction_log.csv"

def commit_predictions(df):
    """Append predictions to a CSV log file."""
    if os.path.exists(log_file):
        old_log = pd.read_csv(log_file)
        df_to_log = pd.concat([old_log, df], ignore_index=True)
    else:
        df_to_log = df
    df_to_log.to_csv(log_file, index=False)

# =========================
# 1️⃣ Bulk Prediction Section
# =========================
st.header("📁 Bulk Churn Prediction")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file, engine='openpyxl')

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    ids = data['CustomerID'] if 'CustomerID' in data.columns else pd.Series(range(len(data)), name='ID')

    model_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    missing_cols = [col for col in model_features if col not in data.columns]
    if missing_cols:
        st.warning(f"Uploaded data is missing these required columns: {missing_cols}")
    else:
        df_model = data[model_features].copy()

        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod'
        ]
        for col in categorical_cols:
            df_model[col] = LabelEncoder().fit_transform(df_model[col].astype(str))

        for col in ['MonthlyCharges', 'TotalCharges']:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

        predictions = model.predict(df_model)
        prediction_probs = model.predict_proba(df_model)

        df_result = df_model.copy()
        df_result['CustomerID'] = ids
        df_result['Churn_Prediction'] = ["Yes" if p==1 else "No" for p in predictions]
        df_result['Probability_Not_Churning'] = prediction_probs[:,0]
        df_result['Probability_Churning'] = prediction_probs[:,1]
        df_result['Prediction_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.subheader("Bulk Predictions")
        st.dataframe(df_result.head(20))

        csv = df_result.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )

        # Commit predictions
        commit_predictions(df_result)
        st.success("✅ Predictions committed to log.")

# =========================
# 2️⃣ Manual Input Section
# =========================
st.sidebar.header("Enter Customer Details")

def user_input_features():
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
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ))
    monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, value=1000.0)

    return pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

input_df = user_input_features()

categorical_cols_manual = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]
for col in categorical_cols_manual:
    input_df[col] = LabelEncoder().fit_transform(input_df[col].astype(str))

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

manual_result = input_df.copy()
manual_result['Churn_Prediction'] = ["Yes" if p==1 else "No" for p in prediction]
manual_result['Probability_Not_Churning'] = prediction_proba[:,0]
manual_result['Probability_Churning'] = prediction_proba[:,1]
manual_result['Prediction_Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.subheader("Manual Input Prediction")
st.write(f"Customer likely to churn? **{manual_result['Churn_Prediction'][0]}**")

st.subheader("Prediction Probability")
st.write(f"Probability of not churning: {manual_result['Probability_Not_Churning'][0]*100:.2f}%")
st.write(f"Probability of churning: {manual_result['Probability_Churning'][0]*100:.2f}%")

# Commit manual input prediction
commit_predictions(manual_result)
st.success("✅ Manual input prediction committed to log.")
