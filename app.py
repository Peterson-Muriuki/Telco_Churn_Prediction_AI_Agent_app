# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import openpyxl

# --- Load the trained model ---
model = joblib.load("churn_model.pkl")

# --- Streamlit Page Config ---
st.set_page_config(page_title="🌟 Telco Customer Churn Predictor", layout="centered")

# --- App Title ---
st.title("🌟 Telco Customer Churn Prediction AI Agent")
st.write("Predict whether a customer is likely to churn based on their profile and account data.")

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
        data = pd.read_excel(uploaded_file)

    st.write("Preview of uploaded data:")
    st.dataframe(data.head())

    # Keep customer IDs for reference if available
    if 'CustomerID' in data.columns:
        ids = data['CustomerID']
    else:
        ids = pd.Series(range(len(data)), name='ID')

    # Columns expected by the model
    model_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    # Check for missing required columns
    missing_cols = [col for col in model_features if col not in data.columns]
    if missing_cols:
        st.warning(f"Uploaded data is missing these required columns: {missing_cols}")
    else:
        df_model = data[model_features].copy()

        # Encode categorical columns
        categorical_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod'
        ]
        for col in categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col])

        # Ensure numeric columns are clean
        for col in ['MonthlyCharges', 'TotalCharges']:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

        # Predictions
        predictions = model.predict(df_model)
        prediction_probs = model.predict_proba(df_model)

        # Prepare result dataframe
        df_result = df_model.copy()
        df_result['CustomerID'] = ids
        df_result['Churn_Prediction'] = ["Yes" if p==1 else "No" for p in predictions]
        df_result['Probability_Not_Churning'] = prediction_probs[:,0]
        df_result['Probability_Churning'] = prediction_probs[:,1]

        st.subheader("Bulk Predictions")
        st.dataframe(df_result.head(20))  # Show top 20 results

        # Download predictions
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )

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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Encode manual input
categorical_cols_manual = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
    'PaperlessBilling', 'PaymentMethod'
]
for col in categorical_cols_manual:
    le = LabelEncoder()
    input_df[col] = le.fit_transform(input_df[col])

# Predict manual input
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Manual Input Prediction")
churn_label = "Yes" if prediction[0] == 1 else "No"
st.write(f"Customer likely to churn? **{churn_label}**")

st.subheader("Prediction Probability")
st.write(f"Probability of not churning: {prediction_proba[0][0]*100:.2f}%")
st.write(f"Probability of churning: {prediction_proba[0][1]*100:.2f}%")
