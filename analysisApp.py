import streamlit as st
import pandas as pd
import joblib

st.write("app started successfuly")
st.write("second line of page")

from xgboost import XGBClassifier

# Load model (JSON format)
model = XGBClassifier()
model.load_model("xgb_model.json")
# st.write(type(model))

# Load scaler and feature columns (PKL format)
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Loan Default Risk Predictor")
st.write("Enter applicant details to check if they are likely to be a **defaulter** or **non-defaulter**.")

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
person_income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, value=10000, step=500)
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=40.0, value=11.0)
loan_percent_income = st.number_input("Loan as % of Income", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=5)


grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
loan_grade_letter = st.selectbox("Loan Grade", list(grade_map.keys()))
loan_grade = grade_map[loan_grade_letter]

home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"])

loan_intent = st.selectbox("Loan Intent", [
    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
    "MEDICAL", "PERSONAL", "VENTURE"
])

default_on_file = st.selectbox("Previous Default on File?", ["Y", "N"])

if st.button("Predict"):
    # Build a single-row dict with all 20 features, defaulting one-hot cols to 0
    input_dict = {col: 0 for col in feature_columns}

    input_dict['person_age'] = person_age
    input_dict['person_income'] = person_income
    input_dict['person_emp_length'] = person_emp_length
    input_dict['loan_grade'] = loan_grade
    input_dict['loan_amnt'] = loan_amnt
    input_dict['loan_int_rate'] = loan_int_rate
    input_dict['loan_percent_income'] = loan_percent_income
    input_dict['cb_person_cred_hist_length'] = cb_person_cred_hist_length

    # Set the correct one-hot columns to 1
    input_dict[f'person_home_ownership_{home_ownership}'] = 1
    input_dict[f'loan_intent_{loan_intent}'] = 1
    input_dict[f'cb_person_default_on_file_{default_on_file}'] = 1

    # Build dataframe in the exact column order the model expects
    input_df = pd.DataFrame([input_dict])[feature_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
     
    if prediction == 1:
        st.error(f"⚠️ Predicted: **Defaulter** (Risk Score: {probability:.2%})")
    else:
        st.success(f"✅ Predicted: **Non-Defaulter** (Risk Score: {probability:.2%})") 
    
