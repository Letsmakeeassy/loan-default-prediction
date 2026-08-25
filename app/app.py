import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load saved model, scaler, and columns
model = joblib.load('../src/model.pkl')
scaler = joblib.load('../src/scaler.pkl')
columns = joblib.load('../src/columns.pkl')
background_sample = joblib.load('../src/background_sample.pkl')

st.set_page_config(page_title="Loan Default Risk Predictor", layout="centered")
st.title("🏦 Loan Default Risk Predictor")
st.write("Enter applicant details to get a risk score and explanation.")

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual Income ($)", min_value=0, value=60000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=20000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    months_employed = st.number_input("Months Employed", min_value=0, value=36)
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=3)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=10.0)

with col2:
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
    has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

if st.button("Predict Risk"):
    # Build input row matching training format
    input_dict = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount,
        'CreditScore': credit_score, 'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio,
        'HasMortgage': 1 if has_mortgage == "Yes" else 0,
        'HasDependents': 1 if has_dependents == "Yes" else 0,
        'HasCoSigner': 1 if has_cosigner == "Yes" else 0,
        'LoanToIncomeRatio': loan_amount / income if income > 0 else 0,
    }

    # One-hot encode categorical fields manually to match training columns
    for col in columns:
        if col.startswith('Education_'):
            input_dict[col] = 1 if col == f'Education_{education}' else 0
        elif col.startswith('EmploymentType_'):
            input_dict[col] = 1 if col == f'EmploymentType_{employment_type}' else 0
        elif col.startswith('MaritalStatus_'):
            input_dict[col] = 1 if col == f'MaritalStatus_{marital_status}' else 0
        elif col.startswith('LoanPurpose_'):
            input_dict[col] = 1 if col == f'LoanPurpose_{loan_purpose}' else 0

    # Build dataframe in the exact column order the model expects
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    risk_prob = model.predict_proba(input_scaled)[0][1]

    # --- Display results ---
    st.subheader("Results")
    st.metric("Default Risk Probability", f"{risk_prob*100:.1f}%")

    if risk_prob < 0.3:
        st.success("Risk Category: LOW")
    elif risk_prob < 0.6:
        st.warning("Risk Category: MEDIUM")
    else:
        st.error("Risk Category: HIGH")

    # SHAP explanation
    st.subheader("Why this prediction?")
    explainer = shap.LinearExplainer(model, background_sample)
    shap_values = explainer.shap_values(input_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=columns
    ), show=False)
    st.pyplot(fig)