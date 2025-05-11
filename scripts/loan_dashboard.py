import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Background
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: #1a2b3c;
            font-family: 'Segoe UI', sans-serif;
            color: white;
        }
  /* Base font settings for the entire app */
        html, body, [class*="css"]  {
            font-size: 1.4em !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Header area styling */
        .stApp header {
            background-color: #1a2b3c !important;
        }

        .stApp > div:first-child {
            background-color: #1a2b3c !important;
        }

        /* Increase header text size */
        .stApp header .title {
            font-size: 2.5em !important;
            color: white !important;
        }

        /* Main header styling */
        .main .block-container h1, .main .block-container h2 {
            font-size: 3em !important;
            color: white !important;
            margin-bottom: 35px;
        }

        .prediction-form {
            background-color: #233446;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin: 25px 0;
        }

        .form-section {
            margin: 20px 0;
        }

        .section-header {
            color: #ffffff;
            font-size: 1.8em;
            font-weight: 600;
            margin-bottom: 25px;
        }

        .stSelectbox > label, .stNumberInput > label {
            color: #e0e5ea;
            font-size: 1.2em;
            font-weight: 500;
            margin-bottom: 10px;
        }

        .stButton>button {
            background-color: #3498db;
            color: white;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.2em;
            font-weight: 600;
            width: 100%;
            margin-top: 25px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
        }

        .prediction-result {
            text-align: center;
            padding: 30px;
            margin-top: 25px;
            border-radius: 10px;
            background-color: #233446;
            color: white;
        }

        .main .block-container h1 {
            color: #ffffff;
            font-size: 2.8em;
            font-weight: 700;
            margin-bottom: 35px;
        }
        /* Enlarge font on Home and other pages */
        .main .block-container {
            font-size: 1.8em;
        }

        .main .block-container p, 
        .main .block-container li, 
        .main .block-container span, 
        .main .block-container div {
            font-size: 1.8em !important;
        }

        .stRadio > label {
            font-size: 1.4em !important;
            color: #ffffff;
        }

        /* Navigation text and arrow styling */
        .stRadio [role="radiogroup"] label {
            font-size: 1.8em !important;
            font-weight: 500;
            margin: 15px 0;
            color: #1a2b3c !important;  /* Dark blue color */
        }

        /* Style the navigation header */
        .sidebar .sidebar-content div:first-child label {
            color: white !important;
            font-size: 2em !important;
            font-weight: 600;
            margin-bottom: 20px;
        }

        /* Style the arrow/expand icon */
        .sidebar .sidebar-content [data-testid="stExpanderToggleIcon"] {
            color: white !important;
        }

        /* Add specific styling for each navigation item */
        .stRadio [role="radiogroup"] label:nth-child(1) { /* Home */
            font-size: 2em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(2) { /* Model Prediction */
            font-size: 1.8em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(3) { /* Feature Importance */
            font-size: 1.8em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(4) { /* Model Insights */
            font-size: 1.8em !important;
        }

        /* Improve spacing between navigation items */
        .stRadio [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        /* Style the Navigation text */
        .sidebar [data-testid="stSidebarNav"] {
            color: #1a2b3c !important;
            font-size: 2em !important;
            font-weight: 600;
            margin-bottom: 20px;
        }

        /* Keep existing navigation item styles */
        .stRadio [role="radiogroup"] label {
            font-size: 1.8em !important;
            font-weight: 500;
            margin: 15px 0;
            color: #1f4e79;
        }
        /* Navigation styling */
        .stRadio > label {
            font-size: 2em !important;
            color: #ffffff;
            font-weight: 600;
        }

        .stRadio [role="radiogroup"] label {
            font-size: 2.2em !important;
            font-weight: 600;
            margin: 20px 0;
            color: #1f4e79;
        }

        /* Specific navigation items */
        .stRadio [role="radiogroup"] label:nth-child(1) { /* Home */
            font-size: 2.4em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(2) { /* Model Prediction */
            font-size: 2.2em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(3) { /* Feature Importance */
            font-size: 2.2em !important;
        }

        .stRadio [role="radiogroup"] label:nth-child(4) { /* Model Insights */
            font-size: 2.2em !important;
        }

        /* Improve spacing between navigation items */
        .stRadio [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 25px;  /* Increased gap */
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()


# Load models
BASE_DIR = os.path.dirname(__file__)
rf_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'random_forest_model_clean.pkl'))
lr_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'logistic_model_clean.pkl'))
xgb_model = joblib.load(os.path.join(BASE_DIR, '..', 'models', 'xgboost_model_clean.pkl'))

# Preprocessing
def preprocess_input(inputs):
    return pd.DataFrame([{
        'Gender': 1 if inputs['gender'] == 'Male' else 0,
        'Married': 1 if inputs['married'] == 'Yes' else 0,
        'Dependents': 3 if inputs['dependents'] == '3+' else int(inputs['dependents']),
        'Education': 1 if inputs['education'] == 'Graduate' else 0,
        'Self_Employed': 1 if inputs['self_employed'] == 'Yes' else 0,
        'ApplicantIncome': inputs['applicant_income'],
        'CoapplicantIncome': inputs['coapplicant_income'],
        'LoanAmount': inputs['loan_amount'],
        'Loan_Amount_Term': inputs['loan_term'],
        'Credit_History': float(inputs['credit_history']),
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[inputs['property_area']],
    }])

def predict_loan(model, inputs):
    input_df = preprocess_input(inputs)
    return model.predict(input_df)[0]

# Navigation
menu = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Model Prediction", "📊 Feature Importance", "📈 Model Insights"])

# Home Page
if menu == "🏠 Home":
    st.title("💼 Loan Prediction Dashboard")
    st.markdown("""
    Welcome to the Loan Prediction Dashboard! This application uses machine learning models to predict loan approval outcomes based on applicant information.
    
    Navigate using the sidebar to:
    - Predict loan outcomes with different models.
    - View model feature importance (Random Forest).
    - Review model insights such as accuracy and confusion matrix.
    """)

# Model Prediction Page
elif menu == "🔮 Model Prediction":
    st.header("🔍 Predict Loan Approval")
    
    with st.container():
        
        st.markdown('<p class="section-header">Select Model</p>', unsafe_allow_html=True)
        model_choice = st.selectbox("Choose your prediction model:", 
                                  ("Random Forest", "Logistic Regression", "XGBoost"))
        
        # Define model based on selection
        model_mapping = {
            "Random Forest": rf_model,
            "Logistic Regression": lr_model,
            "XGBoost": xgb_model
        }
        selected_model = model_mapping[model_choice]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<p class="section-header">Personal Information</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="form-section">', unsafe_allow_html=True)
                gender = st.selectbox('👤 Gender', ('Male', 'Female'))
                married = st.selectbox('💑 Marital Status', ('Yes', 'No'))
                dependents = st.selectbox('👨‍👩‍👧‍👦 Dependents', ('0', '1', '2', '3+'))
                education = st.selectbox('🎓 Education', ('Graduate', 'Not Graduate'))
                self_employed = st.selectbox('💼 Self Employed', ('Yes', 'No'))
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<p class="section-header">Financial Information</p>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="form-section">', unsafe_allow_html=True)
                applicant_income = st.number_input('💰 Applicant Income', 0, 1000000, step=1000)
                coapplicant_income = st.number_input('💵 Coapplicant Income', 0, 1000000, step=1000)
                loan_amount = st.number_input('🏦 Loan Amount', 10, 500000, step=10)
                loan_term = st.number_input('📅 Loan Term (months)', 12, 360, step=12)
                credit_history = st.selectbox('📊 Credit History', ('1.0', '0.0'))
                property_area = st.selectbox('🏠 Property Area', ('Urban', 'Semiurban', 'Rural'))
                st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Predict Loan Approval"):
            with st.spinner('Processing your application...'):
                inputs = {
                    'gender': gender, 'married': married, 'dependents': dependents,
                    'education': education, 'self_employed': self_employed,
                    'applicant_income': applicant_income, 'coapplicant_income': coapplicant_income,
                    'loan_amount': loan_amount, 'loan_term': loan_term,
                    'credit_history': credit_history, 'property_area': property_area
                }
                prediction = predict_loan(selected_model, inputs)
                
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                if prediction == 1:
                    st.success("✅ Loan Approved!")
                    st.info("""
                    💡 Tips for Managing Your Loan:
                    - Make timely monthly payments to maintain a good credit history.
                    - Monitor your loan balance and avoid missing installments.
                    - Budget wisely to ensure you don't overextend your finances.
                    - Consider setting up auto-debit to stay consistent with payments.
                    """)
                else:
                    st.error("❌ Loan Denied.")
                    st.warning("""
                    Your loan has been denied due to the current financial situation. 
                    Here are some ways to improve your chances next time:
                    - Improve your credit score: Clear any past dues and make timely debt payments.
                    - Increase your income: Try boosting your or your coapplicant's income.
                    - Reduce the loan amount: Apply for a smaller loan that suits your financial profile.
                    - Reconsider loan term: Opt for a term that makes EMI manageable.
                    """)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Feature Importance Page
elif menu == "📊 Feature Importance":
    st.header("📊 Feature Importance")

    model_choice = st.selectbox("Select Model", ("Random Forest", "Logistic Regression", "XGBoost"))

    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area']

    if model_choice == "Random Forest":
        importances = rf_model.feature_importances_
        feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})

    elif model_choice == "Logistic Regression":
        importances = np.abs(lr_model.coef_[0])
        feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})

    elif model_choice == "XGBoost":
        booster = xgb_model.get_booster()
        score = booster.get_score(importance_type='weight')
        # Ensure all features are represented
        importances = [score.get(f'f{i}', 0) for i in range(len(features))]
        feature_df = pd.DataFrame({'Feature': features, 'Importance': importances})

    feature_df = feature_df.sort_values('Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax)
    st.pyplot(fig)


# Model Insights Page
elif menu == "📈 Model Insights":
    st.header("📈 Model Insights")

    st.subheader("📌 Accuracy (Example)")
    st.write("Random Forest Accuracy: **85.00%** (example)")

    st.subheader("🧮 Confusion Matrix (Example)")
    conf_matrix = np.array([[50, 10], [5, 35]])
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted No", "Predicted Yes"],
                yticklabels=["Actual No", "Actual Yes"])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)


