import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('logistic_regression_model.pkl')
    encoded_features = joblib.load('encoded_features.pkl')
except FileNotFoundError:
    st.error("Error loading model files! Ensure 'scaler.pkl', 'logistic_regression_model.pkl', and 'encoded_features.pkl' are available.")
    st.stop()

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Customer Churn Prediction App")
st.markdown("### 📊 Logistic Regression Model with SMOTE-Balanced Data")
st.markdown("---")



def user_input_features():
    st.sidebar.header('Input Customer Data')

    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 24)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 118.0, 50.0)

    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    partner = st.sidebar.selectbox('Partner Status', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))

    contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    data = {'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'InternetService': internet_service,
            'Contract': contract,
            'MonthlyCharges': monthly_charges,
            'PaymentMethod': payment_method,
           }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


st.subheader('User Input Features')
st.write(input_df)


df_processed = pd.get_dummies(input_df)


final_input = pd.DataFrame(0, index=[0], columns=encoded_features)

for col in df_processed.columns:
    if col in final_input.columns:
        final_input[col] = df_processed[col]



scaled_input = scaler.transform(final_input)


if st.button('♻️Predict Churn'):
    with st.spinner('Predicting.🔃..'):
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)


        st.markdown("---")
        st.subheader('🔹Prediction Result')

        churn_status = 'YES (High Risk of Churn⚠️)' if prediction[0] == 1 else 'NO (Customer is likely to Stay✅)'

        if prediction[0] == 1:
            st.error(f"### The Model Predicts: **{churn_status}**")
        else:
            st.success(f"### The Model Predicts: **{churn_status}**")

        st.subheader('🔹Prediction Probability')

        proba_df = pd.DataFrame({
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        }, index=['No Churn Probability', 'Churn Probability'])

        st.bar_chart(proba_df)

        st.markdown(f"**Confidence Level:** Churn Probability is **{prediction_proba[0][1]*100:.2f}%**")
        st.markdown("---")

import streamlit as st
st.subheader("Devloped by BINOY😉!")
st.subheader("Supervised by: Avishek Chowdhury Sir🎓")
# ...
