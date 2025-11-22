# MLL-project

%%writefile churn_app.py
# আপনার Streamlit কোড এখানে থাকবে...
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- ১. মডেল এবং প্রিপ্রসেসিং টুলস লোড করা ---
# ধরে নেওয়া হলো ফাইলগুলো একই ডিরেক্টরিতে আছে
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('logistic_regression_model.pkl')
    # এনকোডিং-এর পর ট্রেনিং ডেটা থেকে প্রাপ্ত কলামগুলির তালিকা
    encoded_features = joblib.load('encoded_features.pkl')
except FileNotFoundError:
    st.error("Error loading model files! Ensure 'scaler.pkl', 'logistic_regression_model.pkl', and 'encoded_features.pkl' are available.")
    st.stop()

# --- ২. অ্যাপ্লিকেশনের কনফিগারেশন ---
st.set_page_config(page_title="Telco Churn Prediction", layout="wide")
st.title("📞 Customer Churn Prediction App")
st.markdown("### 📊 Logistic Regression Model with SMOTE-Balanced Data")
st.markdown("---")


# --- ৩. ইনপুট ফাংশন: ব্যবহারকারীর কাছ থেকে ডেটা নেওয়া ---

def user_input_features():
    st.sidebar.header('Input Customer Data')

    # নিউমেরিক্যাল ফিচার্স (Sliders)
    tenure = st.sidebar.slider('Tenure (Months)', 0, 72, 24)
    monthly_charges = st.sidebar.slider('Monthly Charges ($)', 18.0, 118.0, 50.0)

    # ক্যাটেগরিক্যাল ফিচার্স (Select Boxes)
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    partner = st.sidebar.selectbox('Partner Status', ('Yes', 'No'))
    dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))

    # সার্ভিস ও কন্ট্রাক্ট
    contract = st.sidebar.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    # Final data dictionary
    data = {'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'InternetService': internet_service,
            'Contract': contract,
            'MonthlyCharges': monthly_charges,
            'PaymentMethod': payment_method,
            # অন্যান্য গুরুত্বপূর্ণ কলাম যোগ করুন, যেমন 'SeniorCitizen', 'MultipleLines', ইত্যাদি
           }

    features = pd.DataFrame(data, index=[0])
    return features

# ব্যবহারকারীর ইনপুট সংগ্রহ
input_df = user_input_features()

# --- ৪. ইনপুট প্রিপ্রসেসিং (মডেল ফরম্যাটে আনা) ---

# সমস্ত ইনপুটকে একটি ডেটাফ্রেম হিসেবে দেখানো
st.subheader('User Input Features')
st.write(input_df)

# ক্যাটেগরিক্যাল কলামগুলির One-Hot Encoding করা
# নোট: Telco ডেটাসেটে 'Yes'/'No' ধরনের অনেক কলাম আছে। এখানে শুধু দেখানো ফিচারগুলো এনকোড করা হলো।
# আপনার ট্রেনিং কোডের সাথে কলামের নামগুলি অবশ্যই হুবহু মিলতে হবে।
df_processed = pd.get_dummies(input_df)

# ট্রেনিং-এর সময় ব্যবহৃত সমস্ত কলামগুলি সহ একটি টেমপ্লেট ডেটাফ্রেম তৈরি করা
# এটি নিশ্চিত করে যে ইনপুট ডেটাফ্রেমের কলামের ক্রম এবং সংখ্যা, ট্রেনিং ডেটার সাথে মেলে।
final_input = pd.DataFrame(0, index=[0], columns=encoded_features)

# ব্যবহারকারীর ডেটা টেমপ্লেট ডেটাফ্রেমে পপুলেট করা
for col in df_processed.columns:
    if col in final_input.columns:
        final_input[col] = df_processed[col]


# --- ৫. স্কেলিং ও প্রেডিকশন ---

# শুধুমাত্র নিউমেরিক্যাল কলামগুলি স্কেল করা, যা ট্রেনিং-এর সময় করা হয়েছিল
# এক্ষেত্রে 'tenure' এবং 'MonthlyCharges' স্কেলিং করা হবে।
# **গুরুত্বপূর্ণ**: scaler.feature_names_in_ ব্যবহার করে নিশ্চিত করা যেতে পারে যে স্কেলিং সঠিক কলামগুলিতে ঘটছে।
scaled_input = scaler.transform(final_input)


# প্রেডিকশন
if st.button('♻️Predict Churn'):
    with st.spinner('Predicting.🔃..'):
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # --- ৬. ফলাফল প্রদর্শন ---

        st.markdown("---")
        st.subheader('🔹Prediction Result')

        churn_status = 'YES (High Risk of Churn⚠️)' if prediction[0] == 1 else 'NO (Customer is likely to Stay✅)'

        if prediction[0] == 1:
            st.error(f"### The Model Predicts: **{churn_status}**")
        else:
            st.success(f"### The Model Predicts: **{churn_status}**")

        st.subheader('🔹Prediction Probability')

        # প্রোবাবিলিটি বার চার্ট
        proba_df = pd.DataFrame({
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        }, index=['No Churn Probability', 'Churn Probability'])

        st.bar_chart(proba_df)

        st.markdown(f"**Confidence Level:** Churn Probability is **{prediction_proba[0][1]*100:.2f}%**")
        st.markdown("---")

import streamlit as st
st.title("Devloped by BINOY😉!")
# ...
