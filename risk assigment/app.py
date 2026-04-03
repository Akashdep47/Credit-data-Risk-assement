import streamlit as st
import pandas as pd
import joblib

# Load the trained XGBoost model
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.joblib")

# Load the target label encoder
@st.cache_resource
def load_target_encoder():
    return joblib.load("target_encoder.joblib")

# Load feature label encoders
@@st.cache_resource
def load_feature_encoders():
    feature_encoders = {}

    file_mapping = {
        'Sex': 'label_encoder_Sex.joblib',
        'Housing': 'label_encoder_Housing.joblib',
        'Saving accounts': 'label_encoder_saving_accounts.joblib',
        'Checking account': 'label_encoder_checking_account.joblib'
    }

    for col, file_name in file_mapping.items():
        feature_encoders[col] = joblib.load(file_name)

    return feature_encoders

model = load_model()
target_encoder = load_target_encoder()
feature_encoders = load_feature_encoders()

st.title("Credit Risk Prediction App")
st.write("Enter the customer's details to predict their credit risk (Good/Bad).")

# Input fields for features
with st.sidebar:
    st.header("Customer Details")
    age = st.slider("Age",min_value =18,max_value=80,value=30)
    sex = st.selectbox("Sex", ['female', 'male'])
    job = st.slider("Job (0-3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", ['own', 'free', 'rent'])
    saving_accounts = st.selectbox("Saving accounts", ['little', 'moderate', 'quite rich', 'rich'])
    checking_account = st.selectbox("Checking account", ['moderate', 'little', 'rich'])
    credit_amount = st.number_input("Credit amount",min_value=0,value=1000)
    duration = st.slider("Duration (in months)",min_value=1,value=12)

# Create a dictionary from inputs
input_data = {
    "Age": [age],
    "Sex": [feature_encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [feature_encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [feature_encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [feature_encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# Preprocess input data using loaded encoders (redundant now, as data is encoded directly into dict)
# Removed the loop that re-encoded, as it's now handled directly in input_data creation.

# Make prediction
if st.button("Predict Risk"):
    prediction_encoded = model.predict(input_df)
    prediction_label = target_encoder.inverse_transform(prediction_encoded)

    if prediction_label[0] == 'good':
        st.success(f"The predicted credit risk is: **{prediction_label[0].upper()}**")
    else:
        st.warning(f"The predicted credit risk is: **{prediction_label[0].upper()}**")

st.markdown("""
## How to run this app:
1. Save the code above as `app.py` in your Colab environment.
2. In a new cell, run `!pip install streamlit`.
3. In another new cell, run `!streamlit run app.py & npx localtunnel --port 8501`.
4. Click on the public URL provided by localtunnel to view your app.
""")
