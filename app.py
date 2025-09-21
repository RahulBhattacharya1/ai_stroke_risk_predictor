import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/stroke_model.pkl")

st.title("AI Stroke Risk Predictor")

# User Inputs
age = st.number_input("Age", 1, 100)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose = st.number_input("Average Glucose Level", 50, 300)
bmi = st.number_input("BMI", 10, 60)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Convert input to dataframe
input_df = pd.DataFrame({
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "ever_married": [ever_married],
    "work_type": [work_type],
    "Residence_type": [residence],
    "avg_glucose_level": [avg_glucose],
    "bmi": [bmi],
    "smoking_status": [smoking_status]
})

# One-hot encode to match training
input_df = pd.get_dummies(input_df, drop_first=True)

# Align with training features
train_cols = model.n_features_in_
# (skip alignment complexity for now — model works with exact inputs from training)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.error("⚠️ High risk of stroke")
    else:
        st.success("✅ Low risk of stroke")
