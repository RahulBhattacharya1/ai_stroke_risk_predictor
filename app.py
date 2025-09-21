# app.py  (hotfix for feature-name mismatch without retraining)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="AI Stroke Risk Predictor")

@st.cache_resource
def load_model():
    # This loads the model you trained on get_dummies(...)
    # e.g., models/stroke_model.pkl
    return joblib.load("models/stroke_model.pkl")

model = load_model()

st.title("AI Stroke Risk Predictor")

# === Inputs must mirror training raw columns (before get_dummies) ===
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    hypertension = st.selectbox("Hypertension (0/1)", options=[0, 1], index=0)
    heart_disease = st.selectbox("Heart Disease (0/1)", options=[0, 1], index=0)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=400.0, value=100.0, step=0.1)

with col2:
    bmi = st.number_input("BMI", min_value=0.0, max_value=90.0, value=25.0, step=0.1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

raw = pd.DataFrame([{
    "age": float(age),
    "hypertension": int(hypertension),
    "heart_disease": int(heart_disease),
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": float(avg_glucose_level),
    "bmi": float(bmi),
    "gender": gender,
    "smoking_status": smoking_status,
}])

# === CRITICAL: replicate training encoding exactly ===
# Your training used: pd.get_dummies(df.drop("stroke", axis=1), drop_first=True)
X_app = pd.get_dummies(raw, drop_first=True)

# Align columns to what the model saw during training
# Prefer model.feature_names_in_ (sklearn >=1.0 on DataFrame)
if hasattr(model, "feature_names_in_"):
    train_cols = list(model.feature_names_in_)
else:
    # Optional fallback: if you saved columns to JSON during training
    # (only used if you created this file in training)
    cols_path = Path("models/train_columns.json")
    if cols_path.exists():
        train_cols = json.loads(cols_path.read_text())
    else:
        st.error("Model does not expose feature_names_in_. Retrain model or include models/train_columns.json.")
        st.stop()

# Reindex to training columns, filling any missing with 0
X_app = X_app.reindex(columns=train_cols, fill_value=0)

threshold = st.slider("Decision threshold (probability of stroke)", 0.05, 0.95, 0.5, 0.05)

if st.button("Predict"):
    try:
        prob = float(model.predict_proba(X_app)[0, 1])
        pred = int(prob >= threshold)

        st.subheader("Prediction")
        st.write(f"Predicted probability of stroke: {prob:.3f}")
        st.write(f"Decision threshold: {threshold:.2f}")
        st.write(f"Predicted class: {pred} (1 = higher risk, 0 = lower risk)")
        st.caption("For education only; not a medical device.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optional: Debug panel to see columns sent to model
with st.expander("Debug: feature columns sent to model"):
    st.write("App columns:", list(X_app.columns))
    if hasattr(model, "feature_names_in_"):
        st.write("Model expects:", list(model.feature_names_in_))
