
import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load("random_forest_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("Predict Using Random Forest 🎯")

# Create inputs
st.header("Enter Details:")
input_dict = {}
# Replace the following with actual columns from your dataset (before encoding)
input_dict["gender"] = st.selectbox("Gender", ["Male", "Female"])
input_dict["course"] = st.selectbox("Course", ["Engineering", "Science", "Arts"])
input_dict["grade"] = st.selectbox("Grade", ["A", "B", "C"])

# Convert to dataframe and one-hot encode
input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

# Add missing columns from training
for col in columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder to match training
input_encoded = input_encoded[columns]

# Predict
if st.button("Predict"):
    prediction = model.predict(input_encoded)
    st.success(f"🎉 Prediction: {prediction[0]}")
