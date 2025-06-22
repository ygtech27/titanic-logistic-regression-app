import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("Titanic Survival Prediction App")
st.write("Enter the details of the passenger to predict survival.")

# Input fields
Pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
Age = st.slider("Age", 0, 100, 30)
SibSp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
Fare = st.number_input("Passenger Fare", min_value=0.0, value=30.0)
Sex = st.radio("Sex", ["male", "female"])

# Encode Sex as in training
Sex_female = 1 if Sex == "female" else 0
Sex_male = 1 if Sex == "male" else 0

# Create input array in the same order as training
input_data = np.array([[Pclass, Age, SibSp, Parch, Fare, Sex_female, Sex_male]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    st.subheader(f"Prediction: {result}")
