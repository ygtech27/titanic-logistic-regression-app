import streamlit as st
import pandas as pd
import numpy as np
import joblib  # or use pickle
from sklearn.preprocessing import LabelEncoder

# Load trained logistic regression model
model = joblib.load("logistic_model.pkl")  # Make sure this matches your saved model file

# Title
st.title("🚢 Titanic Survival Prediction")
st.write("Enter the passenger details below to predict the survival probability.")

# Sidebar inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", 0.0, 600.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

# Map categorical variables if needed (same way as during training)
sex_encoded = 1 if sex == "female" else 0
embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_map[embarked]

# Create feature vector
features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict button
if st.button("Predict"):
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")
    st.write(f"**Survived:** {'Yes ✅' if prediction[0] == 1 else 'No ❌'}")
    st.write(f"**Probability of Survival:** {probability:.2%}")
