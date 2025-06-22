import streamlit as st
import pandas as pd
import pickle

# Load the scaler and the trained model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('logistic_regression_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Streamlit app
st.title('Titanic Survival Prediction')
age = st.slider('Age', 0, 80, 29)
fare = st.slider('Fare', 0, 500, 50)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
pclass = st.selectbox('Class', [1, 2, 3])
sibsp = st.slider('SibSp', 0, 10, 0)
parch = st.slider('Parch', 0, 10, 0)

# Prepare input data
input_data = pd.DataFrame({
    'Age': [age],
    'Fare': [fare],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0],
    'Pclass_2': [1 if pclass == 2 else 0],
    'Pclass_3': [1 if pclass == 3 else 0],
}, index=[0])

# Ensure all columns match
missing_cols = set(X.columns) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

input_data = input_data[X.columns]  # Ensure the order of columns matches

# Standardize the input data
input_data = scaler.transform(input_data)

# Predict
survival_prob = loaded_model.predict_proba(input_data)[0][1]
st.write(f'The probability of survival is {survival_prob:.2f}')