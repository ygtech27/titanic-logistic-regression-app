# ---------------------------------------------------------
# app.py – Streamlit Titanic Survival Prediction
# ---------------------------------------------------------
"""
This Streamlit application loads a **single, self‑contained scikit‑learn Pipeline**
( preprocessing ➜ logistic‑regression model ) saved as `titanic_lr_pipeline.pkl` and
lets a user enter basic passenger details to predict the probability of survival.

🔧 **How to prepare the model file**
------------------------------------------------
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd, joblib

# X = training features (raw), y = target
cat_cols = ["Pclass", "Sex", "Embarked"]
num_cols = ["Age", "SibSp", "Parch", "Fare"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("clf",  LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)
joblib.dump(pipe, "titanic_lr_pipeline.pkl")
```
Place the resulting `titanic_lr_pipeline.pkl` next to **this** `app.py` (or add the
correct relative path below).

📝 **requirements.txt** should *pin* the same scikit‑learn version you trained
with to avoid the InconsistentVersionWarning, e.g.
```
streamlit==1.46.0
scikit-learn==1.6.1   # ↔️ version used during training
pandas
numpy
joblib
```
"""

# Std‑lib / third‑party imports
import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1️⃣  Helper ‑ load the pipeline once and cache it
# ---------------------------------------------------------
@st.cache_resource  # ↳ persists across reruns; cleared on file change
def load_pipeline(path: str = "logistic_model.pkl"):
    """Load and return the pre‑trained sklearn Pipeline."""
    return joblib.load(path)

pipe = load_pipeline()

# ---------------------------------------------------------
# 2️⃣  Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger details to estimate survival probability based on a logistic‑regression model trained on the Titanic dataset.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass   = st.selectbox("Ticket Class (Pclass)", options=[1, 2, 3], index=0)
        age      = st.slider("Age",         min_value=0,  max_value=80, value=25)
        sibsp    = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)

    with col2:
        sex      = st.selectbox("Sex",      options=["male", "female"], index=0)
        parch    = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=6, value=0)
        fare     = st.number_input("Fare",   min_value=0.0, max_value=600.0, value=32.0, step=0.1)

    embarked = st.selectbox("Port of Embarkation (Embarked)", options=["S", "C", "Q"], index=0)

    submitted = st.form_submit_button("Predict")

# ---------------------------------------------------------
# 3️⃣  Prediction logic
# ---------------------------------------------------------
if submitted:
    # Assemble raw‑feature DataFrame with *exact* column names expected by the pipeline
    input_df = pd.DataFrame({
        "Pclass"  : [pclass],
        "Sex"     : [sex],
        "Age"     : [age],
        "SibSp"   : [sibsp],
        "Parch"   : [parch],
        "Fare"    : [fare],
        "Embarked": [embarked]
    })

    # Predict using the full pipeline
    pred_class = pipe.predict(input_df)[0]
    pred_prob  = pipe.predict_proba(input_df)[0][1]  # probability of survival (class 1)

    st.markdown("---")
    st.subheader("🧾 Prediction Result")

    survived_text = "✅ Survived" if pred_class == 1 else "❌ Did Not Survive"
    st.write(f"**Outcome:** {survived_text}")
    st.write(f"**Estimated Probability of Survival:** {pred_prob:.2%}")

    # Optional: display input dataframe for transparency
    with st.expander("Show model input"):
        st.dataframe(input_df)
