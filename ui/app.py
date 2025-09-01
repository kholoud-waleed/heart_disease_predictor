import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from feature_selectors import RFEChi2Union


# Load the best saved model
model = joblib.load("models\svm_model.pkl")

# Streamlit UI
st.title("Welcome to the Heart Disease Prediction App 🩺❤️")
st.markdown("""Enter your health data below to predict the risk of heart disease.""")

# Example features
age = st.number_input("Age in years:", min_value=1, max_value=120, value=50)
sex = st.selectbox("Gender (0 = Female, 1 = Male):", options=[0, 1])  # 0: female, 1: male
cp = st.selectbox("Chest Pain Type:", options=[1, 2, 3, 4]) # 1: typical, 2: atypical, 3: non-anginal, 4: asymptomatic
trestbps = st.number_input("Resting Blood Pressure in (mm.Hg) on admission to the hospital:", value=120)
chol = st.number_input("Serum Cholesterol in (mg/dl):", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120mg/dl (0 = False; 1 = True):", options=[0, 1])  # 1 = true; 0 = false
restecg = st.selectbox("Resting ECG Results:", options=[0, 1, 2])  # 0: normal, 1: ST-T wave abnormality, 2: probable or definite left ventricular hypertrophy
thalach = st.number_input("Max Heart Rate Achieved:", value=100)
exang = st.selectbox("Exercise Induced Angina (0 = False; 1 = True):", options=[0, 1])  # 1 = yes; 0 = no
oldpeak = st.number_input("ST depression induced by exercise relative to rest:", value=0.0, step=0.1)
slope = st.selectbox("Slope of the peak exercise ST Segment:", options=[1, 2, 3])  # 1: upsloping, 2: flat, 3: downsloping
ca = st.number_input("Number of major vessels (0-3) coloured by flourosopy:", min_value=0, max_value=3, value=0)
thal = st.selectbox("Thalassemia:", options=[3, 6, 7])   # 3 = normal; 6 = fixed defect; 7 = reversable defect

# Collect inputs into a DataFrame
user_data = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
                          'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
                          'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]})

# Prediction
if st.button("Predict🤖"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1] if hasattr(model, "predict_proba") else None
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease! Probability: {probability:.2f}" if probability is not None else "⚠️ High risk of heart disease!")
    else:
        st.success(f"✅ Low risk of heart disease. Probability: {probability:.2f}" if probability is not None else "✅ Low risk of heart disease.")

# Data Visualization Example
st.subheader("📈Explore Heart Disease Trends📉")
# Load dataset for visualization
try:
    data = pd.read_csv("data/heart_diseases.csv")  # safer with forward slash
    st.write("### 🔎Dataset Sample")
    st.dataframe(data.head(20))

    # Convert target column to binary (0 = No Disease, 1 = Disease)
    data['num'] = data['num'].apply(lambda x: 1 if x > 0 else 0)

    # Example: Heart disease count
    st.write("### 📊Heart Disease Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='num', data=data, ax=ax)
    ax.set_xticklabels(["No Disease", "Disease"])
    # Annotate bars with counts
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom')
    st.pyplot(fig)

    # Example: Correlation heatmap
    st.write("### Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

except FileNotFoundError:
    st.warning("Dataset CSV for visualization not found. Please provide your heart disease CSV file named 'heart_diseases.csv'.")
