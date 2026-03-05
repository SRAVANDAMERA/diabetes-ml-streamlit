import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction System")
st.write("AI-powered health risk analysis based on patient medical indicators.")

# Sidebar input
st.sidebar.header("Patient Health Data")

pregnancies = st.sidebar.slider("Pregnancies",0,20,1)
glucose = st.sidebar.slider("Glucose Level",0,200,120)
blood_pressure = st.sidebar.slider("Blood Pressure",0,150,70)
skin_thickness = st.sidebar.slider("Skin Thickness",0,100,20)
insulin = st.sidebar.slider("Insulin Level",0,900,80)
bmi = st.sidebar.slider("BMI",0.0,70.0,25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function",0.0,3.0,0.5)
age = st.sidebar.slider("Age",1,120,30)

input_data = np.array([
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]).reshape(1,-1)

scaled_data = scaler.transform(input_data)

col1, col2 = st.columns(2)

# Prediction section
if st.sidebar.button("Run Prediction"):

    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)

    risk = probability[0][1] * 100

    with col1:

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("High Risk of Diabetes")
        else:
            st.success("Low Risk of Diabetes")

        st.metric("Diabetes Probability", f"{risk:.2f}%")

    # Probability chart
    with col2:

        st.subheader("Prediction Probability")

        labels = ["Non-Diabetic","Diabetic"]
        values = probability[0]

        fig, ax = plt.subplots()

        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

    # Health insights
    st.subheader("Health Insights")

    insights = []

    if glucose > 140:
        insights.append("High glucose level detected. Consider reducing sugar intake.")

    if bmi > 30:
        insights.append("BMI indicates obesity risk. Regular exercise is recommended.")

    if blood_pressure > 90:
        insights.append("Blood pressure is elevated. Monitor cardiovascular health.")

    if age > 45:
        insights.append("Age group has higher diabetes risk. Regular screening advised.")

    if len(insights) == 0:
        insights.append("Health indicators appear within normal ranges.")

    for i in insights:
        st.write("•", i)

    # Feature visualization
    st.subheader("Patient Feature Overview")

    feature_data = pd.DataFrame({
        "Feature":[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age"
        ],
        "Value":[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    })

    st.bar_chart(feature_data.set_index("Feature"))