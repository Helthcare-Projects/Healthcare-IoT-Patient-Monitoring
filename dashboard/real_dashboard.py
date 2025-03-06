# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF

st.set_page_config(page_title="Healthcare IoT Enhanced Dashboard", layout="wide")

# Load models
lstm_model = tf.keras.models.load_model('models/lstm_model_enhanced.h5')
rf_model = joblib.load('models/random_forest_model_enhanced.pkl')
xgb_model = joblib.load('models/xgboost_model_enhanced.pkl')

# Load enhanced dataset
data = pd.read_csv('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/data/enhanced_data.csv')

# Patient selection filter
patient_id = st.selectbox('Select Patient ID', data['Patient ID'].unique())
selected_patient = data[data['Patient ID'] == patient_id]
st.write("### Selected Patient Data:")
st.write(selected_patient)

# Predictive analysis using LSTM
vitals = selected_patient[['Temperature', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']].values.reshape(1, -1, 6)
risk_prediction = lstm_model.predict(vitals)
risk_score = round(risk_prediction[0][0] * 100, 2)

st.metric(label='Predicted Health Risk (%)', value=f"{risk_score}%")

# Anomaly detection using Random Forest
anomalies = rf_model.predict(vitals.reshape(-1, 6))
anomaly_count = int(sum(anomalies))
st.metric(label="⚠️ Anomalies Detected", value=anomaly_count)

# Risk classification using XGBoost
features = selected_patient[['Age', 'BMI', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']]
risk_level = xgb_model.predict(features)[0]
st.metric(label="Risk Level", value=["Low Risk", "Medium Risk", "High Risk"][risk_level])

# PDF report generation
if st.button('📄 Generate PDF Report'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Healthcare IoT Enhanced Report", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {['Low', 'Medium', 'High'][risk_level]}", ln=True)
    pdf_path = '/content/Healthcare_IoT_Enhanced_Report.pdf'
    pdf.output(pdf_path)
    st.download_button(label="📥 Download PDF Report", data=open(pdf_path, "rb"), file_name="Healthcare_IoT_Enhanced_Report.pdf")

st.write('Monitor your health in real-time with AI-driven insights!')
