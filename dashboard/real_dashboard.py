# Install missing libraries if not already installed
import os
os.system('pip install streamlit fpdf xgboost tensorflow joblib pandas')

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
lstm_model = tf.keras.models.load_model('models/lstm_model_enhanced.keras')
rf_model = joblib.load('models/random_forest_model_enhanced.pkl')
xgb_model = joblib.load('models/xgboost_model_enhanced.pkl')

# Load enhanced dataset
data = pd.read_csv('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/data/enhanced_data_realistic.csv')

# 🟢 Patient Selection Filter
st.sidebar.header("Select Patient")
patient_id = st.sidebar.selectbox('Patient ID', data['Patient ID'].unique())
selected_patient = data[data['Patient ID'] == patient_id]

# Display selected patient details
st.sidebar.write("### Selected Patient Details")
st.sidebar.write(selected_patient[['Age', 'Gender', 'Risk_Level', 'Scenario', 'Condition']])

# 🟢 Real-Time Vitals Monitoring
st.subheader(f"📊 Real-Time Vitals for Patient ID: {patient_id}")
st.line_chart(selected_patient[['Temperature', 'Heart Rate', 'BPSYS', 'BPDIA', 'Oxygen Saturation']].tail(50))

# 🟢 Anomaly Detection Insights
st.subheader('🚨 Detailed Anomaly Insights')
anomalies = rf_model.predict(selected_patient[['Temperature', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']])
anomaly_count = int(sum(anomalies))
st.metric(label="⚠️ Total Anomalies Detected", value=anomaly_count)

# 🟢 Breakdown of Anomalies by Type
st.write("### Anomalies Breakdown")
for feature in ['BPSYS', 'BPDIA', 'Oxygen Saturation', 'Heart Rate']:
    count = int(sum(selected_patient[feature] > selected_patient[feature].mean()))
    st.write(f"🔹 {feature}: {count} anomalies")

# 🟢 Predictive Analysis with LSTM
vitals = selected_patient[['Temperature', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']].values.reshape(1, -1, 6)
risk_prediction = lstm_model.predict(vitals)
risk_score = round(risk_prediction[0][0] * 100, 2)
st.metric(label='Predicted Health Risk (%)', value=f"{risk_score}%")

# 🟢 Real-Time Alerts for High-Risk Situations
if risk_score > 75:
    st.error('🚨 High risk detected! Immediate attention required.')
    st.warning('⚠ Predicted cause: Potential cardiac arrest due to hypoxia and hypertension.')
elif risk_score > 50:
    st.warning('⚠ Moderate risk detected. Monitor closely.')
else:
    st.success('✅ Low risk detected.')

# 🟢 Risk Classification with XGBoost
features = selected_patient[['Age', 'BMI', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']]
risk_level = xgb_model.predict(features)[0]
risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
st.metric(label="Risk Level", value=risk_map[risk_level])

# 🟢 Enhanced PDF Report Generation
def generate_pdf_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Healthcare IoT Enhanced Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {selected_patient['Age'].values[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Gender: {selected_patient['Gender'].values[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Level: {risk_map[risk_level]}", ln=True)
    pdf.cell(200, 10, txt=f"Total Anomalies Detected: {anomaly_count}", ln=True)
    pdf.cell(200, 10, txt=f"Predicted Health Risk: {risk_score}%", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Recommended Actions:", ln=True)
    if risk_score > 75:
        pdf.cell(200, 10, txt="Immediate attention required: Administer oxygen and antihypertensives.", ln=True)
    elif risk_score > 50:
        pdf.cell(200, 10, txt="Close monitoring required: Schedule cardiology consult.", ln=True)
    else:
        pdf.cell(200, 10, txt="Routine monitoring is sufficient.", ln=True)
    
    # Save PDF
    pdf_path = '/content/Healthcare_IoT_Enhanced_Report.pdf'
    pdf.output(pdf_path)
    st.download_button(label="📥 Download PDF Report", data=open(pdf_path, "rb"), file_name="Healthcare_IoT_Enhanced_Report.pdf")

# 🟢 Generate PDF Button
if st.button('📄 Generate Enhanced PDF Report'):
    generate_pdf_report()
    st.success("✅ PDF Report generated successfully!")

st.write('Monitor your health in real-time with AI-driven insights!')
