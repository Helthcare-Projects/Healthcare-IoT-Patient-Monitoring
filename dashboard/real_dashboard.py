# Install missing libraries if not already installed
import os
os.system('pip install matplotlib fpdf')  # Ensure matplotlib and fpdf are installed

# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt  # Ensure matplotlib is imported successfully
from sklearn.preprocessing import MinMaxScaler  # Ensure normalization
from fpdf import FPDF  # For PDF report generation

# 🟢 Move this line immediately after imports
st.set_page_config(page_title="Healthcare IoT Enhanced Dashboard", layout="wide")

# 🟢 Initialize global variables
anomalies = 0
cumulative_anomalies = 0  # Cumulative count of anomalies

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies, cumulative_anomalies  # Use global keywords

    st.title('🏥 Healthcare IoT Enhanced Dashboard')

    # Load models
    try:
        lstm_model = tf.keras.models.load_model(
            'models/lstm_model.h5',
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        rf_model = joblib.load('models/random_forest_model.pkl')
        st.success("✅ Models loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.write("Please ensure the model files are correctly uploaded.")
        return
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())
        return

    # 🟢 Load enhanced dataset with new features
    data = pd.read_csv('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/data/enhanced_data.csv')
    st.success("✅ Enhanced dataset loaded successfully!")
    st.write(data.head())

    # 🟢 Add Patient Selection Filter
    patient_id = st.selectbox('Select Patient ID', data['Patient ID'].unique())
    selected_patient_data = data[data['Patient ID'] == patient_id]
    st.write("### Selected Patient Data:")
    st.write(selected_patient_data)

    # 🟢 Show Detailed Vitals for Selected Patient
    st.line_chart(selected_patient_data[['Heart Rate', 'BPSYS', 'BPDIA', 'Oxygen Saturation', 'Temperature']])

    # 🟢 Predict Health Risk for Selected Patient
    if st.button('🔮 Predict Health Risk for Selected Patient'):
        vitals = selected_patient_data[['Temperature', 'Heart Rate', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation']].values.reshape(1, -1, 6)
        risk_prediction = lstm_model.predict(vitals)
        risk_score = round(risk_prediction[0][0] * 100, 2)
        st.metric(label='Predicted Health Risk (%)', value=f"{risk_score}%")
        if risk_score > 75:
            st.error('🚨 High risk detected! Immediate attention required.')
        elif risk_score > 50:
            st.warning('⚠ Moderate risk detected. Monitor closely.')
        else:
            st.success('✅ Low risk detected.')
