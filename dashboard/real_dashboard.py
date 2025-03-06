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

    # 🟢 PDF Report Generation Function
    def generate_pdf_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="Healthcare IoT Patient Monitoring Report", ln=True, align='C')
        pdf.ln(10)
        pdf.cell(200, 10, txt="Key Insights and Alerts", ln=True)
        pdf.cell(200, 10, txt="- Predicted High Risk: Immediate attention required!", ln=True)
        pdf.cell(200, 10, txt="- Moderate Risk: Monitor closely.", ln=True)
        pdf.cell(200, 10, txt="- Low Risk: Stable condition.", ln=True)

        # Save and download PDF
        pdf_path = '/content/Healthcare_IoT_Enhanced_Report.pdf'
        pdf.output(pdf_path)
        st.download_button(label="📥 Download PDF Report", data=open(pdf_path, "rb"), file_name="Healthcare_IoT_Enhanced_Report.pdf")

