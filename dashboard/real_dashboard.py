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
st.set_page_config(page_title="Healthcare IoT Real-Time Dashboard", layout="wide")

# 🟢 Initialize global variables
anomalies = 0
cumulative_anomalies = 0  # Cumulative count of anomalies

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies, cumulative_anomalies  # Use global keywords

    st.title('🏥 Healthcare IoT Real-Time Dashboard')

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

    # Load data
    try:
        data = pd.read_csv('data/cleaned_data.csv')
        st.success("✅ Data loaded successfully.")
        st.write(data.head())  # Print first few rows for verification
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Normalize data
    try:
        scaler = MinMaxScaler()
        data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']] = scaler.fit_transform(
            data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
        st.success("✅ Data normalized successfully.")
    except Exception as e:
        st.error("❌ Error normalizing data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Improved Visualization: Display Separate Graphs for Each Vital
    st.subheader('📊 Real-Time Vital Signs Analysis')
    vital_signs = ['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']
    for vital in vital_signs:
        st.line_chart(data[[vital]])

    # 🟢 Feature Importance Analysis for Random Forest
    st.subheader("🔍 Feature Importances for Anomaly Detection")
    try:
        feature_importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': vital_signs,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.error("❌ Error displaying feature importances.")
        st.text(traceback.format_exc())

    # 🟢 Anomaly Detection with Cumulative Count
    st.subheader('🚨 Anomaly Detection')
    try:
        # Adjust threshold for anomaly detection
        threshold = 0.95  # Increased threshold to reduce false positives
        preds_proba = rf_model.predict_proba(data[vital_signs])[:, 1]
        preds = (preds_proba > threshold).astype(int)
        anomalies = int(sum(preds))
        cumulative_anomalies += anomalies  # Update cumulative count

        # 🟢 Create a time-series chart for anomalies
        anomaly_series = pd.Series(preds_proba, name='Anomaly Probability').rolling(window=10).mean()
        st.line_chart(anomaly_series[:50])  # Show first 50 points with a rolling average

        st.metric(label="⚠️ Anomalies Detected", value=anomalies)
        st.metric(label="📈 Cumulative Anomalies Detected", value=cumulative_anomalies)
        st.progress(min(anomalies / len(data), 1.0))

        # 🟢 Enhanced feedback based on anomaly count
        if anomalies > 3000:
            st.error("🚨 Extremely high anomalies detected! Immediate investigation required.")
        elif anomalies > 1000:
            st.warning("⚠ High anomalies detected. Review data, model thresholds, and feature importances.")
        elif anomalies > 500:
            st.warning("⚠ Moderate anomalies detected. Review data and model thresholds.")
        else:
            st.info("✅ Anomaly detection is within normal range.")
    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())

    # 🟢 Generate PDF Report
    if st.button('📄 Generate PDF Report'):
        generate_pdf_report(importance_df, anomalies, cumulative_anomalies)
        st.success("✅ PDF Report generated successfully!")

    st.write('Monitor your health in real-time with AI-driven insights!')

# 🟢 Enhanced PDF Report Generation Function
import tempfile  # Import tempfile to handle temporary file paths

def generate_pdf_report(importance_df, anomalies, cumulative_anomalies):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Healthcare IoT Patient Monitoring Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Anomalies Detected: {anomalies}", ln=True)
    pdf.cell(200, 10, txt=f"Cumulative Anomalies Detected: {cumulative_anomalies}", ln=True)
    pdf.ln(10)

    # Feature importances
    pdf.cell(200, 10, txt="Feature Importances:", ln=True)
    for index, row in importance_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Feature']}: {row['Importance']:.2f}", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Key Observations:", ln=True)
    if anomalies > 3000:
        pdf.cell(200, 10, txt="Extremely high anomalies detected! Immediate investigation required.", ln=True)
    elif anomalies > 1000:
        pdf.cell(200, 10, txt="High anomalies detected. Review data, model thresholds, and feature importances.", ln=True)
    elif anomalies > 500:
        pdf.cell(200, 10, txt="Moderate anomalies detected. Review data and model thresholds.", ln=True)
    else:
        pdf.cell(200, 10, txt="Anomaly detection is within normal range.", ln=True)

    # 🟢 Embed Feature Importance Bar Chart in PDF using a Temporary File
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        chart_path = tmpfile.name
        importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, title='Feature Importances')
        plt.tight_layout()
        plt.savefig(chart_path)  # Save chart to temporary file
        plt.close()  # Close the plot to free resources
        pdf.image(chart_path, x=10, y=100, w=190)  # Embed chart in PDF

    # 🟢 Save PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        # Provide download button for the generated PDF
        st.download_button(label="📥 Download Enhanced PDF Report", data=open(tmp_pdf.name, "rb"), file_name="Healthcare_IoT_Report_Enhanced.pdf")


    # 🟢 Embed Feature Importance Bar Chart in PDF
    importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False, title='Feature Importances')
    plt.tight_layout()
    plt.savefig('/content/feature_importance_chart.png')
    pdf.image('/content/feature_importance_chart.png', x=10, y=100, w=190)

    # Save PDF
    pdf.output("/content/Healthcare_IoT_Report_Enhanced.pdf")
    st.download_button(label="📥 Download Enhanced PDF Report", data=open("/content/Healthcare_IoT_Report_Enhanced.pdf", "rb"), file_name="Healthcare_IoT_Report_Enhanced.pdf")


    # Feature importances
    pdf.cell(200, 10, txt="Feature Importances:", ln=True)
    for index, row in importance_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Feature']}: {row['Importance']:.2f}", ln=True)

    # Save PDF
    pdf.output("/content/Healthcare_IoT_Report.pdf")
    st.download_button(label="📥 Download PDF Report", data=open("/content/Healthcare_IoT_Report.pdf", "rb"), file_name="Healthcare_IoT_Report.pdf")

# 🟢 Run the main function
if __name__ == "__main__":
    main()
