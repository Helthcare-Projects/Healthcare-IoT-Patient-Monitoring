# Install missing libraries if not already installed
import os
os.system('pip install matplotlib fpdf xgboost')

# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
import xgboost as xgb

# 🟢 Set page config
st.set_page_config(page_title="Healthcare IoT Real-Time Dashboard", layout="wide")

# 🟢 Initialize global variables
anomalies = 0
cumulative_anomalies = 0

# 🟢 Define main function
def main():
    global anomalies, cumulative_anomalies

    # Page title
    st.title('🏥 Healthcare IoT Real-Time Dashboard')

    # Load models
    try:
        lstm_model = tf.keras.models.load_model(
            'models/lstm_model_enhanced.h5',
            custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
        )
        rf_model = joblib.load('models/random_forest_model_enhanced.pkl')
        xgb_model = joblib.load('models/xgboost_model_enhanced.pkl')
        scaler = joblib.load('models/scaler_enhanced.pkl')
        st.success("✅ Models loaded successfully.")
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())
        return

    # Load data
    try:
        data = pd.read_csv('data/enhanced_data_realistic_with_id.csv')
        st.success("✅ Data loaded successfully.")
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Patient Selection
    patient_ids = data['Patient_ID'].unique()
    selected_patient = st.selectbox("Select Patient ID:", patient_ids)
    patient_data = data[data['Patient_ID'] == selected_patient]
    st.write(f"🔍 Viewing data for Patient ID: {selected_patient}")

    # 🟢 Normalize and Scale Data
    try:
        # Load feature names used during training
        expected_features = scaler.feature_names_in_

        # Align current features with the expected features used during training
        features = patient_data.drop(['Patient_ID', 'Risk_Level'], axis=1)
        features = features[expected_features]  # Ensure column order and count match

        # Scale the aligned features
        scaled_features = scaler.transform(features)
        st.success("✅ Data normalized and scaled successfully.")
    except KeyError as e:
        st.error(f"❌ Missing expected columns: {str(e)}")
        st.text(traceback.format_exc())
        return
    except Exception as e:
        st.error("❌ Error during data normalization and scaling.")
        st.text(traceback.format_exc())
        return

    # 🟢 Real-Time Summary Panel
    st.subheader('📊 Real-Time Summary Panel')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Risk Level", value=patient_data['Risk_Level'].values[0])
    with col2:
        st.metric(label="Alert Status", value=patient_data['Alert_Status'].values[0])
    with col3:
        st.metric(label="Device Connectivity", value=patient_data['Device_Connectivity'].values[0])

    # 🟢 Feature Importance Visualization
    st.subheader("🔍 Feature Importance for Risk Prediction")
    try:
        xgb_importance = xgb_model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame({
            'Feature': list(xgb_importance.keys()),
            'Importance': list(xgb_importance.values())
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except Exception as e:
        st.error("❌ Error displaying feature importance.")
        st.text(traceback.format_exc())

    # 🟢 Anomaly Detection
    st.subheader('🚨 Anomaly Detection')
    try:
        preds_proba = rf_model.predict_proba(scaled_features)[:, 1]
        preds = (preds_proba > 0.95).astype(int)
        anomalies = int(sum(preds))
        cumulative_anomalies += anomalies
        st.metric(label="Anomalies Detected", value=anomalies)
        st.metric(label="Cumulative Anomalies", value=cumulative_anomalies)

        # 🟢 Anomaly Explanation
        for i, pred in enumerate(preds):
            if pred:
                st.warning(f"⚠ Anomaly detected at row {i + 1} due to {features.columns[np.argmax(scaled_features[i])]}")

    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())

    # 🟢 Predictive Alerts and Recommendations
    st.subheader('🔮 Predictive Analysis and Recommendations')
    try:
        risk_prediction = xgb_model.predict(scaled_features)
        confidence = np.max(xgb_model.predict_proba(scaled_features)) * 100
        st.metric(label="Risk Prediction", value=risk_prediction[0])
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    except Exception as e:
        st.error("❌ Error during predictive analysis.")
        st.text(traceback.format_exc())

    # 🟢 Real-Time Dynamic Graphs
    st.subheader('📈 Real-Time Vital Signs')
    vitals = ['Heart Rate', 'BPSYS', 'BPDIA', 'Oxygen Saturation', 'Temperature']
    for vital in vitals:
        st.line_chart(patient_data[[vital]])

    # 🟢 Generate Enhanced PDF Report
    if st.button('📄 Generate PDF Report'):
        generate_pdf_report(importance_df, anomalies, cumulative_anomalies, risk_prediction[0], confidence)
        st.success("✅ PDF Report generated successfully!")

# 🟢 PDF Report Generation Function
def generate_pdf_report(importance_df, anomalies, cumulative_anomalies, risk_prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Healthcare IoT Monitoring Report", ln=True, align='C')

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Anomalies: {anomalies}", ln=True)
    pdf.cell(200, 10, txt=f"Cumulative Anomalies: {cumulative_anomalies}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Prediction: {risk_prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence:.2f}%", ln=True)

    pdf.ln(10)
    pdf.cell(200, 10, txt="Feature Importances:", ln=True)
    for index, row in importance_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Feature']}: {row['Importance']:.2f}", ln=True)

    pdf.output('/content/Healthcare_IoT_Report_Enhanced.pdf')
    st.download_button(label="📥 Download PDF Report", data=open('/content/Healthcare_IoT_Report_Enhanced.pdf', 'rb'), file_name="Healthcare_IoT_Report_Enhanced.pdf")

# 🟢 Run main function
if __name__ == "__main__":
    main()
