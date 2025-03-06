# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
import zipfile
import os

# 🟢 Move this line immediately after imports
st.set_page_config(page_title="Healthcare IoT Patient Monitoring", layout="wide")

# 🟢 Initialize anomalies with a safe default at the very top
anomalies = 0

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies  # Use global keyword to ensure anomalies is accessible

    st.title('🏥 Healthcare IoT Patient Monitoring Dashboard')
    
    # 🟢 Extract and load LSTM model from .keras zip file
    try:
        # Create a temporary directory for extraction
        extracted_model_dir = 'models/extracted_lstm_model'
        
        # Check if directory exists, if not, extract the .keras zip file
        if not os.path.exists(extracted_model_dir):
            with zipfile.ZipFile('models/lstm_model.keras', 'r') as zip_ref:
                zip_ref.extractall(extracted_model_dir)
        
        # Load the extracted model
        lstm_model = tf.keras.models.load_model(extracted_model_dir)
        rf_model = joblib.load('models/random_forest_model.pkl')
        st.success("✅ Models loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.write("Please ensure the model files are correctly uploaded.")
        return  # Exit if models cannot be loaded
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())  # Print detailed traceback
        return  # Exit if models cannot be loaded

    # Load data
    try:
        data = pd.read_csv('data/cleaned_data.csv')
        st.success("✅ Data loaded successfully.")
        st.write(data.head())  # Print first few rows for verification
    except FileNotFoundError:
        st.error("❌ Data file not found.")
        return  # Exit if data cannot be loaded
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())  # Print detailed traceback
        return  # Exit if data cannot be loaded

    # 🟢 Display live charts for vital signs
    st.subheader('📊 Real-Time Vital Signs')
    try:
        st.line_chart(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
    except Exception as e:
        st.error("❌ Error displaying live charts.")
        st.text(traceback.format_exc())  # Print detailed traceback

    # 🟢 Anomaly detection using Random Forest
    st.subheader('🚨 Anomaly Detection')
    try:
        preds = rf_model.predict(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
        if isinstance(preds, np.ndarray):
            anomalies = int(sum(preds))
        st.metric(label="⚠️ Anomalies Detected", value=anomalies)
        st.progress(anomalies / len(data) * 100)

        # 🟢 Additional warnings if anomalies are high
        if anomalies > 500:
            st.warning("⚠️ Unusually high anomalies detected. Please check data quality or model thresholds.")
        elif anomalies > 300:
            st.warning("⚠️ Moderate anomalies detected. Consider reviewing input data.")
        else:
            st.info("✅ Anomaly detection is within normal range.")
    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())  # Print detailed traceback

    # 🟢 Live predictions with LSTM
    st.subheader('🔮 LSTM Predictions for Health Deterioration')
    try:
        lstm_input = data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']].values.reshape(-1, 1, 8)
        lstm_preds = lstm_model.predict(lstm_input)
        lstm_preds = np.clip(lstm_preds, 0, 1)  # 🟢 Ensure predictions are within [0, 1]
        if isinstance(lstm_preds, np.ndarray):
            st.line_chart(lstm_preds[:50])  # Show first 50 predictions
        else:
            st.write("No valid predictions from LSTM model.")
    except Exception as e:
        st.error("❌ Error during LSTM prediction.")
        st.text(traceback.format_exc())  # Print detailed traceback

    st.write('Monitor your health in real-time with AI-driven insights!')

# 🟢 Run the main function
if __name__ == "__main__":
    main()
