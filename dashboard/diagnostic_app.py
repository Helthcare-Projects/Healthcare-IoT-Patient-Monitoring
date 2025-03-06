# Import necessary libraries
import streamlit as st
# 🟢 Move this line immediately after imports
st.set_page_config(page_title="Healthcare IoT Patient Monitoring", layout="wide")

import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback


# Rest of your code...

# 🟢 Initialize anomalies with a safe default at the very top
anomalies = 0

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies  # Use global keyword to ensure anomalies is accessible

    st.set_page_config(page_title="Healthcare IoT Patient Monitoring", layout="wide")
    st.title('🏥 Healthcare IoT Patient Monitoring Dashboard')
    
    # Load models
   # Load models
try:
    lstm_model = tf.keras.models.load_model('models/lstm_model.keras')
    rf_model = joblib.load('models/random_forest_model.pkl')
    st.success("✅ Models loaded successfully.")
except ModuleNotFoundError as e:
    st.error("❌ Required module is missing: {}".format(str(e)))
    st.write("Please update requirements.txt to include missing modules.")
except Exception as e:
    st.error("❌ Error loading models.")
    st.text(traceback.format_exc())  # Print detailed traceback


    # Load data
    try:
        data = pd.read_csv('data/cleaned_data.csv')
        st.success("✅ Data loaded successfully.")
        st.write("🔍 DEBUG: First few rows of data:")
        st.write(data.head())  # Print first few rows for verification
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())  # Print detailed traceback

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
        anomalies = 0  # Define locally inside the try block
        preds = rf_model.predict(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
        if isinstance(preds, np.ndarray):
            anomalies = int(sum(preds))
        st.metric(label="⚠️ Anomalies Detected", value=anomalies)
        st.progress(anomalies / len(data) * 100)
    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())  # Print detailed traceback

    # 🟢 Live predictions with LSTM
    st.subheader('🔮 LSTM Predictions for Health Deterioration')
    try:
        lstm_input = data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']].values.reshape(-1, 1, 8)
        lstm_preds = lstm_model.predict(lstm_input)
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
