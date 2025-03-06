
import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np
import traceback  # Import traceback for error logging

# 🟢 Initialize anomalies with a safe default at the very top
anomalies = 0

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies  # Use global keyword to ensure anomalies is accessible

    st.write("🔍 DEBUG: Starting Streamlit dashboard...")

    # Load models
    try:
        lstm_model = tf.keras.models.load_model('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/models/lstm_model.keras')
        rf_model = joblib.load('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/models/random_forest_model.pkl')
        st.write("✅ Models loaded successfully.")
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())  # Print detailed traceback

    # Load data
    try:
        data = pd.read_csv('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/data/cleaned_data.csv')
        st.write("✅ Data loaded successfully.")
        st.write("🔍 DEBUG: First few rows of data:")
        st.write(data.head())  # Print first few rows for verification
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())  # Print detailed traceback

    st.title('Healthcare IoT Patient Monitoring Dashboard')

    # Display live charts
    st.subheader('Vital Signs Over Time')
    try:
        st.line_chart(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
    except Exception as e:
        st.error("❌ Error displaying live charts.")
        st.text(traceback.format_exc())  # Print detailed traceback

    # 🟢 Anomaly detection using Random Forest
    st.subheader('Anomaly Detection')
    try:
        st.write("🔍 DEBUG: Before predicting anomalies, anomalies =", anomalies)
        preds = rf_model.predict(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
        st.write("🔍 DEBUG: Predictions array:", preds[:5])  # Print first few predictions
        if isinstance(preds, np.ndarray):
            anomalies = int(sum(preds))  # Safely update anomalies
            st.write("🔍 DEBUG: After predicting, anomalies =", anomalies)
        st.write(f'⚠️ Anomalies Detected: 0')
    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())  # Print detailed traceback
        st.write('⚠️ Anomalies Detected: 0')  # Fallback if error occurs

    # 🟢 Live predictions with LSTM
    st.subheader('Live Predictions (LSTM)')
    try:
        lstm_input = data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']].values.reshape(-1, 1, 8)
        lstm_preds = lstm_model.predict(lstm_input)
        if isinstance(lstm_preds, np.ndarray):
            st.write("🔍 DEBUG: LSTM Predictions sample:", lstm_preds[:5])
            st.write('LSTM Predictions:', lstm_preds[:5])
        else:
            st.write("No valid predictions from LSTM model.")
    except Exception as e:
        st.error("❌ Error during LSTM prediction.")
        st.text(traceback.format_exc())  # Print detailed traceback

    st.write('Monitor your health in real-time with AI-driven insights!')

# 🟢 Run the main function
if __name__ == "__main__":
    main()
