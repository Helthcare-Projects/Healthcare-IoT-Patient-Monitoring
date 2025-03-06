# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback

# 🟢 Move this line immediately after imports
st.set_page_config(page_title="Healthcare IoT Patient Monitoring", layout="wide")

# 🟢 Initialize anomalies with a safe default at the very top
anomalies = 0

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies  # Use global keyword to ensure anomalies is accessible

    st.title('🏥 Healthcare IoT Patient Monitoring Dashboard')
    
    # Load models
    try:
        # 🟢 Try loading .h5 model instead of .keras
        lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
        rf_model = joblib.load('models/random_forest_model.pkl')
        st.success("✅ Models loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {str(e)}")
        st.write("Please ensure the model files are correctly uploaded.")
        return
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())  # Print detailed traceback
        return

    # Load data
    try:
        data = pd.read_csv('data/cleaned_data.csv')
        st.success("✅ Data loaded successfully.")
        st.write(data.head())  # Print first few rows for verification
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())  # Print detailed traceback
        return

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

        # Limit progress value between 0 and 1
        anomaly_percentage = min(anomalies / len(data), 1.0)
        
        st.metric(label="⚠️ Anomalies Detected", value=anomalies)
        st.progress(anomaly_percentage)  # Updated to use limited range

        # 🟢 Additional warnings if anomalies are high
        if anomalies > 500:
            st.warning("⚠️ Unusually high anomalies detected. Please check data quality or model thresholds.")
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

        # Ensure LSTM predictions are between 0 and 1
        lstm_preds = np.clip(lstm_preds, 0, 1)

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
