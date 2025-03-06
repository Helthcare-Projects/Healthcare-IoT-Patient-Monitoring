# Import necessary libraries
import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import traceback
from sklearn.preprocessing import MinMaxScaler

# 🟢 Move this line immediately after imports
st.set_page_config(page_title="Healthcare IoT Patient Monitoring", layout="wide")

# 🟢 Initialize anomalies with a safe default at the very top
anomalies = 0

# 🟢 Define main function to encapsulate all code
def main():
    global anomalies  # Use global keyword to ensure anomalies is accessible

    st.title('🏥 Healthcare IoT Patient Monitoring Dashboard')

    # 🟢 Load models with custom objects
    try:
        from tensorflow.keras.losses import MeanSquaredError
        lstm_model = tf.keras.models.load_model(
            'models/lstm_model.h5',
            custom_objects={'mse': MeanSquaredError()}
        )
        rf_model = joblib.load('models/random_forest_model.pkl')
        st.success("✅ Models loaded successfully.")
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())
        return

    # 🟢 Load data
    try:
        data = pd.read_csv('data/cleaned_data.csv')
        st.success("✅ Data loaded successfully.")
        st.write(data.head())
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Normalize data
    try:
        scaler = MinMaxScaler()
        data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']] = scaler.fit_transform(
            data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']]
        )
        st.success("✅ Data normalized successfully.")
    except Exception as e:
        st.error("❌ Error normalizing data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Anomaly detection using Random Forest
    st.subheader('🚨 Anomaly Detection')
    try:
        # Adjust threshold for anomaly detection
        threshold = 0.9  # Increase threshold to reduce false positives
        preds = rf_model.predict_proba(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])[:, 1]
        anomalies = int(sum(preds > threshold))
        
        st.metric(label="⚠️ Anomalies Detected", value=anomalies)
        st.progress(min(anomalies / len(data), 1.0))

        # 🟢 Conditional warnings based on anomalies
        if anomalies > 5000:
            st.warning("🚨 Extremely high anomalies detected! Immediate investigation required.")
        elif anomalies > 1000:
            st.warning("⚠ High anomalies detected. Review data and model thresholds.")
        else:
            st.info("✅ Anomaly detection is within normal range.")
    except Exception as e:
        st.error("❌ Error during anomaly detection.")
        st.text(traceback.format_exc())

    # 🟢 LSTM Predictions
    st.subheader('🔮 LSTM Predictions for Health Deterioration')
    try:
        lstm_input = data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']].values.reshape(-1, 1, 8)
        lstm_preds = lstm_model.predict(lstm_input)
        lstm_preds = np.clip(lstm_preds, 0, 1)
        if isinstance(lstm_preds, np.ndarray):
            st.line_chart(lstm_preds[:50])
        else:
            st.write("No valid predictions from LSTM model.")
    except Exception as e:
        st.error("❌ Error during LSTM prediction.")
        st.text(traceback.format_exc())

    st.write('Monitor your health in real-time with AI-driven insights!')

# 🟢 Run the main function
if __name__ == "__main__":
    main()
