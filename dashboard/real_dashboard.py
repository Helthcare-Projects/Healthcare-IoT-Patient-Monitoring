import os
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

# 🟢 Define paths dynamically
MODEL_PATH = 'models/'
LSTM_MODEL_PATH = os.path.join(MODEL_PATH, 'lstm_model_enhanced.keras')
RF_MODEL_PATH = os.path.join(MODEL_PATH, 'random_forest_model_enhanced.pkl')
XGB_MODEL_PATH = os.path.join(MODEL_PATH, 'xgboost_model_enhanced.pkl')
SCALER_PATH = os.path.join(MODEL_PATH, 'scaler_enhanced.pkl')
SCALER_COLUMNS_PATH = os.path.join(MODEL_PATH, 'scaler_columns.pkl')

# 🟢 Initialize global variables
anomalies = 0
cumulative_anomalies = 0

# 🟢 Define main function
def main():
    global anomalies, cumulative_anomalies

    st.title('🏥 Healthcare IoT Real-Time Dashboard')

    # Load models
    try:
        if not os.path.exists(LSTM_MODEL_PATH) or not os.path.exists(RF_MODEL_PATH) or not os.path.exists(XGB_MODEL_PATH):
            st.error("❌ Required model files are missing.")
            return

        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        rf_model = joblib.load(RF_MODEL_PATH)
        xgb_model = joblib.load(XGB_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        if os.path.exists(SCALER_COLUMNS_PATH):
            scaler_columns = joblib.load(SCALER_COLUMNS_PATH)
        else:
            st.warning("⚠️ scaler_columns.pkl not found. Using all available columns.")
            scaler_columns = None

        st.success("✅ Models loaded successfully.")
    except Exception as e:
        st.error("❌ Error loading models.")
        st.text(traceback.format_exc())
        return

    # Load data
    try:
        data = pd.read_csv('data/enhanced_data_realistic.csv')
        st.success("✅ Data loaded successfully.")
    except FileNotFoundError:
        st.error("❌ Data file not found.")
        return
    except Exception as e:
        st.error("❌ Error loading data.")
        st.text(traceback.format_exc())
        return

    # 🟢 Validate if 'Patient_ID' column exists
    if 'Patient_ID' not in data.columns:
        st.error("❌ 'Patient_ID' column not found.")
        st.write("Columns available:", list(data.columns))
        return

    # 🟢 Patient Selection
    patient_ids = data['Patient_ID'].unique()
    selected_patient = st.selectbox("Select Patient ID:", patient_ids)
    patient_data = data[data['Patient_ID'] == selected_patient]
    st.write(f"🔍 Viewing data for Patient ID: {selected_patient}")

    # 🟢 Normalize and Scale Data
    features = patient_data.drop(['Patient_ID', 'Risk_Level'], axis=1)
    if scaler_columns:
        features = features[scaler_columns]  # Align columns if scaler_columns is available
    scaled_features = scaler.transform(features)

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

# 🟢 Run main function
if __name__ == "__main__":
    main()
