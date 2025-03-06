
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

# Load models
try:
    lstm_model = tf.keras.models.load_model('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/models/lstm_model.h5')
    print("✅ LSTM model loaded successfully.")
except:
    print("❌ Error loading LSTM model.")

try:
    rf_model = joblib.load('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/models/random_forest_model.pkl')
    print("✅ Random Forest model loaded successfully.")
except:
    print("❌ Error loading Random Forest model.")

# Load data
try:
    data = pd.read_csv('/content/drive/MyDrive/Healthcare-Project/Healthcare-IoT-Patient-Monitoring/data/cleaned_data.csv')
    print("✅ Data loaded successfully.")
    print("📊 Data sample:")
    print(data.head())
except:
    print("❌ Error loading data.")

# Test Random Forest Prediction
try:
    preds = rf_model.predict(data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']])
    print("✅ Random Forest predictions successful.")
    print("Predictions sample:", preds[:5])
except:
    print("❌ Error during Random Forest prediction.")

# Test LSTM Prediction
try:
    lstm_input = data[['Temperature', 'Heart Rate', 'Pulse', 'BPSYS', 'BPDIA', 'Respiratory Rate', 'Oxygen Saturation', 'PH']].values.reshape(-1, 1, 8)
    lstm_preds = lstm_model.predict(lstm_input)
    print("✅ LSTM predictions successful.")
    print("Predictions sample:", lstm_preds[:5])
except:
    print("❌ Error during LSTM prediction.")
