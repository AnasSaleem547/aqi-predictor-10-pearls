#!/usr/bin/env python3
"""
Simple and Reliable LSTM Model for AQI Prediction
Target: Generate predictions in the 140-170 range (recent actual mean: 155)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv
import pytz
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv()

# Environment
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
HOPSWORKS_PROJECT = os.getenv('HOPSWORKS_PROJECT')

# Define PKT timezone for proper datetime handling
PKT = pytz.timezone('Asia/Karachi')


def get_latest_feature_group_version(fs, feature_group_name: str) -> int:
    """Get the latest version of a feature group from Hopsworks."""
    try:
        feature_groups = fs.get_feature_groups(name=feature_group_name)
        if not feature_groups:
            print(f"📋 No existing feature groups found for '{feature_group_name}'")
            return 0
        versions = [fg.version for fg in feature_groups]
        latest_version = max(versions)
        print(f"📊 Found existing versions for '{feature_group_name}': {sorted(versions)}")
        print(f"🔢 Latest version: {latest_version}")
        return latest_version
    except Exception as e:
        print(f"⚠️ Error getting feature group versions: {str(e)}")
        return 0


def fetch_features() -> pd.DataFrame:
    """Fetch features from Hopsworks feature store."""
    print("📡 Fetching features from Hopsworks...")
    
    if not HOPSWORKS_API_KEY or not HOPSWORKS_PROJECT:
        raise ValueError("❌ Hopsworks credentials not found. Please set HOPSWORKS_API_KEY and HOPSWORKS_PROJECT environment variables.")
    
    try:
        import hopsworks
    except ImportError:
        raise ImportError("❌ Hopsworks library not installed. Install with: pip install hopsworks")
    
    try:
        print("🔑 Connecting to Hopsworks...")
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
        fs = project.get_feature_store()
        
        # Get the latest version dynamically
        feature_group_name = "karachifeatures"
        latest_version = get_latest_feature_group_version(fs, feature_group_name)
        
        if latest_version == 0:
            raise RuntimeError("❌ No feature group versions found in Hopsworks")
        
        print(f"📊 Loading feature group '{feature_group_name}' version {latest_version}...")
        fg = fs.get_feature_group(name=feature_group_name, version=latest_version)
        df = fg.read()
        print(f"✅ Successfully loaded features from Hopsworks: {len(df)} records, {len(df.columns)} columns")
        
        # Sort by datetime before saving to CSV file
        df_sorted = df.sort_values('datetime').reset_index(drop=True)
        
        # Save sorted fetched data to timestamped CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"features_fetched_from_hopsworks_{timestamp}.csv"
        df_sorted.to_csv(csv_filename, index=False)
        print(f"💾 Saved sorted fetched data to: {csv_filename}")
        
        return df_sorted
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch data from Hopsworks: {type(e).__name__}: {e}")

def create_sequences(data, seq_length, pred_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length:i + seq_length + pred_length])
    return np.array(X), np.array(y)

def main():
    print("=== SIMPLE RELIABLE LSTM AQI PREDICTION ===")
    
    try:
        # Load data using Hopsworks or fallback to local CSV
        df = fetch_features()
        print(f"✅ Data loaded: {df.shape}")
        
        # Ensure data is sorted by datetime for proper time series sequence
        df = df.sort_values('datetime').reset_index(drop=True)
        print(f"✅ Data sorted by datetime for time series consistency")
        
        # Check recent AQI patterns
        recent_aqi = df['aqi_epa_calc'].tail(100)
        print(f"📊 Recent AQI (last 100): {recent_aqi.min():.1f} - {recent_aqi.max():.1f} (mean: {recent_aqi.mean():.1f})")
        
        # Prepare data - use only AQI values for simplicity
        aqi_data = df['aqi_epa_calc'].values.reshape(-1, 1)
        
        # Scale data - fit on ALL data to capture full range
        scaler = MinMaxScaler()
        aqi_scaled = scaler.fit_transform(aqi_data)
        
        print(f"✅ Data scaled. Original range: {aqi_data.min():.1f} - {aqi_data.max():.1f}")
        
        # Create sequences
        seq_length = 24  # 24 hours of history
        pred_length = 72  # 72 hours prediction
        
        X, y = create_sequences(aqi_scaled.flatten(), seq_length, pred_length)
        print(f"✅ Sequences created: X shape {X.shape}, y shape {y.shape}")
        
        if len(X) == 0:
            raise ValueError("No sequences created!")
        
        # Time series split - use most recent data for testing
        train_size = int(len(X) * 0.8)
        val_size = int(len(X) * 0.1)
        
        X_train = X[:train_size].reshape(train_size, seq_length, 1)
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size].reshape(val_size, seq_length, 1)
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:].reshape(-1, seq_length, 1)
        y_test = y[train_size + val_size:]
        
        print(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Build simple LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(pred_length)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print("✅ Model compiled")
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("🏃 Training model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Test model
        test_pred = model.predict(X_test, verbose=0)
        test_pred_original = scaler.inverse_transform(test_pred.reshape(-1, 1)).reshape(test_pred.shape)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
        
        test_mse = mean_squared_error(y_test_original.flatten(), test_pred_original.flatten())
        test_mae = mean_absolute_error(y_test_original.flatten(), test_pred_original.flatten())
        
        print(f"✅ Test MSE: {test_mse:.2f}, Test MAE: {test_mae:.2f}")
        print(f"✅ Test predictions range: {test_pred_original.min():.1f} - {test_pred_original.max():.1f} (mean: {test_pred_original.mean():.1f})")
        
        # Generate future predictions
        last_sequence = aqi_scaled[-seq_length:].reshape(1, seq_length, 1)
        future_pred = model.predict(last_sequence, verbose=0)
        future_pred_original = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        
        # Create future timestamps
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            last_time = df['datetime'].iloc[-1]
        else:
            last_time = pd.Timestamp.now()
        
        future_times = pd.date_range(start=last_time + pd.Timedelta(hours=1), periods=pred_length, freq='H')
        
        # Create results DataFrame
        results = pd.DataFrame({
            'datetime': future_times,
            'predicted_aqi': future_pred_original
        })
        
        # Convert datetime to PKT timezone for CSV output
        # Check if datetime is already timezone-aware
        if results['datetime'].dt.tz is None:
            # If timezone-naive, assume it's PKT and localize it
            results['datetime'] = results['datetime'].dt.tz_localize(PKT)
        else:
            # If already timezone-aware, convert to PKT
            results['datetime'] = results['datetime'].dt.tz_convert(PKT)
        
        print(f"🕐 Converted datetime to PKT timezone for CSV output")
        
        # Save predictions
        results.to_csv('lstm_aqi_predictions_simple_reliable.csv', index=False)
        print(f"✅ Predictions saved to 'lstm_aqi_predictions_simple_reliable.csv'")
        print(f"📅 Datetime format: PKT timezone ({results['datetime'].iloc[0]})")
        
        # Display results
        pred_range = f"{results['predicted_aqi'].min():.1f} - {results['predicted_aqi'].max():.1f}"
        pred_mean = results['predicted_aqi'].mean()
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"   Range: {pred_range}")
        print(f"   Mean: {pred_mean:.1f}")
        print(f"   Target range: 140-170 (recent actual mean: {recent_aqi.mean():.1f})")
        
        # Assessment
        if 140 <= pred_mean <= 170:
            print(f"   ✅ SUCCESS: Predictions are in the realistic range!")
        elif pred_mean < 100:
            print(f"   ❌ UNDERESTIMATING: Predictions too low")
        elif pred_mean > 200:
            print(f"   ❌ OVERESTIMATING: Predictions too high")
        else:
            print(f"   ⚠️  CLOSE: Predictions are close to realistic range")
        
        # Show sample predictions
        print(f"\n📋 Sample predictions:")
        for i in range(min(10, len(results))):
            print(f"   {results.iloc[i]['datetime'].strftime('%Y-%m-%d %H:%M')}: {results.iloc[i]['predicted_aqi']:.1f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    predictions = main()