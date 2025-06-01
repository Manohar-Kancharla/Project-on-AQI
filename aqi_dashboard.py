import streamlit as st
import numpy as np
import pandas as pd
import requests
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta, timezone
import joblib
import os
import plotly.express as px
from dotenv import load_dotenv
import math

# Load environment variables
load_dotenv()

# --------------- Configuration ----------------
st.set_page_config(page_title="Real-Time AQI Dashboard", layout="wide")

API_KEY = '91522703826df616b8af614f912b023b'
LAT, LON = 16.5131, 80.5165
UPDATE_INTERVAL = 300  # 5 minutes in seconds

MODEL_PATH = 'lstm_aqi_model.h5'
FEATURE_SCALER_PATH = 'feature_scaler.pkl'
TARGET_SCALER_PATH = 'target_scaler.pkl'
SEQUENCE_LENGTH = 10

FEATURES = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone',
            'Toluene', 'AT', 'RH', 'WS', 'SR', 'BP', 'hour', 'day', 'month',
            'weekday', 'season_Autumn', 'season_Monsoon', 'season_Summer', 'season_Winter']

# --------------- Model Loading ----------------
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

@st.cache_resource
def load_scalers():
    try:
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        return feature_scaler, target_scaler
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
        return None, None

model = load_lstm_model()
feature_scaler, target_scaler = load_scalers()

# --------------- Solar Radiation Calculation ----------------
def calculate_solar_radiation():
    now = datetime.now(timezone.utc)
    doy = now.timetuple().tm_yday
    declination = 23.45 * math.sin(math.radians(360*(284+doy)/365))
    hour_angle = 15 * (now.hour + now.minute/60 - 12)
    alt = math.degrees(math.asin(
        math.sin(math.radians(LAT)) * math.sin(math.radians(declination)) +
        math.cos(math.radians(LAT)) * math.cos(math.radians(declination)) *
        math.cos(math.radians(hour_angle))
    ))
    if alt <= 0: return 0
    return 1353 * 0.6 ** (1/math.sin(math.radians(alt)))

# --------------- Data Fetching ----------------
def fetch_real_time_data():
    try:
        air_url = f'http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}'
        air_data = requests.get(air_url, timeout=10).json()
        weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric'
        weather_data = requests.get(weather_url, timeout=10).json()
        return {
            'PM2.5': air_data['list'][0]['components']['pm2_5'],
            'PM10': air_data['list'][0]['components']['pm10'],
            'NO': air_data['list'][0]['components']['no'],
            'NO2': air_data['list'][0]['components']['no2'],
            'SO2': air_data['list'][0]['components']['so2'],
            'Ozone': air_data['list'][0]['components']['o3'],
            'CO': air_data['list'][0]['components']['co'] / 1000,
            'NOx': (air_data['list'][0]['components']['no'] / 30.01 +
                    air_data['list'][0]['components']['no2'] / 46.01) * 24.45,
            'NH3': max(1.5, 0.15 * air_data['list'][0]['components']['no2']),
            'Toluene': max(0.8, 0.05 * air_data['list'][0]['components']['co']),
            'AT': weather_data['main']['temp'],
            'RH': weather_data['main']['humidity'],
            'WS': weather_data['wind']['speed'],
            'BP': weather_data['main']['pressure'] * 0.750062,
            'SR': calculate_solar_radiation()
        }
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

def get_season_encoding(month):
    season = 'Winter' if month in [12,1,2] else \
             'Summer' if month in [3,4,5] else \
             'Monsoon' if month in [6,7,8] else 'Autumn'
    return {f'season_{s}': int(s == season) for s in ['Autumn', 'Monsoon', 'Summer', 'Winter']}

# --------------- Feature Processing ----------------
def generate_feature_vector():
    now = datetime.now()
    real_time = fetch_real_time_data()
    if not real_time:
        return None, None
    temporal = {
        'hour': now.hour,
        'day': now.day,
        'month': now.month,
        'weekday': now.weekday()
    }
    season_features = get_season_encoding(now.month)
    combined = {**real_time, **temporal, **season_features}
    ordered = [combined.get(f, 0) for f in FEATURES]
    sequence = np.array([ordered] * SEQUENCE_LENGTH)
    return sequence.reshape(1, SEQUENCE_LENGTH, len(FEATURES)), pd.DataFrame([ordered], columns=FEATURES)

# --------------- Data Storage ----------------
def save_results_to_csv(predicted_aqi, timestamp, feature_values=None):
    results_file = 'predictions.csv'
    data = {'timestamp': timestamp, 'predicted_aqi': predicted_aqi}
    if feature_values is not None:
        for f in FEATURES:
            if f in feature_values.columns:
                data[f] = feature_values[f].iloc[0]
    new_row = pd.DataFrame([data])
    if not os.path.exists(results_file):
        new_row.to_csv(results_file, index=False)
    else:
        try:
            existing = pd.read_csv(results_file)
            new_row.to_csv(results_file, mode='a', header=False, index=False)
        except:
            backup_name = f"predictions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            os.rename(results_file, backup_name)
            new_row.to_csv(results_file, index=False)

def load_historical_predictions():
    results_file = 'predictions.csv'
    if not os.path.exists(results_file):
        return pd.DataFrame(columns=['timestamp', 'predicted_aqi'] + FEATURES)
    try:
        df = pd.read_csv(results_file, parse_dates=['timestamp'])
        for col in ['timestamp', 'predicted_aqi'] + FEATURES:
            if col not in df.columns:
                df[col] = None
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(columns=['timestamp', 'predicted_aqi'] + FEATURES)

# --------------- Visualization ----------------
def display_current_conditions(prediction, features):
    prediction = max(0, min(500, prediction))
    aqi_color = "#00E400" if prediction <= 50 else \
               "#FFFF00" if prediction <= 100 else \
               "#FF7E00" if prediction <= 150 else \
               "#FF0000" if prediction <= 200 else \
               "#8F3F97" if prediction <= 300 else "#7E0023"
    st.markdown(f"""
    <div style="background-color:{aqi_color};padding:20px;border-radius:10px;text-align:center;">
        <h1 style="color:white;margin:0;">Current AQI: {prediction:.1f}</h1>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Air Pollutants")
        st.metric("PM2.5 (µg/m³)", f"{features['PM2.5'][0]:.1f}")
        st.metric("PM10 (µg/m³)", f"{features['PM10'][0]:.1f}")
        st.metric("NO (µg/m³)", f"{features['NO'][0]:.1f}")
        st.metric("NO2 (µg/m³)", f"{features['NO2'][0]:.1f}")
    with col2:
        st.markdown("### More Pollutants")
        st.metric("NOx (ppb)", f"{features['NOx'][0]:.1f}")
        st.metric("NH3 (µg/m³)", f"{features['NH3'][0]:.1f}")
        st.metric("SO2 (µg/m³)", f"{features['SO2'][0]:.1f}")
        st.metric("Ozone (µg/m³)", f"{features['Ozone'][0]:.1f}")
    with col3:
        st.markdown("### Weather & Others")
        st.metric("CO (mg/m³)", f"{features['CO'][0]:.3f}")
        st.metric("Temp (°C)", f"{features['AT'][0]:.1f}")
        st.metric("Humidity (%)", f"{features['RH'][0]:.0f}")
        st.metric("Wind (m/s)", f"{features['WS'][0]:.1f}")
        st.metric("Pressure (mmHg)", f"{features['BP'][0]:.1f}")
        st.metric("Solar Radiation (W/m²)", f"{features['SR'][0]:.1f}")

def plot_aqi_trends(df):
    if len(df) < 2:
        st.warning("Collecting data... will show trends after more predictions")
        return
    df = df.set_index('timestamp').sort_index()
    fig = px.line(df, y='predicted_aqi', title='AQI Trend Over Time', labels={'predicted_aqi': 'AQI'})
    aqi_bands = [
        (0, 50, "Good", "green"),
        (51, 100, "Moderate", "yellow"),
        (101, 150, "Unhealthy (Sensitive)", "orange"),
        (151, 200, "Unhealthy", "red"),
        (201, 300, "Very Unhealthy", "purple"),
        (301, 500, "Hazardous", "maroon")
    ]
    for band in aqi_bands:
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor=band[3], opacity=0.2, annotation_text=band[2])
    st.plotly_chart(fig, use_container_width=True)

# --------------- Main App Logic ----------------
def run_prediction():
    with st.spinner("Fetching data and predicting AQI..."):
        seq_input, instance_df = generate_feature_vector()
        if seq_input is None:
            st.error("Feature generation failed")
            return
        if model is None or target_scaler is None:
            st.error("Model or scalers not loaded")
            return
        try:
            prediction_scaled = model.predict(seq_input, verbose=0)[0][0]
            prediction_actual = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
            pm25 = instance_df['PM2.5'].iloc[0]
            expected_aqi = pm25 * 20
            if abs(prediction_actual - expected_aqi) > 100:
                prediction_actual = min(500, expected_aqi * 1.2)
            prediction_actual = max(0, min(500, prediction_actual))
            timestamp = datetime.now()
            st.session_state.last_update = timestamp
            save_results_to_csv(prediction_actual, timestamp, instance_df)
            st.session_state.predictions = load_historical_predictions()
            display_current_conditions(prediction_actual, instance_df)
            st.success(f"Updated at {timestamp.strftime('%H:%M:%S')}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# --------------- Streamlit UI ----------------

# --- ALWAYS SHOW TITLE! ---
st.title("🌍 Real-Time AQI Prediction Dashboard")

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(seconds=UPDATE_INTERVAL+1)
if 'predictions' not in st.session_state:
    st.session_state.predictions = load_historical_predictions()

if (datetime.now() - st.session_state.last_update).seconds > UPDATE_INTERVAL:
    run_prediction()

col1, col2 = st.columns(2)
col1.button("Predict Now", on_click=run_prediction)
col2.button("Refresh Data", on_click=lambda: st.session_state.predictions.update(load_historical_predictions()))

st.subheader("AQI History")
plot_aqi_trends(st.session_state.predictions)

# ---- RAW DATA CHECKBOX ----
if st.checkbox("Show raw data"):
    st.subheader("Raw Prediction Data")
    st.dataframe(st.session_state.predictions, use_container_width=True)

with st.sidebar:
    st.header("Configuration")
    st.markdown(f"*Location:* {LAT}, {LON}")
    st.markdown(f"*Last Update:* {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    st.header("Unit Reference")
    st.markdown("""
    - PM2.5/PM10: µg/m³
    - NO/NO2/SO2/O3/NH3/Toluene: µg/m³
    - CO: mg/m³
    - NOx: ppb
    - Temp: °C
    - Humidity: %
    - Wind: m/s
    - Pressure: mmHg
    - Solar Radiation: W/m²
    """)
    st.header("About")
    st.markdown("Real-time AQI prediction using LSTM and OpenWeatherMap API")
