import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Load Dataset
DATA_PATH = "data1.csv"
df = pd.read_csv(DATA_PATH)

# Debugging: Check if data is loaded
if df.empty:
    st.error("Dataset is empty! Please check the file.")
    st.stop()

# Check required columns
required_columns = ['Reservoir_name', 'Date', 'Level', 'Lat', 'Long']
if not all(col in df.columns for col in required_columns):
    st.error("Missing necessary columns in dataset. Check column names.")
    st.write("Available columns:", df.columns)
    st.stop()

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')

# Drop missing values
df.dropna(subset=['Date', 'Reservoir_name', 'Level', 'Lat', 'Long'], inplace=True)

# Streamlit App Title
st.title("🌊 Water Resources Management & Prediction")

# Select reservoir
reservoirs = df['Reservoir_name'].dropna().unique()
if len(reservoirs) == 0:
    st.error("No reservoirs found in the dataset!")
    st.stop()
selected_reservoir = st.selectbox("Select Reservoir", reservoirs)

# Filter data for selected reservoir
df_reservoir = df[df['Reservoir_name'] == selected_reservoir].copy()
if df_reservoir.empty:
    st.warning(f"No data available for {selected_reservoir}")
    st.stop()

# Ensure data is sorted
df_reservoir = df_reservoir.sort_values(by='Date')

# Take latest Lat and Long
lat = df_reservoir['Lat'].values[0]
lon = df_reservoir['Long'].values[0]

# Set index for time-series analysis
df_reservoir.set_index('Date', inplace=True)
data = df_reservoir['Level']

# Prevent errors if data is too short
if len(data) < 10:
    st.warning("Not enough data to train the model.")
    st.stop()

# Spatial Map
st.subheader(f"🗺️ Location of {selected_reservoir}")
st.write(f"<small>Coordinates: Latitude {lat}, Longitude {lon}</small>", unsafe_allow_html=True)

fig_map = px.scatter_mapbox(
    lat=[lat],
    lon=[lon],
    zoom=7,
    center={"lat": lat, "lon": lon},
    height=500,
    mapbox_style="open-street-map",
)
fig_map.update_traces(
    marker=dict(size=20, color="red"),
    hovertemplate="<b>Reservoir:</b> %{lat}, %{lon}<br><b>Name:</b> " + selected_reservoir
)
fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map)

# Reservoir Trend Line Plot
st.subheader(f"📈 Reservoir Level Trend for {selected_reservoir}")
fig = px.line(df_reservoir, x=df_reservoir.index, y='Level', title="Water Level Over Time")
st.plotly_chart(fig)

# Show Latest Data
st.subheader("📅 Latest Data")
st.write(df_reservoir.tail())

# Sidebar Inputs for ARIMA Model (with emojis)
with st.sidebar:
    st.subheader("⚙️ ARIMA Model Parameters")
    p = st.slider("📉 p (AR)", min_value=0, max_value=10, value=5)
    d = st.slider("📉 d (Differencing)", min_value=0, max_value=2, value=1)
    q = st.slider("📉 q (MA)", min_value=0, max_value=10, value=0)
    forecast_days = st.slider("📅 Forecast Days", min_value=1, max_value=60, value=30)

# Train ARIMA Model
try:
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
except Exception as e:
    st.error(f"Model training failed: {str(e)}")
    st.stop()

# Forecast future
forecast = model_fit.forecast(steps=forecast_days)
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')

# Plot Forecast
st.subheader(f"📉 Forecast for {selected_reservoir}")
plt.figure(figsize=(10, 5))
plt.plot(data.index, data, label="Historical Data", color="blue")
plt.plot(future_dates, forecast, label="Forecast", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Reservoir Level")
plt.legend()
st.pyplot(plt)

# Show Predicted Values
st.subheader("🔮 Predicted Levels")
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Level": forecast.values})
st.write(forecast_df)

# Optional Download Button
st.subheader("⬇️ Download Forecast")
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Predictions as CSV",
    data=csv,
    file_name=f"{selected_reservoir}_forecast.csv",
    mime="text/csv"
)

# Updated Model Performance Section - HORIZONTAL
st.subheader("📊 Model Performance")
try:
    train_predictions = model_fit.predict(start=0, end=len(data)-1)
    mae = mean_absolute_error(data, train_predictions)
    rmse = np.sqrt(mean_squared_error(data, train_predictions))
    mape = np.mean(np.abs((data - train_predictions) / data)) * 100
    accuracy = 100 - mape

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="MAE", value=f"{mae:.2f}")
    col2.metric(label="RMSE", value=f"{rmse:.2f}")
    col3.metric(label="MAPE", value=f"{mape:.2f}%")
    col4.metric(label="Accuracy", value=f"{accuracy:.2f}%")
except:
    st.write("Model performance metrics could not be calculated.")
