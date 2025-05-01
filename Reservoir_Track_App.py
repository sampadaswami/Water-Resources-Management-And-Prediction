import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Load Dataset
DATA_PATH = "D:/WM/data1.csv"
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

# Streamlit App
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

# Take latest Lat and Long **before setting index**
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

fig_map = px.scatter_mapbox(
    lat=[lat],
    lon=[lon],
    zoom=7,
    center={"lat": lat, "lon": lon},
    height=500,
    mapbox_style="open-street-map",
)

# Add big size marker + reservoir name on hover
fig_map.update_traces(
    marker=dict(size=20, color="red"),
    hovertemplate="<b>Reservoir:</b> %{lat}, %{lon}<br><b>Name:</b> " + selected_reservoir
)

fig_map.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
    title=f"Location: {selected_reservoir}"
)
st.plotly_chart(fig_map)

# Interactive Plotly Visualization
st.subheader(f"📈 Reservoir Level Trend for {selected_reservoir}")
fig = px.line(df_reservoir, x=df_reservoir.index, y='Level', title="Water Level Over Time")
st.plotly_chart(fig)

# Train ARIMA Model
try:
    model = ARIMA(data, order=(5,1,0))  # (p,d,q) values can be tuned
    model_fit = model.fit()
except Exception as e:
    st.error(f"Model training failed: {str(e)}")
    st.stop()

# Forecast for next 30 days
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
forecast = model_fit.forecast(steps=30)

# Calculate Model Accuracy
train_predictions = model_fit.predict(start=0, end=len(data)-1)
mae = mean_absolute_error(data, train_predictions)
mse = mean_squared_error(data, train_predictions)
rmse = np.sqrt(mse)

# Display Accuracy in Streamlit
st.subheader("📊 Model Performance")
st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
st.metric(label="Mean Squared Error (MSE)", value=f"{mse:.2f}")
st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")

# Plot Predictions
st.subheader(f"📉 Forecast for {selected_reservoir}")
plt.figure(figsize=(10, 5))
plt.plot(data.index, data, label="Historical Data", color="blue")
plt.plot(future_dates, forecast, label="Forecast", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Reservoir Level")
plt.legend()
st.pyplot(plt)

# Show Latest Data
st.subheader("📅 Latest Data")
st.write(df_reservoir.tail())

# Show Predicted Values
st.subheader("🔮 Predicted Levels")
forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Level": forecast.values})
st.write(forecast_df)

# Save Predictions
OUTPUT_PATH = "D:/WM/predicted_levels.csv"
forecast_df.to_csv(OUTPUT_PATH, index=False)
st.success(f"✅ Predictions saved to {OUTPUT_PATH}")
