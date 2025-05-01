# Water Resources Management & Prediction

This project uses the ARIMA model for time-series forecasting to predict water levels in reservoirs. The application visualizes water level trends and provides forecasts based on historical data.

## Features
- Select a reservoir to visualize its water level trend.
- View the location of the reservoir on an interactive map.
- Forecast future water levels using the ARIMA model.
- Download the forecast as a CSV file.
- View model performance metrics such as MAE, RMSE, and MAPE.

## Requirements
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Statsmodels
- Scikit-learn

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/sampadaswami/Water-Resources-Management-And-Prediction.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Dataset
Ensure the dataset contains the following columns:
- `Reservoir_name`: Name of the reservoir
- `Date`: Date of the record
- `Level`: Water level
- `Lat`: Latitude
- `Long`: Longitude

## Model Parameters
- **p**: Auto-Regressive order
- **d**: Differencing
- **q**: Moving Average order
- **Forecast Days**: Number of days to predict

## License
MIT License

## Contact
Email: sampadasswami24@gmail.com  
LinkedIn: [Sampada Swami](https://www.linkedin.com/in/sampadaswami/)
