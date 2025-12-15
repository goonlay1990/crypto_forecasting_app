import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import yfinance as yf

from models.arima_model import run_arima_prices, run_arima_returns

st.set_page_config(page_title="Crypto Forecasting App", layout="wide")
st.title("Crypto Forecasting App")
st.sidebar.header("Settings")

crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ["Polkadot", "Cardano", "Cosmos", "Dogecoin", "Bitcoin"]
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)

st.sidebar.subheader("ARIMA Settings")
arima_mode = st.sidebar.selectbox(
    "ARIMA mode",
    ["Returns (recommended)", "Prices (levels)"]
)

default_d = 0 if arima_mode == "Returns (recommended)" else 1

p = st.sidebar.number_input("p (AR)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("d (Diff)", min_value=0, max_value=2, value=int(default_d), step=1)
q = st.sidebar.number_input("q (MA)", min_value=0, max_value=5, value=1, step=1)

ticker_map = {
    "Polkadot": "DOT-USD",
    "Cardano": "ADA-USD",
    "Cosmos": "ATOM-USD",
    "Dogecoin": "DOGE-USD",
    "Bitcoin": "BTC-USD",
}

st.write(f"Fetching data for **{crypto}**...")
data = yf.download(ticker_map[crypto], start=start_date, end=end_date)

if data is None or data.empty or "Close" not in data.columns:
    st.error("No data returned. Try a different date range or cryptocurrency.")
    st.stop()

close_series = data["Close"].dropna()
if close_series.empty:
    st.error("Close price series is empty after removing missing values.")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("Historical Price (Close)")
    st.line_chart(close_series)

with right:
    st.subheader("ARIMA Forecast")

    if arima_mode == "Prices (levels)":
        fit, forecast = run_arima_prices(
            close_series=close_series,
            order=(int(p), int(d), int(q)),
            steps=int(forecast_days),
        )
        forecast_values = forecast.values
    else:
        fit, returns_fc, price_fc = run_arima_returns(
            close_series=close_series,
            order=(int(p), int(d), int(q)),
            steps=int(forecast_days),
        )
        forecast_values = price_fc.values

    last_date = close_series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=int(forecast_days),
        freq="D",
    )
    forecast_series = pd.Series(forecast_values, index=future_dates)

    st.line_chart(forecast_series)

    st.caption(
        "Returns-based ARIMA forecasts log returns and then converts them into a price path."
    )

    with st.expander("ARIMA Model Summary"):
        st.text(fit.summary())
