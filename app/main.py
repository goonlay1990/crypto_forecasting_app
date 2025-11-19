import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Crypto Forecasting App")
st.sidebar.header("Settings")

# Sidebar inputs
crypto = st.sidebar.selectbox("Select Cryptocurrency", ["Polkadot", "Cardano", "Cosmos", "Dogecoin", "Bitcoin"])
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 7, 60, 30)

# Fetch data
st.write(f"Fetching data for {crypto}...")
ticker_map = {
    "Polkadot": "DOT-USD",
    "Cardano": "ADA-USD",
    "Cosmos": "ATOM-USD",
    "Dogecoin": "DOGE-USD",
    "Bitcoin": "BTC-USD"
}
data = yf.download(ticker_map[crypto], start=start_date, end=end_date)

st.subheader("Historical Price")
st.line_chart(data['Close'])

st.subheader("Forecast Results")
st.write("Coming soon: OLS, ARIMA, GARCH forecasts")
