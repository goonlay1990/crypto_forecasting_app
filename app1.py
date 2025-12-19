# ============================================================
# Cryptocurrency Forecasting App
# OLS (AR) vs ARIMA vs GARCH
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="Cryptocurrency Forecasting", layout="wide")
st.title("📈 Cryptocurrency Forecasting")

st.sidebar.header("Settings")

crypto_map = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Polkadot": "DOT-USD"
}

crypto_name = st.sidebar.selectbox("Select Cryptocurrency", crypto_map.keys())
symbol = crypto_map[crypto_name]

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
run = st.sidebar.button("Run Forecast")

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df[["Close"]].dropna()

# ------------------------------------------------------------
# OLS (AR(1)) Forecast on Log Returns
# ------------------------------------------------------------
def ols_ar_forecast(series, steps):
    log_ret = np.log(series).diff().dropna()

    X = sm.add_constant(log_ret.shift(1).dropna())
    y = log_ret.iloc[1:]

    model = sm.OLS(y, X).fit()

    last_r = log_ret.iloc[-1]
    preds = []

    for _ in range(steps):
        r_hat = model.params["const"] + model.params[1] * last_r
        preds.append(r_hat)
        last_r = r_hat

    last_price = float(series.iloc[-1])
    price_path = last_price * np.exp(np.cumsum(preds))

    forecast_index = pd.date_range(
        start=series.index[-1],
        periods=steps + 1,
        freq="D"
    )[1:]

    return pd.Series(price_path, index=forecast_index), model

# ------------------------------------------------------------
# ARIMA Forecast
# ------------------------------------------------------------
def arima_forecast(series, steps):
    model = ARIMA(series, order=(1, 1, 1)).fit()
    forecast = model.forecast(steps)

    return forecast, model

# ------------------------------------------------------------
# GARCH Volatility Forecast (not price)
# ------------------------------------------------------------
def garch_forecast(series, steps):
    returns = 100 * np.log(series).diff().dropna()

    model = arch_model(returns, p=1, q=1)
    res = model.fit(disp="off")

    var_forecast = res.forecast(horizon=steps).variance.iloc[-1]
    vol_forecast = np.sqrt(var_forecast)

    return vol_forecast, res

# ------------------------------------------------------------
# Run app
# ------------------------------------------------------------
if run:
    data = load_data(symbol, start_date, end_date)
    close = data["Close"]

    train_size = int(len(close) * 0.8)
    train = close.iloc[:train_size]
    test = close.iloc[train_size:]

    # Forecasts
    ols_pred, ols_model = ols_ar_forecast(train, len(test))
    arima_pred, arima_model = arima_forecast(train, len(test))
    garch_vol, garch_model = garch_forecast(train, 30)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    st.subheader("Price Forecast Comparison")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(close, label="Actual", color="black")
    ax.plot(ols_pred, label="OLS (AR)", linestyle="--")
    ax.plot(arima_pred, label="ARIMA", linestyle="--")

    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)

    # --------------------------------------------------------
    # GARCH output
    # --------------------------------------------------------
    st.subheader("GARCH Volatility Forecast (Next 30 Days)")
    st.line_chart(garch_vol)

else:
    st.info("👈 Select settings and click **Run Forecast**")
