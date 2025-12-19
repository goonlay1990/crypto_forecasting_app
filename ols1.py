# ============================================================
# Streamlit App: BTC-USD OLS (AR) vs ARIMA - 2 Years Daily
# ============================================================
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm  # for OLS

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Crypto Forecast: OLS vs ARIMA", layout="wide")
st.markdown("<style>.main {padding-top: 0px;}</style>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;margin-top:-10px;'>ARIMA & OLS Forecasting Model</h1>", unsafe_allow_html=True)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Model Parameters")
symbol = st.sidebar.text_input("Cryptocurrency Symbol", "BTC-USD")
period = st.sidebar.selectbox("Data Period", options=["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", options=["1d"], index=0)
prediction_ahead = st.sidebar.number_input("Prediction Days Ahead", min_value=1, max_value=90, value=30, step=1)

st.sidebar.markdown("---")
model_choice = st.sidebar.radio("Select Model", ["ARIMA", "OLS (AR via OLS)"], index=1)

st.sidebar.markdown("---")
train_ratio = st.sidebar.slider("Train Ratio (%)", min_value=60, max_value=90, value=80, step=5)

# Grid search ranges
st.sidebar.subheader("Grid Search Ranges")
p_max_arima = st.sidebar.slider("ARIMA max p", 0, 6, 4)
d_max_arima = st.sidebar.slider("ARIMA max d", 0, 3, 2)
q_max_arima = st.sidebar.slider("ARIMA max q", 0, 6, 4)
p_max_ols = st.sidebar.slider("OLS AR(p) max lags", 1, 12, 6)

run_button = st.sidebar.button("Predict")

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def download_crypto(sym, per, inter):
    return yf.download(sym, period=per, interval=inter)

def prepare_close(df):
    s = df[['Close']].dropna()
    s.index = pd.to_datetime(s.index)
    return s['Close']

def make_log_returns(close):
    return np.log(close).diff().dropna()

# ---------- ARIMA ----------
def evaluate_arima_model(train, test, order):
    try:
        model = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit()
        preds = fit.forecast(steps=len(test))
        mse = mean_squared_error(test, preds)
        return mse, fit
    except Exception:
        return float('inf'), None

def grid_search_arima(train, test, p_max, d_max, q_max):
    best_mse = float('inf')
    best_order, best_model = None, None
    for p, d, q in product(range(p_max+1), range(d_max+1), range(q_max+1)):
        mse, fit = evaluate_arima_model(train, test, (p, d, q))
        if mse < best_mse:
            best_mse, best_order, best_model = mse, (p, d, q), fit
    return best_order, best_mse, best_model

# ---------- OLS AR(p) ----------
def fit_ols_ar(returns, p):
    y = returns[p:]
    X = pd.concat([returns.shift(i) for i in range(1, p+1)], axis=1)[p:]
    X.columns = [f"lag{i}" for i in range(1, p+1)]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params, model

def recursive_forecast(params, init, steps):
    const = params[0]
    betas = params[1:]
    history = list(init)
    preds = []
    for _ in range(steps):
        r_hat = const + np.dot(betas, history)
        preds.append(r_hat)
        history = [r_hat] + history[:-1]
    return np.array(preds)

def returns_to_prices(start_price, returns):
    return start_price * np.exp(np.cumsum(returns))

# -----------------------------
# Main app logic
# -----------------------------
if run_button:
    try:
        raw = download_crypto(symbol, period, interval)
        close = prepare_close(raw)

        train_size = int(len(close) * train_ratio / 100)
        train_close = close.iloc[:train_size]
        test_close = close.iloc[train_size:]

        st.success(f"Data loaded: {len(close)} observations")

        if model_choice == "ARIMA":
            best_order, best_mse, best_model = grid_search_arima(
                train_close, test_close, p_max_arima, d_max_arima, q_max_arima
            )
            st.write(f"**Best ARIMA model:** {best_order}, Test MSE = {best_mse:.6f}")

            future_model = ARIMA(close, order=best_order, enforce_stationarity=False,
                                  enforce_invertibility=False).fit()
            future_fc = future_model.forecast(steps=prediction_ahead)

            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(close, label='Actual')
            ax.plot(future_fc.index, future_fc.values, label='Forecast')
            ax.legend()
            st.pyplot(fig)

        else:
            returns = make_log_returns(close)
            train_returns = returns.iloc[:train_size-1]
            test_returns = returns.iloc[train_size-1:]

            best_mse = float('inf')
            best_p, best_params = None, None
            for p in range(1, p_max_ols+1):
                if len(train_returns) <= p:
                    continue
                params, _ = fit_ols_ar(train_returns, p)
                init = train_returns.iloc[-p:][::-1].values
                preds = recursive_forecast(params, init, len(test_returns))
                prices = returns_to_prices(train_close.iloc[-1], preds)
                mse = mean_squared_error(test_close.values, prices)
                if mse < best_mse:
                    best_mse, best_p, best_params = mse, p, params

            st.write(f"**Best OLS AR(p): p={best_p}, Test MSE={best_mse:.6f}**")

            init_full = returns.iloc[-best_p:][::-1].values
            future_returns = recursive_forecast(best_params, init_full, prediction_ahead)
            future_prices = returns_to_prices(close.iloc[-1], future_returns)

            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(close, label='Actual')
            ax.plot(pd.date_range(close.index[-1], periods=prediction_ahead+1, freq='D')[1:],
                    future_prices, label='Forecast')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(str(e))
else:
    st.info("👈 Set parameters and click **Predict** to run the model.")
