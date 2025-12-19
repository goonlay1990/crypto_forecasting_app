
# -*- coding: utf-8 -*-
"""
Crypto Forecast Dashboard: OLS vs ARIMA vs GARCH
- Data: Yahoo Finance via yfinance (daily close)
- Models:
  1) OLS (log-price linear trend)
  2) ARIMA (grid-search order by AIC on price)
  3) GARCH(1,1) on daily returns (mean ARX(1)), price path via return compounding
- Metrics: RMSE, MAE, MAPE, MSFE ratio vs naive random-walk
- UI: Streamlit (dropdown for ticker, sliders, metrics, chart)
- CLI: optional flags to run headless
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit imports wrapped to allow CLI usage without requiring streamlit at import time
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import yfinance as yf


# -------------------------------
# Data fetching and preparation
# -------------------------------
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "ADA-USD",
    "XRP-USD", "DOGE-USD", "LTC-USD", "DOT-USD", "AVAX-USD"
]

def get_crypto_data(ticker: str, years: int = 2) -> pd.Series:
    """
    Download daily close prices for the last `years` years.
    Returns a pandas Series indexed by Date with dtype float.
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * max(1, years))
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}. Try another ticker or increase lookback.")
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq("D")  # daily frequency to keep chart smooth (Yahoo is business days; we forward-fill)
    s = s.ffill()
    return s


# -------------------------------
# Utility: train/test split & metrics
# -------------------------------
def train_test_split(series: pd.Series, test_size: int):
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("Invalid test_size. Must be between 1 and len(series)-1.")
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test

def metrics_vs_naive(test: pd.Series, pred: pd.Series, train_last: float) -> dict:
    """
    Compare predictions against actual test and naive random-walk (last value / previous test value).
    """
    pred = pred.reindex(test.index)
    # naive: first prediction = last train price, then previous actual
    naive = pd.Series(index=test.index, dtype=float)
    naive.iloc[0] = train_last
    if len(test) > 1:
        naive.iloc[1:] = test.values[:-1]

    err = test - pred
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / test)) * 100

    msfe_model = np.mean((test - pred) ** 2)
    msfe_naive = np.mean((test - naive) ** 2)
    msfe_ratio = msfe_model / msfe_naive if msfe_naive > 0 else np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_%": mape,
        "MSFE_ratio_vs_Naive": msfe_ratio
    }


# -------------------------------
# Model 1: OLS trend on log-price
# -------------------------------
def fit_ols_log_trend(train: pd.Series):
    """
    Fit log(price) ~ a + b * t using numpy polyfit.
    """
    y = np.log(train.values)
    t = np.arange(len(train))
    b, a = np.polyfit(t, y, 1)  # slope b, intercept a (polyfit returns [slope, intercept] for deg=1)
    return a, b

def predict_ols(train: pd.Series, test_len: int, horizon: int) -> (pd.Series, pd.Series):
    """
    Predict test and H-step forecast using OLS trend on log-price.
    """
    a, b = fit_ols_log_trend(train)
    # Predict test period
    t_test = np.arange(len(train), len(train) + test_len)
    log_pred_test = a + b * t_test
    pred_test = pd.Series(np.exp(log_pred_test), index=pd.date_range(train.index[-1] + timedelta(days=1), periods=test_len, freq="D"))

    # H-step forecast after full series
    t_fore = np.arange(len(train) + test_len, len(train) + test_len + horizon)
    log_fore = a + b * t_fore
    fore = pd.Series(np.exp(log_fore), index=pd.date_range(pred_test.index[-1] + timedelta(days=1), periods=horizon, freq="D"))

    return pred_test, fore


# -------------------------------
# Model 2: ARIMA on price (grid-search)
# -------------------------------
def select_arima_order(train: pd.Series, max_p: int = 3, max_q: int = 3, d_candidates=(0,1)) -> tuple:
    """
    Simple grid-search over (p,d,q) minimizing AIC.
    """
    best_order = None
    best_aic = np.inf
    for d in d_candidates:
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    res = model.fit()
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    if best_order is None:
        # fallback
        best_order = (1, 1, 1)
    return best_order

def predict_arima(train: pd.Series, test_len: int, horizon: int) -> (pd.Series, pd.Series, tuple):
    order = select_arima_order(train)
    model = ARIMA(train, order=order)
    res = model.fit()
    # Out-of-sample predictions for test
    test_fc = res.get_forecast(steps=test_len)
    pred_test = pd.Series(test_fc.predicted_mean, index=pd.date_range(train.index[-1] + timedelta(days=1), periods=test_len, freq="D"))
    # H-step forecasts
    h_fc = res.get_forecast(steps=horizon)
    fore = pd.Series(h_fc.predicted_mean, index=pd.date_range(pred_test.index[-1] + timedelta(days=1), periods=horizon, freq="D"))
    return pred_test, fore, order


# -------------------------------
# Model 3: GARCH(1,1) on returns
# -------------------------------
def predict_garch(train: pd.Series, test_len: int, horizon: int) -> (pd.Series, pd.Series, pd.Series):
    """
    Fit GARCH(1,1) on daily returns with mean ARX(1).
    Produce price predictions by compounding predicted returns.
    Also return volatility forecast (annualized) series for test+H.
    """
    rets_train = train.pct_change().dropna()
    if len(rets_train) < 50:
        raise ValueError("Not enough data points for GARCH. Increase lookback years or reduce test size.")

    am = arch_model(rets_train * 100.0, vol='GARCH', p=1, q=1, mean='ARX', lags=1, dist='normal')  # returns in %
    res = am.fit(disp='off')

    steps = test_len + horizon
    fc = res.forecast(horizon=steps, reindex=False)

    # Mean returns forecast (%), convert to decimal
    mean_fc_pct = pd.Series(fc.mean.values[-1], index=pd.date_range(train.index[-1] + timedelta(days=1), periods=steps, freq="D"))
    mean_fc = mean_fc_pct / 100.0

    # Volatility forecast (sigma of returns in %); convert to decimal and annualize (sqrt(252))
    vol_fc_pct = pd.Series(np.sqrt(fc.variance.values[-1]), index=mean_fc.index)  # % units
    vol_fc_daily = vol_fc_pct / 100.0
    vol_fc_annualized = vol_fc_daily * np.sqrt(252)

    # Price path: start from last train price
    start_price = train.iloc[-1]
    price_path = start_price * (1.0 + mean_fc).cumprod()

    pred_test = price_path.iloc[:test_len]
    fore = price_path.iloc[test_len:]
    return pred_test, fore, vol_fc_annualized


# -------------------------------
# Plotting
# -------------------------------
def make_comparison_plot(full: pd.Series,
                         train: pd.Series,
                         test: pd.Series,
                         ols_test: pd.Series, arima_test: pd.Series, garch_test: pd.Series,
                         ols_fore: pd.Series, arima_fore: pd.Series, garch_fore: pd.Series):
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(full.index, full.values, label="Actual", color="tab:blue", linewidth=1.8)
    ax.axvline(train.index[-1], color="k", linestyle="--", linewidth=1.2, label="Train/Test Split")

    # Test predictions
    ax.plot(ols_test.index, ols_test.values, label="OLS Test", color="tab:orange", linewidth=1.6)
    ax.plot(arima_test.index, arima_test.values, label="ARIMA Test", color="tab:green", linewidth=1.6)
    ax.plot(garch_test.index, garch_test.values, label="GARCH Test", color="tab:red", linewidth=1.6)

    # H-step forecast
    ax.plot(ols_fore.index, ols_fore.values, label="OLS Forecast", color="tab:orange", linestyle="--")
    ax.plot(arima_fore.index, arima_fore.values, label="ARIMA Forecast", color="tab:green", linestyle="--")
    ax.plot(garch_fore.index, garch_fore.values, label="GARCH Forecast", color="tab:red", linestyle="--")

    ax.set_title("Crypto Price Predictions: OLS vs ARIMA vs GARCH", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# -------------------------------
# Core pipeline
# -------------------------------
def run_pipeline(ticker: str, years: int, test_size: int, horizon: int):
    series = get_crypto_data(ticker, years=years)
    if len(series) < (test_size + 60):
        raise ValueError("Series too short. Increase lookback years or reduce test size.")

    train, test = train_test_split(series, test_size)

    # OLS
    ols_test, ols_fore = predict_ols(train, test_len=len(test), horizon=horizon)
    # ARIMA
    arima_test, arima_fore, arima_order = predict_arima(train, test_len=len(test), horizon=horizon)
    # GARCH
    garch_test, garch_fore, garch_vol_annual = predict_garch(train, test_len=len(test), horizon=horizon)

    # Metrics
    m_ols = metrics_vs_naive(test, ols_test, train.iloc[-1])
    m_arima = metrics_vs_naive(test, arima_test, train.iloc[-1])
    m_garch = metrics_vs_naive(test, garch_test, train.iloc[-1])

    # Latest & forecast endpoints
    latest_price = series.iloc[-1]
    arima_price_after_h = float(arima_fore.iloc[-1])
    ols_price_after_h = float(ols_fore.iloc[-1])
    garch_price_after_h = float(garch_fore.iloc[-1])

    # Figure
    fig = make_comparison_plot(series, train, test, ols_test, arima_test, garch_test, ols_fore, arima_fore, garch_fore)

    return {
        "series": series,
        "train": train,
        "test": test,
        "ols_test": ols_test,
        "arima_test": arima_test,
        "garch_test": garch_test,
        "ols_fore": ols_fore,
        "arima_fore": arima_fore,
        "garch_fore": garch_fore,
        "garch_vol_annual": garch_vol_annual,
        "metrics": {
            "OLS": m_ols,
            "ARIMA": m_arima,
            "GARCH": m_garch
        },
        "arima_order": arima_order,
        "latest_price": latest_price,
        "price_after_h": {
            "OLS": ols_price_after_h,
            "ARIMA": arima_price_after_h,
            "GARCH": garch_price_after_h
        },
        "fig": fig
    }


# -------------------------------
# Streamlit UI
# -------------------------------
def run_streamlit():
    st.set_page_config(page_title="Crypto Forecasting: OLS vs ARIMA vs GARCH", layout="wide")

    # Sidebar controls
    st.sidebar.header("Model Parameters")
    ticker = st.sidebar.selectbox("Cryptocurrency symbol", CRYPTO_TICKERS, index=0)
    years = st.sidebar.slider("Lookback (years)", 1, 5, 2, help="Historical window used to train models.")
    test_size = st.sidebar.slider("Test size (days)", 15, 120, 30, help="Out-of-sample window for comparison.")
    horizon = st.sidebar.slider("Forecast horizon H (days ahead)", 5, 60, 15)

    run_button = st.sidebar.button("Predict", type="primary")

    st.title("ARIMA Forecasting Model (with OLS & GARCH Comparison)")
    st.caption("Daily close prices from Yahoo Finance; models evaluated on a recent out-of-sample window.")

    if not run_button:
        st.info("Choose parameters in the sidebar and click **Predict**.")
        return

    with st.spinner("Fetching data and fitting models..."):
        try:
            out = run_pipeline(ticker, years, test_size, horizon)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # Metrics header cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Close Price", f"${out['latest_price']:,.2f}")
    col2.metric(f"Price After {horizon} Days (ARIMA)", f"${out['price_after_h']['ARIMA']:,.2f}")
    col3.metric(f"Price After {horizon} Days (OLS)", f"${out['price_after_h']['OLS']:,.2f}")
    col4.metric(f"Price After {horizon} Days (GARCH)", f"${out['price_after_h']['GARCH']:,.2f}")

    # Chart
    st.pyplot(out["fig"])

    # Show ARIMA order
    st.write(f"**Selected ARIMA order** for {ticker}: {out['arima_order']} (chosen by AIC grid-search)")

    # Metrics table
    st.subheader("Model Performance on Test Window")
    metric_df = pd.DataFrame(out["metrics"]).T
