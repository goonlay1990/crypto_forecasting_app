import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def run_arima_prices(close_series: pd.Series, order=(1, 1, 1), steps=30):
    """
    ARIMA on price levels (can look flat for financial series).
    """
    close_series = close_series.dropna()
    model = ARIMA(close_series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=steps)
    return fit, forecast


def run_arima_returns(close_series: pd.Series, order=(1, 0, 1), steps=30):
    """
    ARIMA on log returns, then converts forecasted returns into a price path.

    Returns:
      fit: fitted ARIMA model (on returns)
      returns_forecast: pd.Series of forecasted log returns
      price_forecast: pd.Series of forecasted prices (level)
    """
    close_series = close_series.dropna()

    log_price = np.log(close_series)
    returns = log_price.diff().dropna()

    model = ARIMA(returns, order=order)
    fit = model.fit()

    returns_forecast = fit.forecast(steps=steps)

    last_price = float(close_series.iloc[-1])
    cum_log_return = np.cumsum(returns_forecast.values)
    price_path = last_price * np.exp(cum_log_return)

    price_forecast = pd.Series(price_path)

    return fit, pd.Series(returns_forecast), price_forecast

