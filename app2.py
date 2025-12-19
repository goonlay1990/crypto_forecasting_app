
# ============================================================
# Streamlit + Plotly App: Cryptocurrency Forecasting
# Models: OLS (AR on log-returns), ARIMA, GARCH(1,1)
# Includes confidence bands, GARCH volatility envelope, DM test,
# CSV downloads, and custom CSS for a polished UI.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import math

# Statsmodels / sklearn
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Plotly
import plotly.graph_objects as go

# ARCH (for GARCH)
HAVE_ARCH = True
try:
    from arch import arch_model
except Exception:
    HAVE_ARCH = False

# -----------------------------
# Page config & Custom CSS
# -----------------------------
st.set_page_config(page_title="Cryptocurrency Forecasting", layout="wide")

CUSTOM_CSS = """
https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap
<style>
    :root {
        --bg: #0f172a;            /* slate-900 */
        --panel: #111827;         /* gray-900 */
        --card: #0b1220;          /* dark navy */
        --accent: #4f46e5;        /* indigo-600 */
        --accent2: #22c55e;       /* green-500 */
        --accent3: #ef4444;       /* red-500 */
        --accent4: #f59e0b;       /* amber-500 */
        --muted: #94a3b8;         /* slate-400 */
        --text: #e5e7eb;          /* gray-200 */
    }
    .block-container { padding-top: 0.8rem !important; }
    html, body, [class*="main"] {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 60%, #0f172a 100%) !important;
        color: var(--text);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, sans-serif;
    }
    .app-header {
        background: linear-gradient(90deg, rgba(79,70,229,0.25), rgba(34,197,94,0.25));
        border: 1px solid rgba(148,163,184,0.25);
        padding: 16px 22px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.25);
    }
    .card {
        background: var(--card);
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin-bottom: 16px;
    }
    .card h3 {
        margin: 0 0 10px 0;
        font-weight: 700;
        color: var(--text);
        letter-spacing: 0.3px;
    }
    .small-text { color: var(--muted); font-size: 0.92rem; }
    .footer-note { color: var(--muted); font-size: 0.85rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h2 style="margin:0;font-weight:700;">Cryptocurrency Forecasting</h2>
  <div class="small-text">Interactive OLS (AR), ARIMA, and GARCH models with confidence bands & volatility</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("### Select Cryptocurrency:")
CRYPTO_OPTIONS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "Polkadot (DOT)": "DOT-USD",
}
crypto_name = st.sidebar.selectbox("", list(CRYPTO_OPTIONS.keys()), index=4)
symbol = CRYPTO_OPTIONS[crypto_name]

st.sidebar.markdown("### Start Date")
start_date = st.sidebar.date_input(" ", value=date(2022,1,1))
st.sidebar.markdown("### End Date")
end_date   = st.sidebar.date_input("  ", value=date(2023,12,31))
st.sidebar.markdown("### Forecast Horizon (Days):")
H = st.sidebar.number_input("   ", min_value=1, max_value=120, value=30, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Advanced Settings**")
train_ratio = st.sidebar.slider("Train ratio (%)", 60, 90, 80, 5)
pmax_ols   = st.sidebar.slider("OLS AR(p) max", 1, 12, 6)
pmax_arima = st.sidebar.slider("ARIMA max p", 0, 6, 4)
dmax_arima = st.sidebar.slider("ARIMA max d", 0, 3, 2)
qmax_arima = st.sidebar.slider("ARIMA max q", 0, 6, 4)

st.sidebar.markdown("---")
st.sidebar.markdown("**Confidence Bands**")
show_bands = st.sidebar.checkbox("Show confidence bands", value=True)
conf_level = st.sidebar.selectbox("Band level", options=[0.90, 0.95, 0.99], index=1)

st.sidebar.markdown("---")
show_ols   = st.sidebar.checkbox("Show OLS",   value=True)
show_arima = st.sidebar.checkbox("Show ARIMA", value=True)
show_garch = st.sidebar.checkbox("Show GARCH", value=True)

run_btn = st.sidebar.button("Run Forecast")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_data(sym: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(sym, start=pd.to_datetime(start), end=pd.to_datetime(end), interval="1d")
    return df

def prepare_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        raise ValueError("No data returned. Try a wider date range or another symbol.")
    s = df[['Close']].dropna()
    s.index = pd.to_datetime(s.index)
    s = s.asfreq('D').ffill()
    return s['Close']

def make_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()

def returns_to_prices(start_price: float, returns_array: np.ndarray) -> np.ndarray:
    return start_price * np.exp(np.cumsum(returns_array))

def z_for_conf(conf: float) -> float:
    # simple mapping without scipy
    if abs(conf - 0.90) < 1e-9: return 1.645
    if abs(conf - 0.95) < 1e-9: return 1.960
    if abs(conf - 0.99) < 1e-9: return 2.576
    return 1.960

# ---------- OLS (AR on returns) ----------
def fit_ols_ar(train_returns: pd.Series, p: int):
    y = train_returns[p:]
    X = pd.concat([train_returns.shift(i) for i in range(1, p+1)], axis=1)[p:]
    X.columns = [f"lag{i}" for i in range(1, p+1)]
    X = sm.add_constant(X)
    model = sm.OLS(y.values, X.values)
    fit = model.fit()
    return fit.params, fit

def recursive_forecast_returns(params: np.ndarray, init_returns: np.ndarray, steps: int) -> np.ndarray:
    const = params[0]; betas = params[1:]; p = len(betas)
    hist = list(init_returns[:p])
    preds = []
    for _ in range(steps):
        r_hat = const + np.dot(betas, np.array(hist[:p]))
        preds.append(r_hat)
        hist = [r_hat] + hist[:p-1]
    return np.array(preds)

def grid_search_ols(train_returns: pd.Series, test_returns: pd.Series,
                    start_price_for_test: float, p_max: int):
    results = []
    for p in range(1, p_max+1):
        if len(train_returns) <= p:
            continue
        params, fit = fit_ols_ar(train_returns, p)
        init = train_returns.iloc[-p:][::-1].values
        pred_r = recursive_forecast_returns(params, init, steps=len(test_returns))
        pred_prices = returns_to_prices(start_price_for_test, pred_r)
        actual_prices = returns_to_prices(start_price_for_test, test_returns.values)
        mse = mean_squared_error(actual_prices, pred_prices)
        mae = mean_absolute_error(actual_prices, pred_prices)
        results.append((p, mse, mae, params, fit, pred_prices))
    if not results:
        raise ValueError("Not enough data for OLS AR(p). Reduce p_max or expand dates.")
    best = min(results, key=lambda x: x[1])
    best_p, best_mse, best_mae, best_params, best_fit, best_test_pred_prices = best
    return best_p, best_mse, best_mae, best_params, best_fit, best_test_pred_prices

def bootstrap_ols_future_ci(params, init_returns, resid, H, start_price, conf=0.95, M=400):
    """Bootstrap future path by resampling residuals; return (lower, upper) price bands."""
    p = len(params) - 1
    betas = params[1:]; const = params[0]
    paths = []
    for _ in range(M):
        hist = list(init_returns[:p])
        rets = []
        for h in range(H):
            eps = np.random.choice(resid)
            r_hat = const + np.dot(betas, np.array(hist[:p])) + eps
            rets.append(r_hat)
            hist = [r_hat] + hist[:p-1]
        paths.append(returns_to_prices(start_price, np.array(rets)))
    arr = np.stack(paths)                 # M x H
    lower = np.percentile(arr, (1-conf)*100, axis=0)
    upper = np.percentile(arr, conf*100,  axis=0)
    return lower, upper

# ---------- ARIMA ----------
def try_arima(train_close: pd.Series, order):
    try:
        return ARIMA(train_close, order=order,
                     enforce_stationarity=False, enforce_invertibility=False).fit()
    except Exception:
        return None

def grid_search_arima(train_close: pd.Series, test_close: pd.Series,
                      p_max=4, d_max=2, q_max=4):
    results = []
    for p in range(p_max+1):
        for d in range(d_max+1):
            for q in range(q_max+1):
                fit = try_arima(train_close, (p,d,q))
                if fit is None:
                    results.append(((p,d,q), np.inf, np.inf, None, None))
                else:
                    pred = fit.forecast(steps=len(test_close))
                    mse = mean_squared_error(test_close, pred)
                    mae = mean_absolute_error(test_close, pred)
                    results.append(((p,d,q), mse, mae, fit, pred))
    best_order, best_mse, best_mae, best_fit, best_pred = min(results, key=lambda x: x[1])
    return best_order, best_mse, best_mae, best_fit, best_pred

# ---------- GARCH ----------
def garch_forecast(train_returns: pd.Series, test_returns: pd.Series,
                   start_price_for_test: float, horizon_future: int):
    if not HAVE_ARCH:
        raise ImportError("arch not installed.")
    am = arch_model(train_returns, mean='AR', lags=1, vol='GARCH', p=1, q=1, dist='normal')
    res = am.fit(disp='off')

    params = res.params
    mu = float(params.get('mu', 0.0))
    phi = float(params.get('ar.L1', 0.0))

    # last observed return
    last_r = float(train_returns.iloc[-1])

    # ---- Test horizon mean path
    test_pred_returns = []
    prev_r = last_r
    for _ in range(len(test_returns)):
        r_hat = mu + phi * prev_r
        test_pred_returns.append(r_hat)
        prev_r = r_hat
    test_pred_prices = returns_to_prices(start_price_for_test, np.array(test_pred_returns))

    # ---- Future horizon: mean returns + volatility sigma_t
    # Use res.forecast to get forward variance; fallback to recursion if unavailable.
    try:
        fcs = res.forecast(horizon_future)
        future_sigma = np.sqrt(fcs.variance.values[-1])
    except Exception:
        # Simple GARCH recursion: sigma_{t+1}^2 = omega + (alpha+beta)*sigma_t^2 (E[eps^2] = sigma_t^2)
        omega = float(params.get('omega', 0.0))
        alpha = float(params.get('alpha[1]', 0.0))
        beta  = float(params.get('beta[1]', 0.0))
        sigma2 = float(res.conditional_volatility.iloc[-1]**2)
        future_sigma = []
        for _ in range(horizon_future):
            sigma2 = omega + (alpha + beta) * sigma2
            future_sigma.append(np.sqrt(sigma2))
        future_sigma = np.array(future_sigma)

    # future mean returns sequence
    future_mean_returns = []
    prev_r = last_r
    for _ in range(horizon_future):
        r_hat = mu + phi * prev_r
        future_mean_returns.append(r_hat)
        prev_r = r_hat
    future_mean_returns = np.array(future_mean_returns)

    future_prices = returns_to_prices(start_price=start_price_for_test, returns_array=future_mean_returns)
    return res, np.array(test_pred_prices), future_prices, future_sigma, future_mean_returns

# ---------- DM Test ----------
def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2):
    """
    Diebold–Mariano test using Newey–West long-run variance.
    e1, e2: forecast errors (e.g., actual - predicted)
    h: forecast horizon (1 for one-step ahead)
    power: 1 (MAE) or 2 (MSE) loss
    Returns (DM_stat, p_value)
    """
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1**2) - (e2**2)
    n = len(d)
    dmean = d.mean()

    # Newey-West long-run variance with lag = h-1
    L = max(h - 1, 0)
    d_centered = d - dmean
    gamma0 = np.sum(d_centered * d_centered) / n
    lrvar = gamma0
    for i in range(1, L + 1):
        gammai = np.sum(d_centered[i:] * d_centered[:-i]) / n
        lrvar += 2.0 * (1.0 - i / (L + 1.0)) * gammai
    dm_stat = dmean / math.sqrt(lrvar / n)
    p_value = 2.0 * (1.0 - _norm_cdf(abs(dm_stat)))
    return dm_stat, p_value

# -----------------------------
# Main
# -----------------------------
if run_btn:
    try:
        # Load data
        df = load_data(symbol, start_date, end_date)
        close = prepare_close(df)
        returns = make_log_returns(close)
        latest_close = float(close.iloc[-1])

        if len(close) < 60:
            st.error("Not enough data in the selected range. Choose a longer period.")
            st.stop()

        # Train/Test split
        train_size = int(len(close) * (train_ratio / 100))
        train_close = close.iloc[:train_size]
        test_close  = close.iloc[train_size:]
        train_returns = returns.iloc[:train_size-1]
        test_returns  = returns.iloc[train_size-1:]
        start_price_for_test = float(train_close.iloc[-1])

        # Layout
        left, right = st.columns([3, 2])

        # =======================
        # OLS
        # =======================
        if show_ols:
            with st.spinner("Fitting OLS AR(p) on log-returns..."):
                best_p, ols_mse, ols_mae, ols_params, ols_fit, ols_test_pred_prices = grid_search_ols(
                    train_returns, test_returns, start_price_for_test, p_max=pmax_ols
                )
                # Future mean path
                params_full, fit_full = fit_ols_ar(returns, best_p)
                init_full = returns.iloc[-best_p:][::-1].values
                future_r_ols = recursive_forecast_returns(params_full, init_full, H)
                future_p_ols = returns_to_prices(latest_close, future_r_ols)
                # Bootstrap CI
                if show_bands:
                    ols_lower, ols_upper = bootstrap_ols_future_ci(
                        params_full, init_full, fit_full.resid, H, latest_close, conf=conf_level, M=400
                    )

        # =======================
        # ARIMA
        # =======================
        if show_arima:
            with st.spinner("Running ARIMA grid search..."):
                best_order, arima_mse, arima_mae, arima_fit, arima_test_pred = grid_search_arima(
                    train_close, test_close, p_max=pmax_arima, d_max=dmax_arima, q_max=qmax_arima
                )
                final_arima = ARIMA(close, order=best_order,
                                    enforce_stationarity=False, enforce_invertibility=False).fit()
                future_arima_obj = final_arima.get_forecast(steps=H)
                future_p_arima = future_arima_obj.predicted_mean
                # CI (statsmodels)
                if show_bands:
                    ci_df = future_arima_obj.conf_int()
                    # robust pick for columns: 'lower Close'/'upper Close' or generic first/second
                    lower_col = [c for c in ci_df.columns if 'lower' in c.lower()][0]
                    upper_col = [c for c in ci_df.columns if 'upper' in c.lower()][0]
                    arima_lower = ci_df[lower_col]
                    arima_upper = ci_df[upper_col]

        # =======================
        # GARCH
        # =======================
        if show_garch and not HAVE_ARCH:
            st.warning("GARCH requires the `arch` package. Install with: pip install arch")

        if show_garch and HAVE_ARCH:
            with st.spinner("Estimating GARCH(1,1) volatility (mean AR(1))..."):
                garch_res, garch_test_pred_prices, garch_future_prices, future_sigma, future_mean_returns = garch_forecast(
                    train_returns, test_returns, start_price_for_test, H
                )
                garch_mse = mean_squared_error(test_close.values, garch_test_pred_prices)
                garch_mae = mean_absolute_error(test_close.values, garch_test_pred_prices)
                # Volatility band around mean path (log-return ± z*sigma)
                if show_bands:
                    z = z_for_conf(conf_level)
                    r_upper = future_mean_returns + z * future_sigma
                    r_lower = future_mean_returns - z * future_sigma
                    garch_upper_prices = returns_to_prices(latest_close, r_upper)
                    garch_lower_prices = returns_to_prices(latest_close, r_lower)

        # =======================
        # Price & Forecast (Plotly)
        # =======================
        with left:
            st.markdown('<div class="card"><h3>Price and Forecast</h3>', unsafe_allow_html=True)

            future_index = pd.date_range(start=close.index[-1], periods=H+1, freq='D')[1:]

            fig_pf = go.Figure()
            # Actual
            fig_pf.add_trace(go.Scatter(x=close.index, y=close.values, name="Actual",
                                        mode="lines", line=dict(color="#1f77b4", width=2)))

            # OLS test + future
            if show_ols:
                fig_pf.add_trace(go.Scatter(x=test_close.index, y=ols_test_pred_prices,
                                            name="OLS Forecast", mode="lines",
                                            line=dict(color="#ef4444", width=2)))
                fig_pf.add_trace(go.Scatter(x=future_index, y=future_p_ols,
                                            name=f"OLS Future (+{H}d)", mode="lines",
                                            line=dict(color="#ef4444", width=2, dash="dash")))
                if show_bands:
                    # OLS band (fill between upper and lower)
                    fig_pf.add_trace(go.Scatter(x=future_index, y=ols_upper,
                                                name="OLS Upper", mode="lines",
                                                line=dict(color="rgba(239,68,68,0.0)"),
                                                showlegend=False))
                    fig_pf.add_trace(go.Scatter(x=future_index, y=ols_lower,
                                                name=f"OLS {int(conf_level*100)}% Band",
                                                mode="lines", fill='tonexty',
                                                line=dict(color="rgba(239,68,68,0.0)"),
                                                fillcolor="rgba(239,68,68,0.18)"))

            # ARIMA test + future
            if show_arima:
                fig_pf.add_trace(go.Scatter(x=test_close.index, y=arima_test_pred.values,
                                            name="ARIMA Forecast", mode="lines",
                                            line=dict(color="#22c55e", width=2)))
                fig_pf.add_trace(go.Scatter(x=future_index, y=future_p_arima.values,
                                            name=f"ARIMA Future (+{H}d)", mode="lines",
                                            line=dict(color="#22c55e", width=2, dash="dash")))
                if show_bands:
                    fig_pf.add_trace(go.Scatter(x=future_index, y=arima_upper.values,
                                                name="ARIMA Upper", mode="lines",
                                                line=dict(color="rgba(34,197,94,0.0)"),
                                                showlegend=False))
                    fig_pf.add_trace(go.Scatter(x=future_index, y=arima_lower.values,
                                                name=f"ARIMA {int(conf_level*100)}% Band",
                                                mode="lines", fill='tonexty',
                                                line=dict(color="rgba(34,197,94,0.0)"),
                                                fillcolor="rgba(34,197,94,0.18)"))

            # GARCH test + future
            if show_garch and HAVE_ARCH:
                fig_pf.add_trace(go.Scatter(x=test_close.index, y=garch_test_pred_prices,
                                            name="GARCH Forecast", mode="lines",
                                            line=dict(color="#f59e0b", width=2)))
                fig_pf.add_trace(go.Scatter(x=future_index, y=garch_future_prices,
                                            name=f"GARCH Future (+{H}d)", mode="lines",
                                            line=dict(color="#f59e0b", width=2, dash="dash")))
                if show_bands:
                    fig_pf.add_trace(go.Scatter(x=future_index, y=garch_upper_prices,
                                                name="GARCH Upper", mode="lines",
                                                line=dict(color="rgba(245,158,11,0.0)"),
                                                showlegend=False))
                    fig_pf.add_trace(go.Scatter(x=future_index, y=garch_lower_prices,
                                                name=f"GARCH {int(conf_level*100)}% Band",
                                                mode="lines", fill='tonexty',
                                                line=dict(color="rgba(245,158,11,0.0)"),
                                                fillcolor="rgba(245,158,11,0.18)"))

            fig_pf.update_layout(
                height=430,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                margin=dict(l=20, r=20, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
            )
            st.plotly_chart(fig_pf, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # =======================
        # Evaluation + DM Test
        # =======================
        rmse_ols   = np.sqrt(ols_mse)   if show_ols   else np.nan
        rmse_arima = np.sqrt(arima_mse) if show_arima else np.nan
        rmse_garch = np.sqrt(garch_mse) if (show_garch and HAVE_ARCH) else np.nan
        mae_ols    = ols_mae   if show_ols   else np.nan
        mae_arima  = arima_mae if show_arima else np.nan
        mae_garch  = garch_mae if (show_garch and HAVE_ARCH) else np.nan

        with left:
            st.markdown('<div class="card"><h3>Model Evaluation</h3>', unsafe_allow_html=True)
            eval_df = pd.DataFrame({
                "Model": ["OLS", "ARIMA", "GARCH"],
                "RMSE":  [rmse_ols, rmse_arima, rmse_garch],
                "MAE":   [mae_ols,  mae_arima,  mae_garch]
            }).round(3)
            st.table(eval_df)
            st.markdown('</div>', unsafe_allow_html=True)

            # DM Test on test segment (squared-error loss)
            st.markdown('<div class="card"><h3>Diebold–Mariano (DM) Test</h3>', unsafe_allow_html=True)
            dm_rows = []
            actual = test_close.values

            if show_ols and show_arima:
                e1 = actual - ols_test_pred_prices
                e2 = actual - arima_test_pred.values
                dm_stat, pval = dm_test(e1, e2, h=1, power=2)
                dm_rows.append(["OLS vs ARIMA", dm_stat, pval])

            if show_arima and show_garch and HAVE_ARCH:
                e1 = actual - arima_test_pred.values
                e2 = actual - garch_test_pred_prices
                dm_stat, pval = dm_test(e1, e2, h=1, power=2)
                dm_rows.append(["ARIMA vs GARCH", dm_stat, pval])

            if show_ols and show_garch and HAVE_ARCH:
                e1 = actual - ols_test_pred_prices
                e2 = actual - garch_test_pred_prices
                dm_stat, pval = dm_test(e1, e2, h=1, power=2)
                dm_rows.append(["OLS vs GARCH", dm_stat, pval])

            if dm_rows:
                dm_df = pd.DataFrame(dm_rows, columns=["Comparison", "DM stat", "p-value"]).round(4)
                st.table(dm_df)
            else:
                st.write("Select at least two models to run DM tests.")
            st.markdown('</div>', unsafe_allow_html=True)

        # =======================
        # Volatility Forecast (GARCH σ)
        # =======================
        with right:
            st.markdown('<div class="card"><h3>Volatility Forecast</h3>', unsafe_allow_html=True)
            fig_vol = go.Figure()
            if show_garch and HAVE_ARCH:
                vol_index = pd.date_range(start=close.index[-1], periods=H, freq='D')
                fig_vol.add_trace(go.Scatter(x=vol_index, y=future_sigma, mode="lines",
                                             name="GARCH σ (forward)", line=dict(color="#1f77b4", width=2)))
                fig_vol.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(255,255,255,0.02)",
                                      margin=dict(l=20, r=20, t=10, b=10),
                                      xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                                      yaxis=dict(gridcolor="rgba(255,255,255,0.08)"))
            else:
                fig_vol.add_annotation(text="Install `arch` to enable GARCH volatility forecast.",
                                       showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
                                       font=dict(color="white", size=14))
                fig_vol.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(255,255,255,0.02)")
            st.plotly_chart(fig_vol, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # =======================
        # Downloads
        # =======================
        st.markdown('<div class="card"><h3>Download Forecasts (CSV)</h3>', unsafe_allow_html=True)
        future_index = pd.date_range(start=close.index[-1], periods=H+1, freq='D')[1:]

        if show_ols:
            df_ols = {"date": future_index, "OLS_mean": future_p_ols}
            if show_bands:
                df_ols["OLS_lower"] = ols_lower
                df_ols["OLS_upper"] = ols_upper
            st.download_button("⬇️ OLS Future CSV",
                               pd.DataFrame(df_ols).to_csv(index=False).encode("utf-8"),
                               file_name=f"{symbol.replace('-','_').lower()}_ols_future_{H}d.csv",
                               mime="text/csv")

        if show_arima:
            df_arima = {"date": future_index, "ARIMA_mean": future_p_arima.values}
            if show_bands:
                df_arima["ARIMA_lower"] = arima_lower.values
                df_arima["ARIMA_upper"] = arima_upper.values
            st.download_button("⬇️ ARIMA Future CSV",
                               pd.DataFrame(df_arima).to_csv(index=False).encode("utf-8"),
                               file_name=f"{symbol.replace('-','_').lower()}_arima_future_{H}d.csv",
                               mime="text/csv")

        if show_garch and HAVE_ARCH:
            df_garch = {"date": future_index, "GARCH_mean": garch_future_prices}
            if show_bands:
                df_garch["GARCH_lower"] = garch_lower_prices
                df_garch["GARCH_upper"] = garch_upper_prices
            st.download_button("⬇️ GARCH Future CSV",
                               pd.DataFrame(df_garch).to_csv(index=False).encode("utf-8"),
                               file_name=f"{symbol.replace('-','_').lower()}_garch_future_{H}d.csv",
                               mime="text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
<div class="footer-note">
• OLS: AR(p) on log-returns; CI via bootstrap of residuals.  
• ARIMA: statsmodels forecast with built-in confidence intervals.  
• GARCH: mean AR(1) + volatility σ; band uses mean ± z·σ on returns then compounded to price.  
• DM test: compares squared-error losses across models on the test segment (last 20–40% depending on your slider).  
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    
    st.info("""Choose the cryptocurrency and dates on the left, set the horizon, then click **Run Forecast**.
    Use the checkboxes to show/hide models and bands.""")

