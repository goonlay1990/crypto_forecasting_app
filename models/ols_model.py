import pandas as pd
import statsmodels.api as sm

def run_ols(data, target_col='Close'):
    """
    Runs OLS regression on time series data.
    Args:
        data (pd.DataFrame): DataFrame with 'Date' and target column.
        target_col (str): Column name for dependent variable.
    Returns:
        model_summary: OLS model summary
    """
    # Prepare data
    data = data.copy()
    data['t'] = range(len(data))  # time index
    X = sm.add_constant(data['t'])  # independent variable
    y = data[target_col]

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    return model.summary()
