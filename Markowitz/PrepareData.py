import numpy as np
import pandas as pd

def prepare_data(df_prices):
    """ Returns weekly log returns and weekly prices as seperate data frames """
    # Resample to weekly frequency (take the last available price each week)
    df_weekly_prices = df_prices.resample('W').last()

    # Calculate weekly logarithmic returns  
    df_weekly_log_returns = \
        np.log(df_weekly_prices) - np.log(df_weekly_prices.shift(1))

    # Handle missing values: Drop rows where all values are NaN, and fill remaining NaN with 0
    df_weekly_log_returns = df_weekly_log_returns.dropna(how='all')
    df_weekly_log_returns = df_weekly_log_returns.fillna(0)
    
    return df_weekly_log_returns, df_weekly_prices