import numpy as np
import pandas as pd

def prepare_data(df_prices, weekly=False):
    """ Returns weekly log returns and weekly prices as seperate data frames """
    # Resample to weekly frequency (take the last available price each week)
    if weekly:
        df_prices = df_prices.resample('W').last()

    # Calculate weekly logarithmic returns  
    df_log_returns = \
        np.log(df_prices) - np.log(df_prices.shift(1))

    # Handle missing values: Drop rows where all values are NaN, and fill remaining NaN with 0
    df_log_returns = df_log_returns.dropna(how='all')
    df_log_returns = df_log_returns.fillna(0)
    
    return df_log_returns, df_prices