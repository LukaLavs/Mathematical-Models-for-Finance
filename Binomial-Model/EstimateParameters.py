import numpy as np
import pandas as pd

def estimate_volatility(df_prices):
    # Assuming df_prices is in form:
    # df_prices = pd.DataFrame(prices, index=dates, columns=stock_names)

    # Calculate the daily returns (log returns)
    log_returns = np.log(df_prices / df_prices.shift(1))

    # Calculate the daily standard deviation (volatility)
    daily_volatility = log_returns.std()

    # Annualize the volatility by multiplying by sqrt(252)
    annualized_volatility = daily_volatility * np.sqrt(252)

    return annualized_volatility