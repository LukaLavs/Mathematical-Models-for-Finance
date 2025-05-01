import pandas as pd
import numpy as np


def backtest_markowitz(df_prices, model_func, shrinkage_metod, window_size=60, rebalance_freq=20, risk_aversion=0.5):
    """
    df_prices: DataFrame with datetime index and columns as stock prices.
    model_func: function that takes (S, m, delta) and returns (weights, expected return, risk)
    window_size: how many past days to use for estimation (e.g., 60)
    rebalance_freq: how often to rebalance (e.g., every 20 days)
    risk_aversion: delta parameter passed to the model
    """
    df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
    
    returns = df_prices.pct_change().dropna()
    dates = returns.index

    portfolio_returns = []
    portfolio_weights = []
    backtest_dates = []

    for i in range(window_size, len(returns), rebalance_freq):
        
        window_log_returns = df_log_returns.iloc[i - window_size:i]
        window_prices = df_prices.iloc[i - window_size:i]
        
        mu, Sigma = shrinkage_metod(window_log_returns, window_prices)
 
        # Optimize portfolio
        weights, _, _ = model_func(Sigma, mu, risk_aversion)

        # Save weights
        portfolio_weights.append(weights)
        backtest_dates.append(dates[i])

        # Apply weights to next return period (could be 1 day, or next rebalance window)
        next_returns = returns.iloc[i:i + rebalance_freq].values
        for daily_ret in next_returns:
            port_ret = np.dot(weights, daily_ret)
            portfolio_returns.append(port_ret)

    # Build performance DataFrame
    perf = pd.DataFrame({
        'date': dates[window_size:window_size + len(portfolio_returns)],
        'portfolio_return': portfolio_returns
    }).set_index('date')
    perf['cumulative_return'] = (1 + perf['portfolio_return']).cumprod()

    return perf, portfolio_weights, backtest_dates
