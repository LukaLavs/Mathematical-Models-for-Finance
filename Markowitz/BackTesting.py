import pandas as pd
import numpy as np
import ShrinkageMethods

def backtest_markowitz(df_prices, model_func, model_params, window_size=60, rebalance_freq=20):
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
        
        mu, Sigma = ShrinkageMethods.shrinkage_metod(window_log_returns, window_prices, horizon=window_size, tau=1)
        model_params["m"], model_params["S"] = mu, Sigma
        
        # Optimize portfolio
        weights, _, _ = model_func(**model_params) #Sigma, mu, risk_aversion)

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
    
    final_amount = perf['cumulative_return'].iloc[-1]


    return final_amount, perf, portfolio_weights, backtest_dates


def backtest_random_portfolios(df_prices, num_portfolios=10000):
    """
    Fast simulation of multiple random portfolios over the full period, without rebalancing.
    Returns average final return (portfolio value) starting from 1.0 capital.
    """
    returns = df_prices.pct_change().dropna().values  # Shape: (T, N)
    T, N = returns.shape

    # Generate random weights for all portfolios at once
    random_weights = np.random.rand(num_portfolios, N)
    random_weights /= random_weights.sum(axis=1, keepdims=True)  # Normalize to sum to 1

    # Matrix multiply: (portfolios x stocks) dot (stocks x time) = (portfolios x time)
    # Transpose returns to (N x T) for dot product
    portfolio_returns = random_weights @ returns.T  # Shape: (num_portfolios, T)

    # Cumulative return for each portfolio (simulate investing 1 unit)
    cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)  # Shape: (num_portfolios, T)

    # Extract final portfolio values (last day)
    final_values = cumulative_returns[:, -1]  # Shape: (num_portfolios,)

    # Return average final value
    return final_values.mean()

