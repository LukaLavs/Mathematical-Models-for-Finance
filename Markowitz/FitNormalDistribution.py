import numpy as np
import pandas as pd


def fit_normal_distribution(df_weekly_log_returns, df_weekly_prices):
    """ Finds mu and Sigma for linear returns """
    # Fit the weekly logarithmic returns to a multivariate normal distribution
    return_array = df_weekly_log_returns.to_numpy()
    m_weekly_log = np.mean(return_array, axis=0)
    S_weekly_log = np.cov(return_array.transpose())
    
    # Project the distribution of the weekly logarithmic returns to the one year investment horizon
    # Using formula parameter_h = h/tau * parameter_tau
    m_log = 52 * m_weekly_log
    S_log = 52 * S_weekly_log
    
    # Derive the distribution of security prices at the investment horizon
    # Using formula P_h = p0 * exp(R^log)
    p_0 = df_weekly_prices.iloc[0].to_numpy()
    m_P = p_0 * np.exp(m_log + 1/2*np.diag(S_log))
    S_P = np.outer(m_P, m_P) * (np.exp(S_log) - 1)
    
    # Then the estimated moments of the linear return are easy to get
    m = 1 / p_0 * m_P - 1
    S = 1 / np.outer(p_0, p_0) * S_P
    
    return m, S