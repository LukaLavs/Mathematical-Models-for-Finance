import numpy as np
import pandas as pd

def from_log_to_linear(m, S, df_weekly_prices, horizon, tau):
    # tau tells how spaced are the data 7 for weekly, 1 for daily, ...
    # Project the distribution of the weekly logarithmic returns to the one year investment horizon
    # Using formula parameter_h = h/tau * parameter_tau, 5/7 because out of every 7 days 5 are trading days
    m_log = np.floor(5/7 * horizon/tau) * m
    S_log = np.floor(5/7 * horizon/tau) * S

    # Derive the distribution of security prices at the investment horizon
    # Using formula P_h = p0 * exp(R^log)
    p_0 = df_weekly_prices.iloc[0].to_numpy()
    m_P = p_0 * np.exp(m_log + 1/2*np.diag(S_log))
    S_P = np.outer(m_P, m_P) * (np.exp(S_log) - 1)
    
    # Then the estimated moments of the linear return are easy to get
    m = 1 / p_0 * m_P - 1
    S = 1 / np.outer(p_0, p_0) * S_P
    
    return m, S


def shrinkage_metod(df_weekly_log_returns, df_weekly_prices, horizon, tau=1):
    
    # Fit the weekly logarithmic returns to a multivariate normal distribution
    return_array = df_weekly_log_returns.to_numpy()
    m_weekly_log = np.mean(return_array, axis=0)
    S_weekly_log = np.cov(return_array.transpose())
    
    # Prepare for Ledoit-Wolf shrinkage
    def alpha_numerator(Z, S):
        s = 0
        T = Z.shape[1]
        for k in range(T):
            z = Z[:, k][:, np.newaxis]
            X = z @ z.T - S
            s += np.trace(X @ X)
        s /= (T**2)
        return s

    # Apply Ledoit-Wolf shrinkage with the target
    N = S_weekly_log.shape[0]
    T = return_array.shape[0]

    # Ledoit--Wolf shrinkage
    S = S_weekly_log
    s2_avg = np.trace(S) / N
    B = s2_avg * np.eye(N)
    Z = return_array.T - m_weekly_log[:, np.newaxis]
    alpha_num = alpha_numerator(Z, S)
    alpha_den = np.trace((S - B) @ (S - B))
    alpha = alpha_num / alpha_den
    S_shrunk = (1 - alpha) * S + alpha * B
    
    # Implement the James–Stein estimator for the expected return vector
    # James--Stein estimator
    m = m_weekly_log[:, np.newaxis]
    o = np.ones(N)[:, np.newaxis]
    S = S_shrunk
    iS = np.linalg.inv(S)
    b = (o.T @ m / N) * o
    N_eff = np.trace(S) / np.max(np.linalg.eigvalsh(S))
    alpha_num = max(N_eff - 3, 0)
    alpha_den = T * (m - b).T @ iS @ (m - b)
    alpha = alpha_num / alpha_den
    m_shrunk = b + max(1 - alpha, 0) * (m - b)
    m_shrunk = m_shrunk[:, 0]
    
    # Convert
    m_shrunk, S_shrunk = from_log_to_linear(m_shrunk, S_shrunk, df_weekly_prices, horizon, tau)
    
    return m_shrunk, S_shrunk


def james_stein_estimator(df_weekly_log_returns, df_weekly_prices, horizon, tau=1):
    # Fit the weekly logarithmic returns to a multivariate normal distribution
    return_array = df_weekly_log_returns.to_numpy()
    m_weekly_log = np.mean(return_array, axis=0)
    S_weekly_log = np.cov(return_array.transpose())
    
    N = S_weekly_log.shape[0]
    T = return_array.shape[0]
    
    S = S_weekly_log
    m = m_weekly_log
    
    b = np.sum(m) / N
    iS = np.linalg.inv(S)
    
    alpha_den = T * (m - b).T @ iS @ (m - b)
    alpha_nom = np.linalg.trace(S)/np.max(np.linalg.svd(S, compute_uv=False)) - 2
    alpha = alpha_nom / alpha_den
    
    m_log =  alpha * b + (1 - alpha) * m
    # This was for log returns, now we convert to linear returns
    
    # Convert, note: no shrinkage transformations on S were performed
    m_shrunk, S_shrunk = from_log_to_linear(m_log, S, df_weekly_prices, horizon, tau)
    
    return m_shrunk, S_shrunk