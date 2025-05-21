import numpy as np
from scipy.stats import norm
from math import exp, sqrt, factorial, log

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes European option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def merton_jump_diffusion_european_option_price(S, K, T, r, sigma, 
                                                lam, mu_j, sigma_j,
                                                option_type='call', N=50):
    """
    Merton Jump Diffusion European Option price (call or put)
    
    Parameters:
    - S: spot price
    - K: strike price
    - T: time to maturity
    - r: risk-free rate
    - sigma: volatility of underlying asset
    - lam: jump intensity (average number of jumps per year)
    - mu_j: mean of jump size (log-normal jump distribution)
    - sigma_j: std dev of jump size (log-normal jump distribution)
    - option_type: 'call' or 'put'
    - N: number of terms in sum
    
    Returns:
    - option price
    """
    
    # Adjusted risk-free rate with jump risk
    k = exp(mu_j + 0.5 * sigma_j ** 2) - 1
    r_adj = r - lam * k
    
    price = 0.0
    for n in range(N):
        # Probability of n jumps in time T (Poisson)
        p_n = exp(-lam * T) * (lam * T) ** n / factorial(n)
        
        # Adjust volatility and drift for n jumps
        sigma_n = sqrt(sigma ** 2 + (n * sigma_j ** 2) / T)
        r_n = r_adj + (n * mu_j) / T
        
        # Price with adjusted parameters in Black-Scholes formula
        price += p_n * black_scholes_price(S, K, T, r_n, sigma_n, option_type)
    
    return price


# Example usage
S = 100       # Current price
K = 100       # Strike price
T = 1.0       # Time to maturity (1 year)
r = 0.05      # Risk-free rate (5%)
sigma = 0.2   # Volatility (20%)

lam = 0.01     # Jump intensity (0.1 jumps per year)
mu_j = -0.1   # Mean jump size (log scale, e.g., -10%)
sigma_j = 0.3 # Jump size volatility

call_price = merton_jump_diffusion_european_option_price(S, K, T, r, sigma, lam, mu_j, sigma_j, 'call')
put_price = merton_jump_diffusion_european_option_price(S, K, T, r, sigma, lam, mu_j, sigma_j, 'put')

print(f"Call option price: {call_price:.4f}")
print(f"Put option price: {put_price:.4f}")
print(f"Black sholes call",  black_scholes_price(S, K, T, r, sigma, option_type='call'))
print(f"Black sholes call",  black_scholes_price(S, K, T, r, sigma, option_type='put'))