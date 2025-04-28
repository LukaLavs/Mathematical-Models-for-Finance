import math
from scipy.stats import norm


def binomial_tree_model(S, K, T, r, sigma, q, n, 
                    model="crr", 
                    option_type="put", 
                    option_style="european"):
    """
    General binomial model for options.

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - q: Dividend yield
    - n: Number of time steps
    - model: 'crr', 'jarrow-rudd', or 'tian'
    - option_type: 'put' or 'call'
    - option_style: 'european' or 'american'
    """
    # Step 1: calculate model-specific constants
    delta_t = T / n
    discount = math.exp(-r * delta_t)
    
    if model.lower() == "crr":
        u = math.exp(sigma * math.sqrt(delta_t))
        d = 1 / u
        p = (math.exp((r - q) * delta_t) - d) / (u - d)
    elif model.lower() == "jarrow-rudd":
        drift = (r - q - 0.5 * sigma**2) * delta_t
        u = math.exp(drift + sigma * math.sqrt(delta_t))
        d = math.exp(drift - sigma * math.sqrt(delta_t))
        p = 0.5
    elif model.lower() == "tian":
        v = math.exp((r - q) * delta_t)
        a = math.exp(sigma**2 * delta_t)
        u = 0.5 * v * a * (a + 1 + math.sqrt(a**2 + 2*a - 3))
        d = 0.5 * v * a * (a + 1 - math.sqrt(a**2 + 2*a - 3))
        p = (v - d) / (u - d)
    else:
        print("Model not supported.")
        return None
    
    # Step 2: initialize asset prices at maturity
    stock_prices = [0] * (n + 1)
    option_values = [0] * (n + 1)

    for i in range(n + 1):
        stock_prices[i] = S * (u ** i) * (d ** (n - i))
        if option_type.lower() == "put":
            option_values[i] = max(K - stock_prices[i], 0)
        elif option_type.lower() == "call":
            option_values[i] = max(stock_prices[i] - K, 0)
        else:
            print("Option type not supported.")
            return None

    # Step 3: move backward through the tree
    for j in range(n-1, -1, -1):
        for i in range(j + 1):
            continuation = discount * (p * option_values[i+1] + (1 - p) * option_values[i])
            if option_style.lower() == "european":
                option_values[i] = continuation
            elif option_style.lower() == "american":
                stock_price = S * (u ** i) * (d ** (j - i))
                if option_type.lower() == "put":
                    exercise = max(K - stock_price, 0)
                else:  # call
                    exercise = max(stock_price - K, 0)
                option_values[i] = max(continuation, exercise)
            else:
                print("Option style not supported.")
                return None

    # Step 4: result
    return option_values[0]



def black_scholes(S, K, T, r, sigma, q, option_type="put"):
    """ 
    Black-Scholes model for European options (Call and Put).
    
    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - q: Dividend yield
    - option_type: 'put' or 'call'
    
    Returns:
    - Option price
    """
    if T == 0:
        # At expiry, just intrinsic value
        if option_type.lower() == "call":
            return max(S - K, 0)
        elif option_type.lower() == "put":
            return max(K - S, 0)
        else:
            print("Option type not supported.")
            return None
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    elif option_type.lower() == "call":
        price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        print("Option type not supported.")
        return None

    return price


