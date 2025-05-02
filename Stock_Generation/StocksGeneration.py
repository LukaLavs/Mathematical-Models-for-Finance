import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_n_stock_data1(
    n_stocks=10,
    n_days=1000,
    random_seed=19,
    n_sectors=3,
    initial_price_range=(50, 500),
    annual_drift=0.1,
    annual_volatility_range=(0.23, 0.28),
    global_crash_prob=0.00001,  # about 1 every 4 years
    sector_crash_prob=0.002,   # about 1 every 500 days
    global_crash_impact=(-0.15, -0.30),
    sector_crash_impact=(-0.07, -0.15),
    recovery_days=40,
    recovery_boost=0.05,
    start_date='2020-01-01'
):
    """
    Simulate realistic stock prices using GBM with market effects and rare crashes.
    """
    np.random.seed(random_seed)
    
    # Dates and basic constants
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1/252  # Daily time step

    # Stocks setup
    initial_prices = np.random.uniform(*initial_price_range, size=n_stocks)
    stock_vols_annual = np.random.uniform(*annual_volatility_range, size=n_stocks)
    stock_vols_daily = stock_vols_annual * np.sqrt(dt)
    stock_drifts_daily = (annual_drift - 0.5 * stock_vols_annual**2) * dt
    sectors = np.random.choice(range(n_sectors), size=n_stocks)

    # Initialize arrays
    prices = np.zeros((n_days, n_stocks))
    prices[0] = initial_prices
    recovery_timer = np.zeros(n_stocks)

    # Start simulating
    for t in range(1, n_days):
        # Draw random shocks
        Z_market = np.random.normal(0, 1)
        Z_sector = np.random.normal(0, 1, size=n_sectors)
        Z_idio = np.random.normal(0, 1, size=n_stocks)

        # Crash events
        global_crash = np.random.rand() < global_crash_prob
        sector_crashes = (np.random.rand(n_sectors) < sector_crash_prob)

        # Crash impacts
        global_crash_return = 0
        if global_crash:
            global_crash_return = np.random.uniform(*global_crash_impact)

        sector_crash_return = np.zeros(n_sectors)
        for s in range(n_sectors):
            if sector_crashes[s]:
                sector_crash_return[s] = np.random.uniform(*sector_crash_impact)

        # Update each stock
        for i in range(n_stocks):
            sector_id = sectors[i]

            # Weighting of shocks
            w_market = 0.2
            w_sector = 0.2
            w_idio = 0.6

            # During crash, shift weights
            if global_crash:
                w_market = 0.7
                w_sector = 0.2
                w_idio = 0.1
                recovery_timer[i] = recovery_days

            if sector_crashes[sector_id]:
                w_market = 0.3
                w_sector = 0.6
                w_idio = 0.1
                recovery_timer[i] = recovery_days

            # Build the return
            noise_component = (
                w_market * Z_market +
                w_sector * Z_sector[sector_id] +
                w_idio * Z_idio[i]
            )

            drift = stock_drifts_daily[i]
            volatility = stock_vols_daily[i]

            # Recovery boost (stronger drift temporarily)
            if recovery_timer[i] > 0:
                drift += recovery_boost * dt
                recovery_timer[i] -= 1

            # Final daily return
            r_t = drift + volatility * noise_component
            r_t += global_crash_return + sector_crash_return[sector_id]

            # Price update
            prices[t, i] = prices[t-1, i] * np.exp(r_t)

    # Build DataFrame
    stock_names = [f'Stock_{i+1}_Sector_{sectors[i]}' for i in range(n_stocks)]
    df_prices = pd.DataFrame(prices, index=dates, columns=stock_names)

    return df_prices



# Faster
def generate_n_stock_data(
    n_stocks=10,
    n_days=1000,
    random_seed=19,
    n_sectors=3,
    initial_price_range=(50, 500),
    annual_drift=0.1,
    annual_volatility_range=(0.23, 0.28),
    global_crash_prob=0.00001,  # about 1 every 4 years
    sector_crash_prob=0.002,   # about 1 every 500 days
    global_crash_impact=(-0.15, -0.30),
    sector_crash_impact=(-0.07, -0.15),
    recovery_days=40,
    recovery_boost=0.05,
    start_date='2020-01-01'
):
    """
    Simulate realistic stock prices using GBM with market effects and rare crashes.
    """
    np.random.seed(random_seed)
    
    # Dates and basic constants
    dates = pd.bdate_range(start=start_date, periods=n_days)
    dt = 1/252  # Daily time step

    # Stocks setup
    initial_prices = np.random.uniform(*initial_price_range, size=n_stocks)
    stock_vols_annual = np.random.uniform(*annual_volatility_range, size=n_stocks)
    stock_vols_daily = stock_vols_annual * np.sqrt(dt)
    stock_drifts_daily = (annual_drift - 0.5 * stock_vols_annual**2) * dt
    sectors = np.random.choice(range(n_sectors), size=n_stocks)

    # Initialize arrays
    prices = np.zeros((n_days, n_stocks))
    prices[0] = initial_prices
    recovery_timer = np.zeros(n_stocks)

    # Precompute shocks
    Z_market = np.random.normal(0, 1, size=n_days)
    Z_sector = np.random.normal(0, 1, size=(n_days, n_sectors))
    Z_idio = np.random.normal(0, 1, size=(n_days, n_stocks))

    # Crash probabilities and impacts
    global_crash = np.random.rand(n_days) < global_crash_prob
    sector_crashes = np.random.rand(n_days, n_sectors) < sector_crash_prob

    global_crash_impact = np.random.uniform(*global_crash_impact, size=n_days)
    sector_crash_impact = np.random.uniform(*sector_crash_impact, size=(n_days, n_sectors))

    # Main simulation loop
    for t in range(1, n_days):
        # Adjust for crashes
        crash_factor = np.zeros((n_stocks,))

        # Apply global crash impacts
        crash_factor += global_crash[t] * global_crash_impact[t]

        # Apply sector crash impacts
        for s in range(n_sectors):
            crash_factor[sectors == s] += sector_crashes[t, s] * sector_crash_impact[t, s]

        # Recovery boost
        recovery_boosts = np.zeros(n_stocks)
        recovery_boosts[recovery_timer > 0] = recovery_boost * dt
        recovery_timer = np.maximum(0, recovery_timer - 1)

        # Calculate returns using vectorized operations
        noise_component = (
            0.2 * Z_market[t] +
            0.2 * Z_sector[t, sectors] +
            0.6 * Z_idio[t]
        )

        drift = stock_drifts_daily + recovery_boosts
        volatility = stock_vols_daily

        # Final return calculation
        r_t = drift + volatility * noise_component + crash_factor

        # Update prices using vectorized calculation
        prices[t] = prices[t-1] * np.exp(r_t)

    # Build DataFrame
    stock_names = [f'Stock_{i+1}_Sector_{sectors[i]}' for i in range(n_stocks)]
    df_prices = pd.DataFrame(prices, index=dates, columns=stock_names)

    return df_prices



def visualize_generated_stocks(df_prices):
    df_prices.plot(figsize=(10, 6))  # Plot the entire DataFrame
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="Stock Symbols")
    plt.grid(True)
    plt.show()