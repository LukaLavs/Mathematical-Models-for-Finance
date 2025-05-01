import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def display_results(models, show_df=False):
    """ Assumes models is given by [[modeloutput, modelName], ...]"""
    if show_df == True:
        for (df, name) in models:
            print(name)
            print(df.head())    
    # Plot
    plt.figure(figsize=(8, 6))
    for (df, name) in models:
        plt.plot(df["risk"], df["return"], '-o', label=name)
    # Labels and title
    plt.xlabel("Portfolio Risk (Std. Dev.)")
    plt.ylabel("Portfolio Return")
    plt.title("Efficient Frontier Comparison")
    plt.grid(True)
    plt.legend()
    plt.show()
    

def visualize_generated_stocks(df_prices):
    df_prices.plot(figsize=(10, 6))  # Plot the entire DataFrame
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(title="Stock Symbols")
    plt.grid(True)
    plt.show()
    
    
def make_df(deltas, model, model_params):
    m = model_params["m"]
    S = model_params["S"]
    N = m.shape[0]
    columns = ["delta", "return", "risk"] + [f"weight_{i}" for i in range(N)]
    rows = []
    for delta in deltas:
        model_params["delta"] = delta
        weights, returns, _ = model(**model_params)
        risk = np.sqrt(weights @ S @ weights)
        row = [delta, returns, risk] + list(weights)
        rows.append(row)
    # Create DataFrame once, with all rows
    df_result = pd.DataFrame(rows, columns=columns)
    return df_result


    
