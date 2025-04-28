import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def display_results(models):
    """ Assumes models is given by [[modeloutput, modelName], ...]"""
    #deltas = np.logspace(start=-1, stop=1.5, num=20)[::-1]

    #df_result1 = fusion_model(m, S, deltas)
    #df_result2 = fusion_model(m_shr, S_shr, deltas)
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
    
