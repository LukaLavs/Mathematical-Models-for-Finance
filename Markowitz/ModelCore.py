import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cp


def model_2_13(S, m, delta):
    """ max return - delta*(standard deviation)
    subject to: use all capital and no sort selling"""
    #Note: problem is in conic form described under model (2.13)
    
    N = m.shape[0]  # Number of securities

    # Cholesky factor of S
    G = np.linalg.cholesky(S)

    # Variables
    x = cp.Variable(N)
    s = cp.Variable()

    for delta_value in [delta]:
        delta = cp.Parameter(nonneg=True)
        delta.value = delta_value

        # Objective
        objective = cp.Maximize(m.T @ x - delta * s)

        # Constraints
        constraints = [
            cp.sum(x) == 1,
            x >= 0,
            cp.SOC(s, G.T @ x)  # Only this expresses the risk constraint!
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve()

        portfolio_return = m.T @ x.value
        portfolio_risk = np.sqrt(x.value @ S @ x.value)
        portfolio_weights = x.value

    return portfolio_weights, portfolio_return, portfolio_risk

