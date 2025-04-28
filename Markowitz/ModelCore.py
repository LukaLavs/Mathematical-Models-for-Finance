import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cp


def model_2_13(m, S, deltas):
    """ max return - delta*(standard deviation)
    subject to: use all capital and no sort selling"""
    #Note: problem is in conic form described under model (2.13)
    
    N = m.shape[0]  # Number of securities

    # Cholesky factor of S
    G = np.linalg.cholesky(S)

    # Variables
    x = cp.Variable(N)
    s = cp.Variable()

    columns = ["delta", "obj", "return", "risk"] + [f"weight_{i}" for i in range(N)]
    df_result = pd.DataFrame(columns=columns)

    for delta_value in deltas:
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
        portfolio_risk = np.sqrt(s.value)
        portfolio_weights = x.value

        row = pd.Series([delta_value, problem.value, portfolio_return, portfolio_risk] + list(portfolio_weights),
                        index=columns)
        df_result = pd.concat([df_result, row.to_frame().T], ignore_index=True)

    return df_result















    
    
#m, S = calculate_distribution_parameters()
#print(m)
#print(S)

