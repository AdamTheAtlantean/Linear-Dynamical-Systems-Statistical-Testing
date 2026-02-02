"""
Starting with a LDS and, assuming stability, we appx. the outputs with a VAR(p) model then
estimate the VAR coeff. matrix from observered time series data using Least Squares.

Pipeline: 
1) Simulate (x_k, y_k) from the constrained LDS.
2) Build VAR(p) regression matrces (Y = XB + U) from y_k only.
3) Estimate B by Least Squares.
4) Unpack B into VAR coefficient Matrices Phi_1, ... , Phi_p and plot a demo prediction.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1)  -------------------------- Simulate the LDS --------------------------

def simluate_LDS (n, A, C, L, e_scale=0.2, seed=0):
    """
    Simulating the system as follows:

    x_{k+1} = Ax_k + Le_k
    y_k     = Cx_k + e_k (where e_k ~ N(O, e_sclae^2 I_{d_y}))

    Returns:
        x: (n, d_x)
        y: (n, d_y)
        e: (n, d_y)
    
    """

    rng = np.random.default_rng(seed) # fixed seed for reproducible randomness
    d_x = A.shape[0]
    d_y = C.shape[0]

    x = np.zeros((n, d_x))
    y = np.zeros((n, d_y))
    e = e_scale * rng.normal(size=(n, d_y))

    # initial state
    x[0] = rng.normal(size=d_x)
    y[0] =  C @ x[0] + e[0]

    for k in range(0, n - 1):
        x[k + 1] = A @ x[k] + L @ e[k]
        y[k + 1] = C @ x[k + 1] + e[k + 1]

    return x, y, e



# 2)  -------------------------- Simulate the LDS --------------------------

def build_var_xy(y, p)
    """
    
    """

    n, d_y = y.shape # touple unpacking (get the shape of y and assign it to 'n' and 'd_y')
    if p < 1:
        raise ValueError("p must be >= 1")
    if n <= p:
        raise ValueError("N must be < p")
    
    Y = y[p:]
    X = np.zeros((n - p, p * d_y))

    for i in range(1, p + 1):
        # lag 'i' fills columns with y_{t-i}
        X[:, (i - 1) * d_y : i * d_y] = y[p - 1 : n - i]

        return X, Y
    

    
    












