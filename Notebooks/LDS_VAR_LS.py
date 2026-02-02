"""
LDS to VAR(p) Least Squares

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

    






