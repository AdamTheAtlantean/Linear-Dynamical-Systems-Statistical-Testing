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



# 2)  -------------------------- Build the VAR(p) Design Matrices (i.e., Y = XB) --------------------------

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
    

    def fit_ls(Y, X):
        """    
        Solve min_B ||Y - XB||^2 using Ordinary Least Squares (OLS)

        Returns B-hat

        """

        XtX = X.T @ X
        XtY = X.T @ Y
        #B_hat = np.linalg.solve(XtX, XtY) # apparently this has benefits over the vanilla below
        B_hat = np.linalg.inv(XtX) @ XtY

        return B_hat
    

    def unpack_B_to_Phi(B_hat, d_y, p):
        """
        """

        Phi = []
        for i in range(p):
            block = B-hat[i * d_y : (i + 1) * d_y, :]
            Phi.append(block.T)
        return Phi
    

    def main():
        # Dimensions 
        n = 1500
        d_x = 2
        d_y = 5
        p = 10
        seed = 0

        rng = np.random.default_rng(seed)

        # Define the systems matrices used to generate simulate data
        A = np.array([[0.9, -0.2],
                     [0.2, 0.8]])
        C = rng.normal(size=(d_y, d_x))
        L = rng.normal(size=(d_x, d_y))

        # Check contraction from the theory
        F = A - L @ C
        rhoF = np.max(np.abs(np.linalg.eigenvals(F)))
        print("Spectral radius rho(A - LC):", rhoF)

        # Simulate constrained LDS
        x, y, e = simluate_LDS(n=n, A=A, C=C, L=L, e_scale=0.2, seed=seed)
        print("x shape:" x.shape, "y_shape:", y.shape)

        # VAR(p) regression matrices 
        X, Y = build_var_xy(y, p=p)
        print("X shape:", X.shape, "Y shape:", Y.shape)

        # Least squares estiamte of B
        B_hat - fit_ls(Y=Y, X=X)

    











