import numpy as np

def build_var_xy(y, p):
    """
    
    """

    n, d_y = y.shape # touple unpacking (get the shape of y and assign it to 'n' and 'd_y')
    if p < 1:
        raise ValueError("p must be >= 1")
    if n <= p:
        raise ValueError("N must be > p")
    

    # Build the target matrix
    Y = y[p:]
    # Allocate design matrix
    X = np.zeros((n - p, p * d_y))


    for i in range(1, p + 1):
        # lag 'i' fills columns with y_{t-i}
        # Fill a block of columns in X with a shifted slice of 'y'.
        X[:, (i - 1) * d_y : i * d_y] = y[p - i : n - i]

    return X, Y
    

def fit_ls(Y, X):
    """    
    Solve min_B ||Y - XB||^2 using Ordinary Least Squares (OLS)

    Returns B-hat

    """

    XtX = X.T @ X
    XtY = X.T @ Y
    B_hat = np.linalg.solve(XtX, XtY) # apparently this has benefits over the vanilla below
    #B_hat = np.linalg.inv(XtX) @ XtY

    return B_hat


def unpack_B_to_Phi(B_hat, d_y, p):
    """
    """

    Phi = []
    for i in range(p):
        block = B_hat[i * d_y : (i + 1) * d_y, :]
        Phi.append(block.T)
    return Phi