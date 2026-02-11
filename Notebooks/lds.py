import numpy as np

def simulate_lds (n, A, C, L, rng, e_scale=0.2):
    """
    Simulating the system as follows:

    x_{k+1} = Ax_k + Le_k
    y_k     = Cx_k + e_k 

    Returns:
        x: (n, d_x)
        y: (n, d_y)
        e: (n, d_y)
    
    """

    # State and output dimensions from system matrices
    d_x = A.shape[0] 
    d_y = C.shape[0]

    # Create matrices for x and y with appropriate dimensions 
    x = np.zeros((n, d_x))
    y = np.zeros((n, d_y))
    e = e_scale * rng.normal(size=(n, d_y))

    # Initial state
    x[0] = rng.normal(size=d_x)
    y[0] =  C @ x[0] + e[0]

    for k in range(0, n - 1):
        x[k + 1] = A @ x[k] + L @ e[k]
        y[k + 1] = C @ x[k + 1] + e[k + 1]

    return x, y, e


def spectral_radius(M: np.ndarray) -> float:
    """
    Spectral radius rho(M) = max |lambda_i(M)|
    """
    eigs = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigs)))


def sample_CL_in_band(
    A: np.ndarray,
    d_x: int,
    d_y: int,
    rho_low: float,
    rho_high: float,
    rng: np.random.Generator,
    max_tries: int = 20000,
    C_scale: float = 1.0,
    L_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Sample random C (d_y x d_x) and L (d_x x d_y) until
    rho(F) is in [rho_low, rho_high], where F = A - L C.

    Returns:
        C, L, F, rhoF

    Raises:
        RuntimeError if not found within max_tries.

    Practical notes:
    - For narrow bands near 1 (e.g., [0.985, 0.9999]) you might need large max_tries.
    - Scaling (C_scale, L_scale) can make it easier/harder to hit certain bands.
    """
    if not (0 <= rho_low <= rho_high):
        raise ValueError("Require 0 <= rho_low <= rho_high.")
    if rho_high >= 1.5:  
        raise ValueError("rho_high seems unusually large; double-check inputs.")

    for _ in range(max_tries):
        C = C_scale * rng.normal(size=(d_y, d_x))
        L = L_scale * rng.normal(size=(d_x, d_y))

        F = A - L @ C
        rhoF = spectral_radius(F)

        if rho_low <= rhoF <= rho_high:
            return C, L, F, rhoF

    raise RuntimeError(
        f"Couldn't sample C,L with rho(A-LC) in [{rho_low}, {rho_high}] "
        f"after {max_tries} tries."
    )


