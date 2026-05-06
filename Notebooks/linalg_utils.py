import numpy as np

def spectral_radius(M: np.ndarray) -> float:
    """Return spectral radius rho(M) = max_i |lambda_i(M)|."""
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def effective_F(A: np.ndarray, C: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the effective observed-memory matrix F = A - L C."""
    return A - L @ C