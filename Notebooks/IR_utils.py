import numpy as np
from linalg_utils import effective_F

def impulse_response_matrices(A: np.ndarray, C: np.ndarray, L: np.ndarray, K: int):
    """
    Returns [H_1, ..., H_K] where H_k = C F^{k-1} L and F = A - L C.
    """
    F = effective_F(A, C, L)
    d_x = A.shape[0]
    Fk = np.eye(d_x)
    H = []

    for _ in range(K):
        H.append(C @ Fk @ L)
        Fk = Fk @ F

    return H


def impulse_response_distance(
    A1, C1, L1,
    A2, C2, L2,
    K: int = 25,
    normalize: bool = True
) -> float:
    H1 = impulse_response_matrices(A1, C1, L1, K)
    H2 = impulse_response_matrices(A2, C2, L2, K)

    num = 0.0
    den = 0.0

    for h1, h2 in zip(H1, H2):
        diff = h1 - h2
        num += float(np.sum(diff * diff))
        if normalize:
            den += float(np.sum(h1 * h1))

    return num / (den + 1e-12) if normalize else num


