import numpy as np
from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi


def sample_stable_A(d_x, rho_min, rho_max, rng):
    """Sample A and rescale so its spectral radius lies in [rho_min, rho_max]."""
    A_raw = rng.normal(size=(d_x, d_x))
    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)
    return (rho_target / (rho_raw + 1e-12)) * A_raw


def sample_C(d_y, d_x, rng):
    """Sample observation matrix C."""
    return rng.normal(size=(d_y, d_x))


def sample_L(d_x, d_y, A, C, rng, target_rho=0.8, max_iter=200):
    """
    Sample L and shrink it if needed so that F = A - L C
    has spectral radius below target_rho.
    """
    L = rng.normal(size=(d_x, d_y))
    for _ in range(max_iter):
        F = A - L @ C
        rho_F = np.max(np.abs(np.linalg.eigvals(F)))
        if rho_F < target_rho:
            return L
        L *= 0.95
    return L


def theoretical_var_coeffs(A, C, L, p):
    """
    Compute theoretical VAR(p) coefficient matrices induced by the LDS:
        Phi_i = C (A - L C)^{i-1} L
    """
    d_y = C.shape[0]
    d_x = A.shape[0]
    F = A - L @ C

    Phi = []
    F_power = np.eye(d_x)
    for i in range(1, p + 1):
        if i == 1:
            F_power = np.eye(d_x)
        else:
            F_power = F_power @ F
        Phi_i = C @ F_power @ L
        Phi.append(Phi_i)

    return Phi


def test_fit_ls_B():
    rng = np.random.default_rng(0)

    n = 50000
    p = 10
    d_x = 5
    d_y = 5
    e_scale = 0.01

    # Generate LDS the same way you do elsewhere
    A = sample_stable_A(d_x=d_x, rho_min=0.7, rho_max=0.9, rng=rng)
    C = sample_C(d_y=d_y, d_x=d_x, rng=rng)
    L = sample_L(d_x=d_x, d_y=d_y, A=A, C=C, rng=rng, target_rho=0.8)

    # Simulate data from the LDS
    x, y, e = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

    # Build regression matrices and fit
    X, Y = build_var_xy(y, p)
    B_hat = fit_ls(Y, X)

    # Theoretical truncated VAR(p) coefficients
    Phi_true = theoretical_var_coeffs(A, C, L, p)
    B_true = np.vstack([Phi_i.T for Phi_i in Phi_true])

    diff  = B_hat - B_true
    print("Difference (B_hat - B_true): ")
    print("Frobenius norm of difference:", np.linalg.norm(diff, ord="fro"))

    #print("B_true shape:", B_true.shape)
    #print("B_hat shape: ", B_hat.shape)

    #abs_err = np.linalg.norm(B_hat - B_true, ord="fro")
    #rel_err = abs_err / (np.linalg.norm(B_true, ord="fro") + 1e-12)

    #print("Frobenius absolute error:", abs_err)
    #print("Frobenius relative error:", rel_err)

    #Phi_hat = unpack_B_to_Phi(B_hat, d_y, p)

    #for i, (Phi_i_true, Phi_i_hat) in enumerate(zip(Phi_true, Phi_hat), start=1):
    #    lag_abs = np.linalg.norm(Phi_i_hat - Phi_i_true, ord="fro")
    #    lag_rel = lag_abs / (np.linalg.norm(Phi_i_true, ord="fro") + 1e-12)
    #    print(f"Lag {i}:")
    #    print("  Phi_true shape:", Phi_i_true.shape)
    #    print("  Phi_hat  shape:", Phi_i_hat.shape)
    #   print(f"  abs err = {lag_abs:.6e}")
    #    print(f"  rel err = {lag_rel:.6e}")


test_fit_ls_B()





