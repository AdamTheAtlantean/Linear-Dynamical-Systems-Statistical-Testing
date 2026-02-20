import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from lds import simulate_lds, sample_CL_in_band
from var_model import build_var_xy, fit_ls
from metrics import mahalanobis_var_distance


def fit_var_and_get_components(y: np.ndarray, p: int):
    """
    Fit VAR(p) by LS and return:
      - pi_hat     = vec(B_hat) (column-stacking / Fortran order)
      - Sigma_hat  = (U^T U)/T  residual covariance estimate
      - QX_hat     = (X^T X)/T  regressor covariance estimate

    VAR regression is: Y = X B + U
      X: (T, k_x), Y: (T, d_y), B: (k_x, d_y), U: (T, d_y)
    """
    X, Y = build_var_xy(y, p=p)        # X: (T, k_x), Y: (T, d_y)
    B_hat = fit_ls(Y=Y, X=X)           # B_hat: (k_x, d_y)

    T = X.shape[0]
    U_hat = Y - X @ B_hat              # (T, d_y)

    Sigma_hat = (U_hat.T @ U_hat) / T  # (d_y, d_y)
    QX_hat = (X.T @ X) / T             # (k_x, k_x)

    # vec(B_hat)
    pi_hat = B_hat.reshape(-1, order="F")  # (k_x * d_y,)

    return pi_hat, Sigma_hat, QX_hat


def main():

    # -------------------- global settings --------------------
    seed = 0
    rng = np.random.default_rng(seed)

    trials = 300     
    n = 1500
    d_x = 2
    d_y = 5
    p = 10
    e_scale = 0.2

    A = np.array([[0.9, -0.2],
                  [0.2,  0.8]])

    # Define two LDS "modes" via rho(F)=rho(A-LC) bands
    same_mode_band  = (0.75, 0.80)   # same-LDS case uses this band for its ONE sampled model
    other_mode_band = (0.95, 0.98)   # different-modes case samples a second model from this band

    # -------------------- collect distances --------------------
    D_same = np.zeros(trials)
    D_diff = np.zeros(trials)

    for t in range(trials):
        # ===== Case 1: SAME LDS model (same A,C,L; different noise) =====
        C, L, _, _ = sample_CL_in_band(
            A=A, d_x=d_x, d_y=d_y,
            rho_low=same_mode_band[0], rho_high=same_mode_band[1],
            rng=rng, max_tries=20000
        )

        _, y1, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)
        _, y2, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

        pi1, Sig1, QX1 = fit_var_and_get_components(y1, p=p)
        pi2, Sig2, QX2 = fit_var_and_get_components(y2, p=p)

        D_same[t] = mahalanobis_var_distance(
            pi1_hat=pi1, pi2_hat=pi2,
            Sigma1_hat=Sig1, Sigma2_hat=Sig2,
            QX1_hat=QX1, QX2_hat=QX2,
            regularize=1e-8
        )

        # ----- Case 2: DIFFERENT LDS modes (different C,L from different bands) -----
        C_a, L_a, _, _ = sample_CL_in_band(
            A=A, d_x=d_x, d_y=d_y,
            rho_low=same_mode_band[0], rho_high=same_mode_band[1],
            rng=rng, max_tries=20000
        )
        C_b, L_b, _, _ = sample_CL_in_band(
            A=A, d_x=d_x, d_y=d_y,
            rho_low=other_mode_band[0], rho_high=other_mode_band[1],
            rng=rng, max_tries=20000
        )

        _, ya, _ = simulate_lds(n=n, A=A, C=C_a, L=L_a, rng=rng, e_scale=e_scale)
        _, yb, _ = simulate_lds(n=n, A=A, C=C_b, L=L_b, rng=rng, e_scale=e_scale)

        pi_a, Sig_a, QX_a = fit_var_and_get_components(ya, p=p)
        pi_b, Sig_b, QX_b = fit_var_and_get_components(yb, p=p)

        D_diff[t] = mahalanobis_var_distance(
            pi1_hat=pi_a, pi2_hat=pi_b,
            Sigma1_hat=Sig_a, Sigma2_hat=Sig_b,
            QX1_hat=QX_a, QX2_hat=QX_b,
            regularize=1e-8
        )

    # -------------------- numeric summary --------------------
    print("Task 3: Empirical distance distributions (new metric)")
    print(f"trials={trials}, n={n}, d_y={d_y}, p={p}, e_scale={e_scale}, seed={seed}")
    print(f"same_mode_band  = {same_mode_band}")
    print(f"other_mode_band = {other_mode_band}")
    print("")
    print(f"D_same: mean={D_same.mean():.6g}, std={D_same.std(ddof=1):.6g}, median={np.median(D_same):.6g}")
    print(f"D_diff: mean={D_diff.mean():.6g}, std={D_diff.std(ddof=1):.6g}, median={np.median(D_diff):.6g}")

    # -------------------- smooth density (KDE) plot --------------------
    use_log_x = True

    eps = 1e-12
    xmin = float(min(D_same.min(), D_diff.min()))
    xmax = float(max(D_same.max(), D_diff.max()))

    if use_log_x:
        xmin = max(xmin, eps)
        x_vals = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
    else:
        x_vals = np.linspace(xmin, xmax, 1000)

    kde_same = gaussian_kde(D_same)
    kde_diff = gaussian_kde(D_diff)

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, kde_same(x_vals), label="Same LDS (noise only)")
    plt.plot(x_vals, kde_diff(x_vals), label="Different LDS modes")

    if use_log_x:
        plt.xscale("log")

    plt.xlabel(r"Covariance-weighted distance $D$")
    plt.ylabel("Probability density (KDE)")
    plt.title("Smoothed empirical distributions (new covariance-weighted metric)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- overlay hist + KDE  --------------------
    plt.figure(figsize=(8, 4))
    plt.hist(D_same, bins=50, density=True, alpha=0.35, label="Same LDS (hist)")
    plt.hist(D_diff, bins=50, density=True, alpha=0.35, label="Different modes (hist)")
    plt.plot(x_vals, kde_same(x_vals), label="Same LDS (KDE)")
    plt.plot(x_vals, kde_diff(x_vals), label="Different modes (KDE)")

    if use_log_x:
        plt.xscale("log")

    plt.xlabel(r"Covariance-weighted distance $D$")
    plt.ylabel("Probability density")
    plt.title("Histogram + KDE (new metric)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Quantiles (50%, 90%, 99%)")
    print("D_same:", np.quantile(D_same, [0.5, 0.9, 0.99]))
    print("D_diff:", np.quantile(D_diff, [0.5, 0.9, 0.99]))


if __name__ == "__main__":
    main()
