import numpy as np
import matplotlib.pyplot as plt

from lds import simulate_lds, sample_CL_in_band
from var_model import build_var_xy, fit_ls
from metrics import (
    mahalanobis_var_distance,
    mahalanobis_lag_contributions_from_B,
)


def fit_var_and_get_components(y: np.ndarray, p: int):
    """
    Fit VAR(p) by LS and return:
      - B_hat      = LS estimate in regression Y = X B + U
      - pi_hat     = vec(B_hat) (column-stacking / Fortran order)
      - Sigma_hat  = (U^T U)/T  residual covariance estimate
      - QX_hat     = (X^T X)/T  regressor covariance estimate
      - T          = number of rows in X (effective sample size)
    """
    X, Y = build_var_xy(y, p=p)        # X: (T, p*d_y), Y: (T, d_y)
    B_hat = fit_ls(Y=Y, X=X)           # B_hat: (p*d_y, d_y)

    T = X.shape[0]
    U_hat = Y - X @ B_hat              # (T, d_y)

    Sigma_hat = (U_hat.T @ U_hat) / T  # (d_y, d_y)
    QX_hat = (X.T @ X) / T             # (p*d_y, p*d_y)

    pi_hat = B_hat.reshape(-1, order="F")  # vec(B_hat)

    return B_hat, pi_hat, Sigma_hat, QX_hat, T


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

    same_mode_band  = (0.75, 0.80)
    other_mode_band = (0.95, 0.98)

    # Store per-lag contributions
    contrib_same = np.zeros((trials, p))
    contrib_diff = np.zeros((trials, p))

    # Also store total D for sanity check
    D_same = np.zeros(trials)
    D_diff = np.zeros(trials)

    for t in range(trials):
        # ----- Case 1: SAME LDS -----
        C, L, _, _ = sample_CL_in_band(
            A=A, d_x=d_x, d_y=d_y,
            rho_low=same_mode_band[0], rho_high=same_mode_band[1],
            rng=rng, max_tries=20000
        )

        _, y1, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)
        _, y2, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

        B1, pi1, Sig1, QX1, _ = fit_var_and_get_components(y1, p=p)
        B2, pi2, Sig2, QX2, _ = fit_var_and_get_components(y2, p=p)

        contrib_same[t, :] = mahalanobis_lag_contributions_from_B(
            B1_hat=B1, B2_hat=B2,
            Sigma1_hat=Sig1, Sigma2_hat=Sig2,
            QX1_hat=QX1, QX2_hat=QX2,
            d_y=d_y, p=p,
            regularize=1e-8
        )

        D_same[t] = mahalanobis_var_distance(
            pi1_hat=pi1, pi2_hat=pi2,
            Sigma1_hat=Sig1, Sigma2_hat=Sig2,
            QX1_hat=QX1, QX2_hat=QX2,
            regularize=1e-8
        )

        # ----- Case 2: DIFFERENT LDS MODES -----
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

        Ba, pi_a, Sig_a, QX_a, _ = fit_var_and_get_components(ya, p=p)
        Bb, pi_b, Sig_b, QX_b, _ = fit_var_and_get_components(yb, p=p)

        contrib_diff[t, :] = mahalanobis_lag_contributions_from_B(
            B1_hat=Ba, B2_hat=Bb,
            Sigma1_hat=Sig_a, Sigma2_hat=Sig_b,
            QX1_hat=QX_a, QX2_hat=QX_b,
            d_y=d_y, p=p,
            regularize=1e-8
        )

        D_diff[t] = mahalanobis_var_distance(
            pi1_hat=pi_a, pi2_hat=pi_b,
            Sigma1_hat=Sig_a, Sigma2_hat=Sig_b,
            QX1_hat=QX_a, QX2_hat=QX_b,
            regularize=1e-8
        )

    # -------------------- summaries --------------------
    print("===== Lag-wise Mahalanobis contributions (FROM B blocks) =====")
    print(f"trials={trials}, n={n}, d_y={d_y}, p={p}, e_scale={e_scale}, seed={seed}")
    print(f"same_mode_band  = {same_mode_band}")
    print(f"other_mode_band = {other_mode_band}")
    print("")
    print("Total D (full metric):")
    print(f"  D_same: mean={D_same.mean():.6g}, std={D_same.std(ddof=1):.6g}, median={np.median(D_same):.6g}")
    print(f"  D_diff: mean={D_diff.mean():.6g}, std={D_diff.std(ddof=1):.6g}, median={np.median(D_diff):.6g}")
    print("")
    print("Sum of lag contributions (blockwise approx, ignores cross-lag blocks):")
    print(f"  sum contrib_same: mean={contrib_same.sum(axis=1).mean():.6g}")
    print(f"  sum contrib_diff: mean={contrib_diff.sum(axis=1).mean():.6g}")

    # -------------------- box plots --------------------
    data_same = [contrib_same[:, i] for i in range(p)]
    data_diff = [contrib_diff[:, i] for i in range(p)]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data_same, showfliers=False)
    plt.xlabel("Lag i")
    plt.ylabel("Lag contribution $D_i$")
    plt.ylim(0,375)
    plt.title("Same LDS: lag-wise covariance-weighted contributions (blockwise)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.boxplot(data_diff, showfliers=False)
    plt.xlabel("Lag i")
    plt.ylabel("Lag contribution $D_i$")
    plt.ylim(0,375)
    plt.title("Different modes: lag-wise covariance-weighted contributions (blockwise)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -------------------- median curves --------------------
    med_same = np.median(contrib_same, axis=0)
    med_diff = np.median(contrib_diff, axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, p + 1), med_same, marker="o", label="Same (median)")
    plt.plot(range(1, p + 1), med_diff, marker="o", label="Diff (median)")
    plt.xlabel("Lag i")
    plt.ylabel("Median lag contribution")
    plt.title("Median lag contributions across lags (blockwise)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
