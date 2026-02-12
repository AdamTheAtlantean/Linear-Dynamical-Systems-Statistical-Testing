import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from lds import simulate_lds, sample_CL_in_band
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import phi_distance_between_models


def fit_var_and_get_phi(y: np.ndarray, p: int):
    """Helper: fit VAR(p) by LS and return list of Phi_i matrices."""
    X, Y = build_var_xy(y, p=p)
    B_hat = fit_ls(Y=Y, X=X)
    d_y = y.shape[1]
    return unpack_B_to_Phi(B_hat, d_y=d_y, p=p)


def main():
    
    def fit_var_and_get_phi(y: np.ndarray, p: int):
        """Fit VAR(p) by LS and return list of Phi_i matrices."""
        X, Y = build_var_xy(y, p=p)
        B_hat = fit_ls(Y=Y, X=X)
        d_y = y.shape[1]
        return unpack_B_to_Phi(B_hat, d_y=d_y, p=p)

    # -------------------- global settings --------------------
    seed = 0
    rng = np.random.default_rng(seed)

    trials = 300      # bump this for smoother KDEs (200–1000 are typical)
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
        # ===== Case 1: SAME LDS model (same A,C,L; different noise realizations) =====
        C, L, _, _ = sample_CL_in_band(
            A=A, d_x=d_x, d_y=d_y,
            rho_low=same_mode_band[0], rho_high=same_mode_band[1],
            rng=rng, max_tries=20000
        )

        _, y1, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)
        _, y2, _ = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

        Phi1 = fit_var_and_get_phi(y1, p=p)
        Phi2 = fit_var_and_get_phi(y2, p=p)

        # no 1/p, squared Frobenius norm summed over lags
        D_same[t] = phi_distance_between_models(Phi1, Phi2, squared=True, average=True)

        # ===== Case 2: DIFFERENT LDS modes (different C,L sampled from different bands) =====
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

        Phi_a = fit_var_and_get_phi(ya, p=p)
        Phi_b = fit_var_and_get_phi(yb, p=p)

        D_diff[t] = phi_distance_between_models(Phi_a, Phi_b, squared=True, average=True)

    # -------------------- quick numeric summary --------------------
    print("===== Task 3: Empirical distance distributions =====")
    print(f"trials={trials}, n={n}, d_y={d_y}, p={p}, e_scale={e_scale}, seed={seed}")
    print(f"same_mode_band  = {same_mode_band}")
    print(f"other_mode_band = {other_mode_band}")
    print("")
    print(f"D_same: mean={D_same.mean():.6g}, std={D_same.std(ddof=1):.6g}, median={np.median(D_same):.6g}")
    print(f"D_diff: mean={D_diff.mean():.6g}, std={D_diff.std(ddof=1):.6g}, median={np.median(D_diff):.6g}")

    # -------------------- smooth density (KDE) plot --------------------
    # Use a log-x plot if values are heavy-tailed 
    use_log_x = True

    # Build x grid safely (avoid <= 0 when using log scale)
    eps = 1e-12
    xmin = min(D_same.min(), D_diff.min())
    xmax = max(D_same.max(), D_diff.max())

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
    plt.hist(D_same, density=True)
    plt.hist(D_diff, density=True) 

    if use_log_x:
        plt.xscale("log")

    plt.xlabel(r"$D=\sum_{i=1}^p \|\Phi^{(1)}_i - \Phi^{(2)}_i\|_F^2$")
    plt.ylabel("Probability density (KDE)")
    plt.title("Smoothed empirical distributions of VAR distance")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------- overlay hist + KDE for sanity --------------------
    plt.figure(figsize=(8, 4))
    plt.hist(D_same, bins=50, density=True, alpha=0.35, label="Same LDS (hist)")
    plt.hist(D_diff, bins=50, density=True, alpha=0.35, label="Different modes (hist)")
    plt.plot(x_vals, kde_same(x_vals), label="Same LDS (KDE)")
    plt.plot(x_vals, kde_diff(x_vals), label="Different modes (KDE)")

    if use_log_x:
        plt.xscale("log")

    plt.xlabel(r"$D=\sum_{i=1}^p \|\Phi^{(1)}_i - \Phi^{(2)}_i\|_F^2$")
    plt.ylabel("Probability density")
    plt.title("Histogram + KDE (sanity check)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
