import numpy as np
import matplotlib.pyplot as plt
from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import phi_distance_between_models


def p_sensitivity_report(
    p_list=(8, 10, 12),
    regimes=(
        ("short", 0.75, 0.80),
        ("long",  0.95, 0.98),
    ),
    trials=30,
    n=1500,
    d_x=2,
    d_y=5,
    e_scale=0.2,
    max_tries=20000,
    seed=None,
):
    """
    Runs a p-sensitivity test for different memory regimes.

    For each regime (name, rho_low, rho_high):
      - repeatedly sample random C,L until rho(A - L C) lies in [rho_low, rho_high]
      - simulate LDS
      - fit VAR(p) by LS
      - compute:
          * Training MSE = mean((Y - X B_hat)^2)
          * Mean Phi error = mean_i ||Phi_hat_i - C(A-LC)^(i-1)L||_F^2
          * Var across lags of Phi error = Var_i(||...||_F^2)
    Prints mean ± std across trials for each p.
    """

    rng = np.random.default_rng(seed)

    A = np.array([[0.9, -0.2],
                  [0.2,  0.8]])

    def sample_CL_in_band(rho_low, rho_high):
        for _ in range(max_tries):
            C = rng.normal(size=(d_y, d_x))
            L = rng.normal(size=(d_x, d_y))
            F = A - L @ C
            rhoF = np.max(np.abs(np.linalg.eigvals(F)))
            if rho_low <= rhoF <= rho_high:
                return C, L, F, rhoF
        raise RuntimeError(f"Couldn't sample C,L with rho in [{rho_low}, {rho_high}] after {max_tries} tries.")

    print("\n====================== p SENSITIVITY REPORT ======================")
    print(f"trials={trials}, n={n}, d_y={d_y}, e_scale={e_scale}")
    print("p_list =", list(p_list))
    print("regimes =", regimes)

    for regime_name, rho_low, rho_high in regimes:
        print(f"\n--- Regime: {regime_name}  (rho in [{rho_low}, {rho_high}]) ---")

        for p in p_list:
            if p < 1:
                raise ValueError(f"p must be >= 1 (got p={p})")
            if n <= p:
                raise ValueError(f"need n > p (got n={n}, p={p})")

            mses = []
            phi_means = []
            phi_vars = []
            rhos = []

            for _ in range(trials):
                C, L, F, rhoF = sample_CL_in_band(rho_low, rho_high)

                # simulate
                x, y, e = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

                # fit VAR(p)
                X, Y = build_var_xy(y, p=p)
                B_hat = fit_ls(Y=Y, X=X)

                # training MSE
                Y_hat = X @ B_hat
                residuals = Y - Y_hat
                mses.append(float(np.mean(residuals**2)))

                # unpack Phi-hat
                Phi_list = unpack_B_to_Phi(B_hat, d_y=d_y, p=p)

                # compute per-lag squared Phi errors
                phi_sq_errors = []
                Fpow = np.eye(d_x)  # F^(i-1), starts at i=1 -> power 0
                for i in range(p):
                    Phi_theory_i = C @ Fpow @ L
                    diff = Phi_list[i] - Phi_theory_i
                    phi_sq_errors.append(np.linalg.norm(diff, 'fro')**2)
                    Fpow = Fpow @ F

                phi_means.append(float(np.mean(phi_sq_errors)))
                # ddof=1 gives sample variance; if p==1 we'd guard, but your p>=8
                phi_vars.append(float(np.var(phi_sq_errors, ddof=1)))
                rhos.append(float(rhoF))

            print(f"\np = {p}")
            print(f"  achieved rhoF:   {np.mean(rhos):.4f} ± {np.std(rhos, ddof=1):.4f}")
            print(f"  Training MSE:    {np.mean(mses):.6f} ± {np.std(mses, ddof=1):.6f}")
            print(f"  Mean Phi error:  {np.mean(phi_means):.6f} ± {np.std(phi_means, ddof=1):.6f}")
            print(f"  Var Phi error:   {np.mean(phi_vars):.6f} ± {np.std(phi_vars, ddof=1):.6f}")



def phi_error_spread(
    regime_name="short",
    rho_low=0.75,
    rho_high=0.80,
    trials=50,
    n=1500,
    d_x=2,
    d_y=5,
    p=10,
    e_scale=0.2,
    max_tries=20000,
):
    """
    For each trial, resample (C,L) such that rho(F)=rho(A-LC) in a set range (i.e., [rho_low, rho_high]).
    Then simulate, fit VAR(p) via LS, and compute per-lag squared Frobenius errors:
        e_i = ||Phi_hat_i - Phi_i^*||_F^2
    where Phi_i^* = C F^(i-1) L (with F = A - L C).
    Finally plot mean ± std of e_i across trials, for each lag i=1..p.

    Returns:
        phi_errs: (trials, p) array of errors
        mean_err: (p,) mean across trials
        std_err:  (p,) std across trials
        rhos:     (trials,) achieved rho(F) values
    """

    rng = np.random.default_rng()  # no seed: fresh randomness each run

    A = np.array([[0.9, -0.2],
                  [0.2,  0.8]])

    def sample_CL_in_band():
        for _ in range(max_tries):
            C = rng.normal(size=(d_y, d_x))
            L = rng.normal(size=(d_x, d_y))
            F = A - L @ C
            rhoF = np.max(np.abs(np.linalg.eigvals(F)))
            if rho_low <= rhoF <= rho_high:
                return C, L, F, rhoF
        raise RuntimeError(f"Couldn't sample C,L with rho in [{rho_low}, {rho_high}] after {max_tries} tries.")

    phi_errs = np.zeros((trials, p))
    rhos = np.zeros(trials)

    for t in range(trials):
        C, L, F, rhoF = sample_CL_in_band()
        rhos[t] = rhoF

        # simulate
        x, y, e = simulate_lds(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

        # fit VAR(p)
        X, Y = build_var_xy(y, p=p)
        B_hat = fit_ls(Y=Y, X=X)
        Phi_hat_list = unpack_B_to_Phi(B_hat, d_y=d_y, p=p)

        # theoretical Phi_i^* for this trial
        Fpow = np.eye(d_x)
        for i in range(p):
            Phi_star_i = C @ Fpow @ L
            diff = Phi_hat_list[i] - Phi_star_i
            phi_errs[t, i] = np.linalg.norm(diff, 'fro')**2
            Fpow = Fpow @ F

    mean_err = phi_errs.mean(axis=0)
    std_err  = phi_errs.std(axis=0, ddof=1)


    plt.figure(figsize=(8, 4))

    # Boxplot expects a list of arrays, one per box
    data_per_lag = [phi_errs[:, i] for i in range(p)]

    plt.boxplot(
        data_per_lag,
        positions=np.arange(1, p + 1),
        widths=0.6,
        showfliers=True,      # show outliers
        patch_artist=True,  
    )

    for box in plt.gca().artists:
        box.set_facecolor("#cfe2f3")
        box.set_edgecolor("black")



    plt.xlabel("Lag index i")
    plt.ylabel(r"$\|\hat{\Phi}_i - \Phi_i^*\|_F^2$")
    plt.title(
        f"{regime_name} memory: VAR coefficient error distribution\n"
        f"rho(F) in [{rho_low}, {rho_high}], trials={trials}"
    )

    plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


    return phi_errs, mean_err, std_err, rhos


import numpy as np

def phi_distance_between_models(Phi1, Phi2, *, squared: bool = True, average: bool = False) -> float:
    """
    Distance between two VAR(p) models via their lag matrices.

    Given two lists of VAR coefficient matrices:
        Phi1 = [Phi1_1, ..., Phi1_p]
        Phi2 = [Phi2_1, ..., Phi2_p]

    Computes:
        D = sum_{i=1..p} ||Phi1_i - Phi2_i||_F^2        (default)

    Options:
        - squared=False: uses ||.||_F (not squared) per lag, then sums/averages
        - average=True:  returns (1/p)*sum(...) instead of sum(...)

    Args:
        Phi1, Phi2: lists (or tuples) of numpy arrays with identical shapes.
        squared: whether to square the Frobenius norm per lag.
        average: whether to divide by p at the end.

    Returns:
        A float distance.

    Raises:
        ValueError: if lengths mismatch or matrix shapes mismatch.
    """
    if len(Phi1) != len(Phi2):
        raise ValueError(f"Phi lists must have same length (got {len(Phi1)} vs {len(Phi2)}).")

    p = len(Phi1)
    if p == 0:
        raise ValueError("Phi lists must be non-empty.")

    total = 0.0
    for i, (A, B) in enumerate(zip(Phi1, Phi2), start=1):
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch at lag {i}: {A.shape} vs {B.shape}.")
        diff = A - B
        fro = np.linalg.norm(diff, ord="fro")
        total += float(fro * fro if squared else fro)

    if average:
        total /= p

    return total
