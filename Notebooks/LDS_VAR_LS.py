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

def simluate_LDS (n, A, C, L, rng, e_scale=0.2):
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



# 2)  -------------------------- Build the VAR(p) Design Matrices (i.e., Y = XB) --------------------------

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



def plot_variance_of_phi_error_across_lags(
    targets=(0.8, 0.98),
    trials=30,
    n=1500,
    d_x=2,
    d_y=5,
    p=10,
    e_scale=0.2,
    max_tries=5000,
    seed=None,
):
    """
    For each target rho(A-LC), run `trials` experiments.
    In each experiment:
      - sample C,L until rho(A-LC) <= target
      - simulate, fit VAR(p), compute per-lag squared Phi errors e_i
      - compute Var(e_1..e_p) across lags
    Then plot mean±std of Var(e_i) over trials for each target.
    """

    rng = np.random.default_rng(seed)
    A = np.array([[0.9, -0.2],
                  [0.2,  0.8]])

    var_across_lags_means = []
    var_across_lags_stds  = []
    achieved_rhos = []

    for target in targets:
        vars_this_target = []
        rhos_this_target = []

        for _ in range(trials):
            # --- resample C,L until rho(A-LC) <= target ---
            for _try in range(max_tries):
                C = rng.normal(size=(d_y, d_x))
                L = rng.normal(size=(d_x, d_y))
                F = A - L @ C
                rhoF = np.max(np.abs(np.linalg.eigvals(F)))
                if rhoF <= target:
                    break
            else:
                raise RuntimeError(f"Couldn't find C,L with rho(A-LC) <= {target} after {max_tries} tries.")

            # simulate + fit VAR
            x, y, e = simluate_LDS(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)
            X, Y = build_var_xy(y, p=p)
            B_hat = fit_ls(Y=Y, X=X)
            Phi_list = unpack_B_to_Phi(B_hat, d_y=d_y, p=p)

            # per-lag squared Phi errors e_i
            phi_sq_errors = []
            Fpow = np.eye(d_x)
            for i in range(p):
                Phi_theory_i = C @ Fpow @ L
                diff = Phi_list[i] - Phi_theory_i
                phi_sq_errors.append(np.linalg.norm(diff, 'fro')**2)
                Fpow = Fpow @ F

            # variance of error across lags (this is the "variance" in the note)
            vars_this_target.append(np.var(phi_sq_errors, ddof=1))
            rhos_this_target.append(rhoF)

        var_across_lags_means.append(np.mean(vars_this_target))
        var_across_lags_stds.append(np.std(vars_this_target))
        achieved_rhos.append(np.mean(rhos_this_target))

    # --- plot mean ± std ---
    x = np.arange(len(targets))
    plt.figure()
    plt.errorbar(x, var_across_lags_means, yerr=var_across_lags_stds, marker='o', capsize=4)
    plt.xticks(x, [f"target={t}\n(avg rho≈{achieved_rhos[i]:.3f})" for i, t in enumerate(targets)])
    plt.xlabel(r"Contraction target for $\rho(A-LC)$")
    plt.ylabel(r"$\mathrm{Var}\left(\|\hat{\Phi}_i-\Phi_i\|_F^2\right)$ across $i=1..p$")
    plt.title("Short vs long memory: variance of Phi-error across lags")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
                x, y, e = simluate_LDS(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

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



def plot_phi_error_spread_per_lag_rhoband(
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
    Option A (no seed): For each trial, resample (C,L) such that rho(F)=rho(A-LC) in [rho_low, rho_high].
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
        x, y, e = simluate_LDS(n=n, A=A, C=C, L=L, rng=rng, e_scale=e_scale)

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

    lags = np.arange(1, p + 1)

    plt.figure()
    plt.errorbar(lags, mean_err, yerr=std_err, marker='o', capsize=4)
    plt.xlabel("Lag index i")
    plt.ylabel(r"Across-trial mean ± std of $\|\hat{\Phi}_i-\Phi_i^*\|_F^2$")
    plt.title(
        f"{regime_name} memory: per-lag Phi error spread\n"
        f"rho(F) in [{rho_low}, {rho_high}], trials={trials}, avg rho≈{rhos.mean():.3f}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return phi_errs, mean_err, std_err, rhos




def main():
    # Dimensions 
    n = 1500
    d_x = 2
    d_y = 5
    p = 10

    rng = np.random.default_rng()

    # Define the systems matrices used to generate simulate data
    A = np.array([[0.9, -0.2],
                    [0.2, 0.8]])
    C = rng.normal(size=(d_y, d_x))
    L = rng.normal(size=(d_x, d_y))

 

    # Check contraction from the theory
    # Ensure spectral radius is below desired threshold (i.e., < 1)
    L0_norm = np.linalg.norm(L)

    max_iter = 50
    target =  0.98
    for _ in range(max_iter):
        F = A - L @ C
        rhoF = np.max(np.abs(np.linalg.eigvals(F)))
        if rhoF <= target:
            break
        L *= (target / rhoF)

    F = A - L @ C
    rhoF = np.max(np.abs(np.linalg.eigvals(F)))

    print("Spectral radius rho(A - LC):", rhoF)
    print("L shrink factor:", np.linalg.norm(L) / L0_norm)


    # Simulate constrained LDS
    x, y, e = simluate_LDS(n=n, A=A, C=C, L=L, rng=rng, e_scale=0.2)
    print("x shape:", x.shape, "y shape:", y.shape)


    # VAR(p) regression matrices 
    X, Y = build_var_xy(y, p=p)
    print("X shape:", X.shape, "Y shape:", Y.shape) 

    # Least squares estiamte of B
    B_hat = fit_ls(Y=Y, X=X)
    print("B-hat shape:", B_hat.shape)

    # Unpacking into VAR coeff. matrices
    Phi_list = unpack_B_to_Phi(B_hat, d_y=d_y, p=p)
    print("Phi_1 shape:", Phi_list[0].shape)


    # Comparing variance of different spectral radii (i.e., .80 vs .98, short term vs long term memory respectively)
    #plot_variance_of_phi_error_across_lags(targets=(0.80, 0.98), trials=30, p=10)

    p_sensitivity_report(
    p_list=(8, 10, 12),
    regimes=(("short", 0.75, 0.80), ("long", 0.95, 0.98)),
    trials=30,
)



    # --------- Compare Phi_hat_i to theoretical Phi_i = C (A-LC)^(i-1) L ---------

    F = A - L @ C
    Fpow = np.eye(d_x)   # (A-LC)^(i-1), starts at power 0

    phi_sq_errors = []

    for i in range(p):
        Phi_theory_i = C @ Fpow @ L               # C(A-LC)^(i-1) L
        diff = Phi_list[i] - Phi_theory_i
        err_sq = np.linalg.norm(diff, 'fro')**2   # squared Frobenius norm
        phi_sq_errors.append(err_sq)
        Fpow = Fpow @ F                           # next power

    print("Mean squared Phi error:", np.mean(phi_sq_errors))
    print("Var across lags of Phi error:", np.var(phi_sq_errors, ddof=1))



    plt.figure()
    plt.plot(range(1, p+1), phi_sq_errors, marker='o')
    plt.xlabel("Lag index i")
    plt.ylabel("Squared Frobenius error")
    plt.title(r"Squared error $\|\hat{\Phi}_i - C(A-LC)^{i-1}L\|_F^2$ vs lag")
    plt.grid(True)
    plt.show()


    # Predict and compute residuals 
    Y_hat = X @ B_hat
    residuals = Y - Y_hat
    mse = np.mean(residuals**2)
    print("Training MSE,", mse)

    # Plotting a single output dimension and its VAR prediction
    j = 1 # output component index
    t = np.arange(n)

    y_true = y[:, j]
    y_pred = np.full(n, np.nan)
    y_pred[p:] = Y_hat[:, j]

    plt.figure()
    plt.plot(t, y_true, label=f"y[:, {j}] true")
    plt.plot(t, y_pred, label=f"VAR({p}) prediction", linewidth=2)
    plt.title(f"Constrained LDS -> VAR({p}) via LS (output dim {j})")
    plt.xlabel("time index 'k'")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot residuals for specific component
    #plt.figure()
    #plt.plot(t[p:], residuals[:, j])
    #plt.title(f"Residuals (y - XB) for output dim {j}")
    #plt.xlabel("time index 'k'")
    #lt.ylabel("residual")
    #plt.tight_layout()
    #plt.show()


        # Short memory regime
    plot_phi_error_spread_per_lag_rhoband(
        regime_name="short",
        rho_low=0.75, rho_high=0.80,
        trials=30, n=1500, p=10, e_scale=0.2
    )

    # Long memory regime
    plot_phi_error_spread_per_lag_rhoband(
        regime_name="long",
        rho_low=0.95, rho_high=0.98,
        trials=30, n=1500, p=10, e_scale=0.2
)




if __name__ == "__main__":
    main()


        