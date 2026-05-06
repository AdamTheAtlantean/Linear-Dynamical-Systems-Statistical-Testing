import numpy as np
import matplotlib.pyplot as plt

from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import mahalanobis_var_distance, isotropic_var_distance
from threshold import summarize_threshold_analysis

"""
Global dimension conventions:
1. n      = length of the observed time series
2. T      = effective VAR sample size, i.e., n - p
3. d_x    = latent state dimension
4. d_y    = observation dimension
5. p      = VAR order

LDS objects:
1. A      ∈ R^{d_x, d_x}
2. C      ∈ R^{d_y, d_x}
3. L      ∈ R^{d_x, d_y}
4. x_t    ∈ R^{d_x}
5. y_t    ∈ R^{d_y}
6. e_t    ∈ R^{d_y}

VAR(p) objects:
1. X          ∈ R^{T, (p * d_y)}
2. Y          ∈ R^{T, d_y}
3. B_hat      ∈ R^{(p * d_y), d_y}
4. U_hat      ∈ R^{T, d_y}
5. QX_hat     ∈ R^{(p * d_y), (p * d_y)}
6. Sigma_hat  ∈ R^{d_y, d_y}
7. pi_hat     ∈ R^{p * d_y^2}   (stored in NumPy as a 1D array)

IR response objects:
1. A^k               ∈ R^{d_x, d_x}
2. H_k = C A^{k-1} L ∈ R^{d_y, d_y}

Regime-sweep parameterization:
1. Left DIFFERENT regime:
       R3 = [a, a + w]
2. Right DIFFERENT regime:
       R4(Delta) = [a + w + Delta, a + 2w + Delta]
3. Therefore:
       gap = Delta
       both regime widths = w
"""



def sample_stable_A(
    d_x: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample A ∈ R^{d_x, d_x} and rescale so its spectral radius lies in [rho_min, rho_max].
    """
    A_raw = rng.normal(size=(d_x, d_x))
    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)
    return (rho_target / (rho_raw + 1e-12)) * A_raw


def sample_stable_A_identity_centered(
    d_x: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator,
    beta: float = 1.0,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """
    Sample A_raw = beta * I + noise_scale * Z, where Z has i.i.d. standard normal
    entries, then rescale so rho(A) lies within [rho_min, rho_max].
    """
    Z = rng.normal(size=(d_x, d_x))
    A_raw = beta * np.eye(d_x) + noise_scale * Z

    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)

    A = (rho_target / (rho_raw + 1e-12)) * A_raw
    return A


def sample_C(d_y: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    """Sample observation matrix C ∈ R^{d_y, d_x}."""
    return rng.normal(size=(d_y, d_x))


def sample_L(d_x: int, d_y: int, rng: np.random.Generator) -> np.ndarray:
    """Sample noise injection matrix L ∈ R^{d_x, d_y}."""
    return rng.normal(size=(d_x, d_y))



def impulse_response_matrices(A: np.ndarray, C: np.ndarray, L: np.ndarray, K: int):
    """
    Return [H_1, ..., H_K] where H_k = C A^{k-1} L.
    """
    d_x = A.shape[0]
    Ak = np.eye(d_x)
    H = []

    for _ in range(K):
        H.append(C @ Ak @ L)
        Ak = Ak @ A

    return H


def impulse_response_distance(
    A1, C1, L1,
    A2, C2, L2,
    K: int = 25,
    normalize: bool = True
) -> float:
    """
    D_H = sum_{k=1..K} ||H_k^{(1)} - H_k^{(2)}||_F^2

    If normalize=True, divide by sum_k ||H_k^{(1)}||_F^2.
    """
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



# Simulation wrapper
def simulate_y_only(
    n: int,
    A: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    rng: np.random.Generator,
    e_scale: float
) -> np.ndarray:
    """
    Extract y from simulate_lds return.
    Assumes simulate_lds returns (x, y, e), so y is index 1.
    """
    out = simulate_lds(n, A, C, L, rng, e_scale)
    return out[1] if isinstance(out, tuple) else out   # y ∈ R^{n, d_y}



# VAR fit 
def fit_var_and_components(y: np.ndarray, p: int):
    """
    Fit a VAR(p) model via least squares and return the components needed
    for the Mahalanobis distance between VAR parameter vectors.

    Returns
    -------
    pi_hat : np.ndarray
        Vectorized VAR coefficients, shape (p * d_y^2,)
    Sigma_hat : np.ndarray
        Residual covariance matrix, shape (d_y, d_y)
    QX_hat : np.ndarray
        Regressor covariance matrix, shape ((p * d_y), (p * d_y))
    """
    X, Y = build_var_xy(y, p=p)
    T_eff = X.shape[0]

    if T_eff <= 0:
        raise ValueError(f"Empty VAR design matrix: got T={T_eff}. Increase n or decrease p.")

    B_hat = fit_ls(Y, X)
    U_hat = Y - X @ B_hat

    Sigma_hat = (U_hat.T @ U_hat) / T_eff
    QX_hat = (X.T @ X) / T_eff

    d_y = Y.shape[1]
    Phi_list = unpack_B_to_Phi(B_hat, p=p, d_y=d_y)
    pi_hat = np.concatenate([Phi.flatten(order="F") for Phi in Phi_list])

    return pi_hat



# Main experiment: realization sensitivity
def run_realization_sensitivity(
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 5,
    trials: int = 120,
    n: int = 1500,
    p: int = 10,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    diff_regime_y3: tuple[float, float] | None = None, 
    diff_regime_y4: tuple[float, float] | None = None,
    K_ir: int = 25,
):
    """
    Outer loop over fixed realizations.
    Inner loop over Monte Carlo trials.

    SAME condition:
        y1 vs y2 from the same fixed realization (A, C, L)

    DIFFERENT condition:
        y3 vs y4 from two independently sampled LDS realizations
        (no reuse of y1)

    If diff_regime is None:
        y3 and y4 are both drawn from the same regime [rho_min, rho_max].

    If diff_regime is not None:
        y3 is drawn from [rho_min, rho_max]
        y4 is drawn from diff_regime
    """
    rng = np.random.default_rng(seed)

    all_same_M, all_diff_M = [], []

    all_same_delta_pi = []
    all_diff_delta_pi = []
    

    print(f"\n--- Regime: {regime_name}  rho in [{rho_min}, {rho_max}] ---\n")

    systems: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    diff_pairs: list[
        tuple[
            tuple[np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray]

        ]
    ] = []

    for r in range(realizations):
        print(f"Realization {r+1}/{realizations}")

        # Fixed realization for SAME comparisons in this outer loop
        A = sample_stable_A(d_x, rho_min, rho_max, rng)              # A: (d_x, d_x)
        C = sample_C(d_y, d_x, rng)                                  # C: (d_y, d_x)
        L = sample_L(d_x, d_y, rng)                                  # L(1,2): (d_x, d_y)

        # System for y3
        if diff_regime_y3 is None:
            rho_min3, rho_max3 = rho_min, rho_max
        else:
            rho_min3, rho_max3 = diff_regime_y3


        A3 = sample_stable_A(d_x, rho_min3, rho_max3, rng)           # A3: (d_x, d_x)
        C3 = sample_C(d_y, d_x, rng)                                 # C3: (d_y, d_x)
        L3 = sample_L(d_x, d_y, rng)                                 # L3: (d_x, d_y)
  
        # System for y4
        if diff_regime_y4 is None:
            rho_min4, rho_max4 = rho_min, rho_max
        else:
            rho_min4, rho_max4 = diff_regime_y4

        A4 = sample_stable_A_identity_centered(d_x, rho_min4, rho_max4, rng)   # A4: (d_x, d_x)
        C4 = sample_C(d_y, d_x, rng)                                           # C4: (d_y, d_x)
        L4 = sample_L(d_x, d_y, rng)                                           # L4: (d_x, d_y)

        systems.append((A, C, L))
        diff_pairs.append(((A3, C3, L3), (A4, C4, L4)))

        D_same_M, D_diff_M = [], [] # vectors of scalars to be populated / appended below 
        cond_QX_list = []

        for _ in range(trials):
            
            # SAME LDS: y1 vs y2 from the same fixed system
            y1 = simulate_y_only(n, A, C, L, rng, e_scale)   # y1: (n, d_y)
            y2 = simulate_y_only(n, A, C, L, rng, e_scale)   # y2: (n, d_y)

            pi1 = fit_var_and_components(y1, p) # pi1: (p * d_y^2, 1)
            pi2 = fit_var_and_components(y2, p) # pi2: (p * d_y^2, 1)

            all_same_delta_pi.append((pi1 - pi2).ravel())


            d_same = isotropic_var_distance(pi1, pi2) # scalar distance
            D_same_M.append(d_same)

            
            # DIFFERENT LDS: y3 vs y4 from two fixed but distinct and independent LDS realizations 
            y3 = simulate_y_only(n, A3, C3, L3, rng, e_scale) # y3: (n, d_y)
            y4 = simulate_y_only(n, A4, C4, L4, rng, e_scale) # y4: (n, d_y)

            pi3 = fit_var_and_components(y3, p) # pi3: (p * d, y^2)
            pi4 = fit_var_and_components(y4, p) # pi4: (p * d, y^2)

            all_diff_delta_pi.append((pi3 - pi4).ravel())


            d_diff = isotropic_var_distance(pi3, pi4)
            D_diff_M.append(d_diff)

        # Realization summary
        print("  isotropic same mean:", float(np.mean(D_same_M)))
        print("  isotropic same std:", float(np.std(D_same_M, ddof=1)))
        print("  isotropic diff mean:", float(np.mean(D_diff_M)))
        print("  isotropic diff std:", float(np.std(D_diff_M, ddof=1)))

        d_ir_pair = impulse_response_distance(
            A3, C3, L3,
            A4, C4, L4,
            K=K_ir,
            normalize=True
        )
        print("  IR distance of fixed DIFF pair:", d_ir_pair)



        same = np.asarray(D_same_M)
        diff = np.asarray(D_diff_M)

        print("  SAME q50/q90/q99:", np.quantile(same, [0.5, 0.9, 0.99]))
        print("  DIFF q50/q90/q99:", np.quantile(diff, [0.5, 0.9, 0.99]))
        print("  Pr(DIFF > SAME):", float(np.mean(diff > same)))
        prob_cross = np.mean(diff[:, None] > same[None, :])
        print("  Cross Prob.:", prob_cross)
        print()

        all_same_M.append(D_same_M)
        all_diff_M.append(D_diff_M)

    
    # Pairwise IR distances across fixed outer-loop realizations
    ir_dists: list[float] = []

    for i in range(len(systems)):
        A1, C1, L1 = systems[i]
        for j in range(i + 1, len(systems)):
            A2, C2, L2 = systems[j]
            ir_dists.append(
                impulse_response_distance(A1, C1, L1, A2, C2, L2, K=K_ir, normalize=True)
            )

    if len(ir_dists) == 0:
        print(f"IR response distance across realizations (K={K_ir}): (need at least 2 realizations)")
    else:
        ir_arr = np.asarray(ir_dists)
        std = np.std(ir_arr, ddof=1) if ir_arr.size >= 2 else 0.0
        print(
            f"IR response distance across realizations (K={K_ir}): "
            f"mean={np.mean(ir_arr):.4g}, std={std:.4g}, "
            f"min={np.min(ir_arr):.4g}, max={np.max(ir_arr):.4g}"
        )

    diff_ir_dists: list[float] = []

    for r in range(len(diff_pairs)):
        (A3, C3, L3), (A4, C4, L4) = diff_pairs[r]
        d_ir_diff = impulse_response_distance(
            A3, C3, L3,
            A4, C4, L4,
            K=K_ir,
            normalize=True
        )
        diff_ir_dists.append(d_ir_diff)

    if len(diff_ir_dists) == 0:
        print(f"IR distance within DIFF pairs (K={K_ir}): none")
    else:
        diff_ir_arr = np.asarray(diff_ir_dists)
        std = np.std(diff_ir_arr, ddof=1) if diff_ir_arr.size >= 2 else 0.0
        print(
            f"IR distance within DIFF pairs (K={K_ir}): "
            f"mean={np.mean(diff_ir_arr):.4g}, std={std:.4g}, "
            f"min={np.min(diff_ir_arr):.4g}, max={np.max(diff_ir_arr):.4g}"
        )
        
    if len(all_same_delta_pi) > 0:
        same_delta_arr = np.asarray(all_same_delta_pi)
        same_var_per_coord = np.var(same_delta_arr, axis=0)

        print("\nSAME Δπ coordinate variance summary:")
        print("  min:", float(np.min(same_var_per_coord)))
        print("  median:", float(np.median(same_var_per_coord)))
        print("  max:", float(np.max(same_var_per_coord)))
        print("  max/min ratio:", float(np.max(same_var_per_coord) / (np.min(same_var_per_coord) + 1e-12)))

    if len(all_diff_delta_pi) > 0:
        diff_delta_arr = np.asarray(all_diff_delta_pi)
        diff_var_per_coord = np.var(diff_delta_arr, axis=0)

        print("\nDIFF Δπ coordinate variance summary:")
        print("  min:", float(np.min(diff_var_per_coord)))
        print("  median:", float(np.median(diff_var_per_coord)))
        print("  max:", float(np.max(diff_var_per_coord)))
        print("  max/min ratio:", float(np.max(diff_var_per_coord) / (np.min(diff_var_per_coord) + 1e-12)))


    return all_same_M, all_diff_M, ir_dists, diff_ir_dists, systems, diff_pairs


# Regime helpers
def regimes_from_delta(a: float, w: float, delta: float):
    """
    Build the two DIFFERENT regime intervals from a single width w and gap delta.

    R3 = [a, a + w]
    R4 = [a + w + delta, a + 2w + delta]
    """
    if w <= 0:
        raise ValueError(f"w must be positive, got w={w}")
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got delta={delta}")

    r3 = (a, a + w)
    r4 = (a + w + delta, a + 2 * w + delta)
    return r3, r4


def validate_regime_bounds(
    a: float,
    w: float,
    delta_values,
    rho_upper_limit: float = 0.98
):
    """
    Ensure all generated right-hand regimes stay below rho_upper_limit.
    """
    for delta in delta_values:
        _, r4 = regimes_from_delta(a, w, delta)
        if r4[1] > rho_upper_limit + 1e-12:
            raise ValueError(
                f"Generated diff_regime_y4={r4} exceeds rho_upper_limit={rho_upper_limit}. "
                f"Adjust a, w, or delta_values."
            )



# Single regime sweep
def run_regime_sweep(
    delta_values,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    a: float,
    w: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 2,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    K_ir: int = 25,
    best_by: str = "f1",
    rho_upper_limit: float = 0.98,
):
    """
    Sweep over regime separation Delta while holding regime width w fixed.

    Left DIFFERENT regime:
        R3 = [a, a + w]

    Right DIFFERENT regime:
        R4(Delta) = [a + w + Delta, a + 2w + Delta]
    """
    validate_regime_bounds(a=a, w=w, delta_values=delta_values, rho_upper_limit=rho_upper_limit)

    sweep_results = []

    for k, delta in enumerate(delta_values):
        diff_regime_y3, diff_regime_y4 = regimes_from_delta(a=a, w=w, delta=delta)

        print("\n" + "=" * 70)
        print(
            f"Running regime-sweep experiment for Delta = {delta:.4f} | "
            f"y3={diff_regime_y3} | y4={diff_regime_y4}"
        )
        print("=" * 70)

        same_M, diff_M, ir_dists, diff_ir_dists, systems, diff_pairs = run_realization_sensitivity(
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            realizations=realizations,
            trials=trials,
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            seed=seed + k,
            diff_regime_y3=diff_regime_y3,
            diff_regime_y4=diff_regime_y4,
            K_ir=K_ir,
        )

        summary = summarize_threshold_analysis(
            same_M,
            diff_M,
            best_by=best_by,
            make_plots=False,
        )

        best = summary["best"]

        row = {
            "delta": float(delta),
            "diff_regime_y3": diff_regime_y3,
            "diff_regime_y4": diff_regime_y4,
            "auc": float(summary["auc"]),
            "best_threshold": float(best["threshold"]),
            "accuracy": float(best["accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "specificity": float(best["specificity"]),
            "fpr": float(best["fpr"]),
            "fnr": float(best["fnr"]),
            "f1": float(best["f1"]),
            "TP": int(best["TP"]),
            "FP": int(best["FP"]),
            "TN": int(best["TN"]),
            "FN": int(best["FN"]),
        }

        sweep_results.append(row)

        print(f"\nSummary for Delta = {delta:.4f}")
        print(f"  diff_regime_y3 = {diff_regime_y3}")
        print(f"  diff_regime_y4 = {diff_regime_y4}")
        print(f"  AUC            = {row['auc']:.6f}")
        print(f"  Best threshold = {row['best_threshold']:.6f}")
        print(f"  Accuracy       = {row['accuracy']:.6f}")
        print(f"  Precision      = {row['precision']:.6f}")
        print(f"  Recall / TPR   = {row['recall']:.6f}")
        print(f"  Specificity    = {row['specificity']:.6f}")
        print(f"  FPR            = {row['fpr']:.6f}")
        print(f"  FNR            = {row['fnr']:.6f}")
        print(f"  F1             = {row['f1']:.6f}")

    return sweep_results


def plot_regime_sweep_results(sweep_results):
    """
    Plot key classification metrics as functions of Delta.
    """
    delta_vals = np.array([r["delta"] for r in sweep_results], dtype=float)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    fpr_vals = np.array([r["fpr"] for r in sweep_results], dtype=float)
    fnr_vals = np.array([r["fnr"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(delta_vals, auc_vals, marker="o", linewidth=2)
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("AUC")
    plt.title(r"AUC vs regime gap $\Delta$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(delta_vals, f1_vals, marker="o", linewidth=2, label="F1")
    plt.plot(delta_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(delta_vals, 1 - fpr_vals, marker="o", linewidth=2, label="TNR")
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Score")
    plt.title(r"Classification metrics vs regime gap $\Delta$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(delta_vals, fpr_vals, marker="o", linewidth=2, label="FPR")
    plt.plot(delta_vals, fnr_vals, marker="o", linewidth=2, label="FNR")
    plt.plot(delta_vals, tpr_vals, marker="o", linewidth=2, linestyle="--", label="TPR")
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Rate")
    plt.title(r"Error rates vs regime gap $\Delta$")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(delta_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Best threshold")
    plt.title(r"Best threshold vs regime gap $\Delta$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Repeated regime sweep for variance study
def run_regime_sweep_repeated(
    delta_values,
    n_repeats: int,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    a: float,
    w: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 2,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    base_seed: int = 0,
    K_ir: int = 25,
    best_by: str = "f1",
    rho_upper_limit: float = 0.98,
):
    """
    Repeat the full regime sweep n_repeats times with changing seeds and
    aggregate metrics across repetitions.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    validate_regime_bounds(a=a, w=w, delta_values=delta_values, rho_upper_limit=rho_upper_limit)

    all_runs = []

    for rep in range(n_repeats):
        print("\n" + "#" * 70)
        print(f"Repeated regime-sweep run {rep + 1}/{n_repeats}")
        print("#" * 70)

        sweep_results = run_regime_sweep(
            delta_values=delta_values,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            a=a,
            w=w,
            realizations=realizations,
            trials=trials,
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            seed=base_seed + rep,
            K_ir=K_ir,
            best_by=best_by,
            rho_upper_limit=rho_upper_limit,
        )

        all_runs.append(sweep_results)

    summary_rows = []

    for j, delta in enumerate(delta_values):
        auc_vals = np.array([run[j]["auc"] for run in all_runs], dtype=float)
        f1_vals = np.array([run[j]["f1"] for run in all_runs], dtype=float)
        tpr_vals = np.array([run[j]["recall"] for run in all_runs], dtype=float)
        fpr_vals = np.array([run[j]["fpr"] for run in all_runs], dtype=float)
        fnr_vals = np.array([run[j]["fnr"] for run in all_runs], dtype=float)
        tau_vals = np.array([run[j]["best_threshold"] for run in all_runs], dtype=float)
        acc_vals = np.array([run[j]["accuracy"] for run in all_runs], dtype=float)
        prec_vals = np.array([run[j]["precision"] for run in all_runs], dtype=float)
        spec_vals = np.array([run[j]["specificity"] for run in all_runs], dtype=float)

        diff_regime_y3, diff_regime_y4 = regimes_from_delta(a=a, w=w, delta=delta)

        summary_rows.append({
            "delta": float(delta),
            "diff_regime_y3": diff_regime_y3,
            "diff_regime_y4": diff_regime_y4,
            "auc_mean": float(np.mean(auc_vals)),
            "auc_std": float(np.std(auc_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "f1_mean": float(np.mean(f1_vals)),
            "f1_std": float(np.std(f1_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "tpr_mean": float(np.mean(tpr_vals)),
            "tpr_std": float(np.std(tpr_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "fpr_mean": float(np.mean(fpr_vals)),
            "fpr_std": float(np.std(fpr_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "fnr_mean": float(np.mean(fnr_vals)),
            "fnr_std": float(np.std(fnr_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "tau_mean": float(np.mean(tau_vals)),
            "tau_std": float(np.std(tau_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "accuracy_mean": float(np.mean(acc_vals)),
            "accuracy_std": float(np.std(acc_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "precision_mean": float(np.mean(prec_vals)),
            "precision_std": float(np.std(prec_vals, ddof=1)) if n_repeats > 1 else 0.0,
            "specificity_mean": float(np.mean(spec_vals)),
            "specificity_std": float(np.std(spec_vals, ddof=1)) if n_repeats > 1 else 0.0,
        })

    return all_runs, summary_rows


def plot_regime_sweep_variance(summary_rows):
    """
    Plot mean ± std for repeated regime-sweep experiments.
    """
    delta_vals = np.array([r["delta"] for r in summary_rows], dtype=float)

    auc_mean = np.array([r["auc_mean"] for r in summary_rows], dtype=float)
    auc_std = np.array([r["auc_std"] for r in summary_rows], dtype=float)

    f1_mean = np.array([r["f1_mean"] for r in summary_rows], dtype=float)
    f1_std = np.array([r["f1_std"] for r in summary_rows], dtype=float)

    tpr_mean = np.array([r["tpr_mean"] for r in summary_rows], dtype=float)
    tpr_std = np.array([r["tpr_std"] for r in summary_rows], dtype=float)

    fpr_mean = np.array([r["fpr_mean"] for r in summary_rows], dtype=float)
    fpr_std = np.array([r["fpr_std"] for r in summary_rows], dtype=float)

    fnr_mean = np.array([r["fnr_mean"] for r in summary_rows], dtype=float)
    fnr_std = np.array([r["fnr_std"] for r in summary_rows], dtype=float)

    tau_mean = np.array([r["tau_mean"] for r in summary_rows], dtype=float)
    tau_std = np.array([r["tau_std"] for r in summary_rows], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.errorbar(delta_vals, auc_mean, yerr=auc_std, marker="o", linewidth=2, capsize=4)
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("AUC")
    plt.title(r"Regime sweep: AUC mean $\pm$ std")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(delta_vals, f1_mean, yerr=f1_std, marker="o", linewidth=2, capsize=4, label="F1")
    plt.errorbar(delta_vals, tpr_mean, yerr=tpr_std, marker="o", linewidth=2, capsize=4, label="TPR")
    plt.errorbar(delta_vals, 1 - fpr_mean, yerr=fpr_std, marker="o", linewidth=2, capsize=4, label="TNR")
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Score")
    plt.title(r"Regime sweep: classification metrics mean $\pm$ std")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(delta_vals, fpr_mean, yerr=fpr_std, marker="o", linewidth=2, capsize=4, label="FPR")
    plt.errorbar(delta_vals, fnr_mean, yerr=fnr_std, marker="o", linewidth=2, capsize=4, label="FNR")
    plt.errorbar(delta_vals, tpr_mean, yerr=tpr_std, marker="o", linewidth=2, capsize=4, linestyle="--", label="TPR")
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Rate")
    plt.title(r"Regime sweep: error rates mean $\pm$ std")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(delta_vals, tau_mean, yerr=tau_std, marker="o", linewidth=2, capsize=4)
    plt.xlabel(r"Gap $\Delta$")
    plt.ylabel("Best threshold")
    plt.title(r"Regime sweep: best threshold mean $\pm$ std")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# Printing helpers
def print_regime_sweep_summary_table(sweep_results):
    """
    Print single-run regime-sweep summary.
    """
    print("\n" + "=" * 100)
    print("REGIME-SWEEP SUMMARY TABLE")
    print("=" * 100)
    for row in sweep_results:
        print(
            f"Delta={row['delta']:.4f} | "
            f"y3={row['diff_regime_y3']} | "
            f"y4={row['diff_regime_y4']} | "
            f"AUC={row['auc']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"TPR={row['recall']:.4f} | "
            f"FPR={row['fpr']:.4f} | "
            f"tau*={row['best_threshold']:.4f}"
        )


def print_regime_sweep_variance_summary_table(summary_rows, n_repeats: int):
    """
    Print repeated-run regime-sweep summary with mean ± std.
    """
    print("\n" + "=" * 110)
    print(f"REGIME-SWEEP SUMMARY TABLE ({n_repeats} REPEATS)")
    print("=" * 110)
    for row in summary_rows:
        print(
            f"Delta={row['delta']:.4f} | "
            f"AUC mean={row['auc_mean']:.4f} | AUC std={row['auc_std']:.4f} | "
            f"F1 mean={row['f1_mean']:.4f} | F1 std={row['f1_std']:.4f}"
        )



# Main
if __name__ == "__main__":

    
    # Choose which experiment to run
    RUN_SINGLE_SWEEP = True
    RUN_REPEATED_SWEEP = False

    
    # Shared configuration
    delta_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.45, 0.48]

    # Left DIFFERENT regime: [a, a + w]
    a = 0.30
    w = 0.10

    regime_name = "short (within-regime diff)"
    rho_min = 0.75
    rho_max = 0.85

    realizations = 25
    trials = 40
    n = 1500
    p = 2
    d_x = 5
    d_y = 5
    e_scale = 0.2
    K_ir = 25
    best_by = "f1"

    # Upper limit to keep generated regime bands sensible
    rho_upper_limit = 0.98

    
    # Single sweep
    if RUN_SINGLE_SWEEP:
        sweep_results = run_regime_sweep(
            delta_values=delta_values,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            a=a,
            w=w,
            realizations=realizations,
            trials=trials,
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            seed=0,
            K_ir=K_ir,
            best_by=best_by,
            rho_upper_limit=rho_upper_limit,
        )

        print_regime_sweep_summary_table(sweep_results)
        plot_regime_sweep_results(sweep_results)

    
    # Repeated sweep for variance study
    if RUN_REPEATED_SWEEP:
        n_repeats = 100

        all_runs, summary_rows = run_regime_sweep_repeated(
            delta_values=delta_values,
            n_repeats=n_repeats,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            a=a,
            w=w,
            realizations=realizations,
            trials=trials,
            n=n,
            p=p,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            base_seed=0,
            K_ir=K_ir,
            best_by=best_by,
            rho_upper_limit=rho_upper_limit,
        )

        print_regime_sweep_variance_summary_table(summary_rows, n_repeats=n_repeats)
        plot_regime_sweep_variance(summary_rows)