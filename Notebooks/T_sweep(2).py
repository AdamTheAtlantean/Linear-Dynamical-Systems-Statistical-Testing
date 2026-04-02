import numpy as np
import matplotlib.pyplot as plt

from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import mahalanobis_var_distance
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

    return pi_hat, Sigma_hat, QX_hat


def run_realization_sensitivity(
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 5,
    trials: int = 120,
    n: int = 1500,
    p: int = 2,
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

    If diff_regime_y3 is None:
        y3 is drawn from [rho_min, rho_max]

    If diff_regime_y4 is None:
        y4 is drawn from [rho_min, rho_max]
    """
    rng = np.random.default_rng(seed)

    all_same_M, all_diff_M = [], []

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

        # SAME system
        A = sample_stable_A(d_x, rho_min, rho_max, rng)
        C = sample_C(d_y, d_x, rng)
        L = sample_L(d_x, d_y, rng)

        # DIFF system y3
        if diff_regime_y3 is None:
            rho_min3, rho_max3 = rho_min, rho_max
        else:
            rho_min3, rho_max3 = diff_regime_y3

        A3 = sample_stable_A(d_x, rho_min3, rho_max3, rng)
        C3 = sample_C(d_y, d_x, rng)
        L3 = sample_L(d_x, d_y, rng)

        # DIFF system y4
        if diff_regime_y4 is None:
            rho_min4, rho_max4 = rho_min, rho_max
        else:
            rho_min4, rho_max4 = diff_regime_y4

        A4 = sample_stable_A_identity_centered(d_x, rho_min4, rho_max4, rng)
        C4 = sample_C(d_y, d_x, rng)
        L4 = sample_L(d_x, d_y, rng)

        systems.append((A, C, L))
        diff_pairs.append(((A3, C3, L3), (A4, C4, L4)))

        D_same_M, D_diff_M = [], []
        cond_QX_list = []

        for _ in range(trials):
            # SAME
            y1 = simulate_y_only(n, A, C, L, rng, e_scale)
            y2 = simulate_y_only(n, A, C, L, rng, e_scale)

            pi1, Sigma1, QX1 = fit_var_and_components(y1, p)
            pi2, Sigma2, QX2 = fit_var_and_components(y2, p)

            cond_QX_list.append(np.linalg.cond(QX1))
            cond_QX_list.append(np.linalg.cond(QX2))

            d_same = mahalanobis_var_distance(pi1, pi2, Sigma1, Sigma2, QX1, QX2)
            D_same_M.append(d_same)

            # DIFFERENT
            y3 = simulate_y_only(n, A3, C3, L3, rng, e_scale)
            y4 = simulate_y_only(n, A4, C4, L4, rng, e_scale)

            pi3, Sigma3, QX3 = fit_var_and_components(y3, p)
            pi4, Sigma4, QX4 = fit_var_and_components(y4, p)

            cond_QX_list.append(np.linalg.cond(QX3))
            cond_QX_list.append(np.linalg.cond(QX4))

            d_diff = mahalanobis_var_distance(pi3, pi4, Sigma3, Sigma4, QX3, QX4)
            D_diff_M.append(d_diff)

        print("  Mahalanobis same mean:", float(np.mean(D_same_M)))
        print("  Mahalanobis same std:", float(np.std(D_same_M, ddof=1)))
        print("  Mahalanobis diff mean:", float(np.mean(D_diff_M)))
        print("  Mahalanobis diff std:", float(np.std(D_diff_M, ddof=1)))

        d_ir_pair = impulse_response_distance(
            A3, C3, L3,
            A4, C4, L4,
            K=K_ir,
            normalize=True
        )
        print("  IR distance of fixed DIFF pair:", d_ir_pair)

        cond_arr = np.array(cond_QX_list)
        print("  QX condition number (median):", np.median(cond_arr))
        print("  QX condition number (max):", np.max(cond_arr))

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

    # Across SAME outer-loop realizations
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

    # Within DIFF pairs
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

    return all_same_M, all_diff_M, ir_dists, diff_ir_dists, systems, diff_pairs



def run_T_sweep(
    T_values,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    p: int = 2,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    diff_regime_y3: tuple[float, float] | None = None,
    diff_regime_y4: tuple[float, float] | None = None,
    K_ir: int = 25,
    best_by: str = "f1",
):
    """
    Sweep over effective sample sizes T = n - p while fixing p.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1, got p={p}")

    sweep_results = []

    for T in T_values:
        if T <= 0:
            raise ValueError(f"T must be positive, got T={T}")

        n = T + p

        print("\n" + "=" * 70)
        print(f"Running T-sweep experiment for T = {T} (n = {n}, p = {p})")
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
            seed=seed,
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
            "T": int(T),
            "n": int(n),
            "p": int(p),
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

        print(f"\nSummary for T = {T} (n = {n}, p = {p})")
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


def plot_T_sweep_results(sweep_results):
    """
    Plot key classification metrics as functions of effective sample size T.
    """
    T_vals = np.array([r["T"] for r in sweep_results], dtype=int)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    fpr_vals = np.array([r["fpr"] for r in sweep_results], dtype=float)
    fnr_vals = np.array([r["fnr"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(T_vals, auc_vals, marker="o", linewidth=2)
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("AUC")
    plt.title("AUC vs effective sample size T")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(T_vals, f1_vals, marker="o", linewidth=2, label="F1")
    plt.plot(T_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(T_vals, 1 - fpr_vals, marker="o", linewidth=2, label="TNR")
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Score")
    plt.title("Classification metrics vs effective sample size T")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(T_vals, fpr_vals, marker="o", linewidth=2, label="FPR")
    plt.plot(T_vals, fnr_vals, marker="o", linewidth=2, label="FNR")
    plt.plot(T_vals, tpr_vals, marker="o", linewidth=2, linestyle="--", label="TPR")
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Rate")
    plt.title("Error rates vs effective sample size T")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(T_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs effective sample size T")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



def run_T_sweep_repeated(
    T_values,
    n_repeats: int,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    p: int = 2,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    base_seed: int = 0,
    diff_regime_y3: tuple[float, float] | None = None,
    diff_regime_y4: tuple[float, float] | None = None,
    K_ir: int = 25,
    best_by: str = "f1",
):
    """
    Repeat the full T-sweep n_repeats times with changing seeds and
    aggregate metrics across repetitions.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    all_runs = []

    for rep in range(n_repeats):
        print("\n" + "#" * 70)
        print(f"Repeated T-sweep run {rep + 1}/{n_repeats}")
        print("#" * 70)

        sweep_results = run_T_sweep(
            T_values=T_values,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            realizations=realizations,
            trials=trials,
            p=p,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            seed=base_seed + rep,
            diff_regime_y3=diff_regime_y3,
            diff_regime_y4=diff_regime_y4,
            K_ir=K_ir,
            best_by=best_by,
        )

        all_runs.append(sweep_results)

    summary_rows = []

    for j, T in enumerate(T_values):
        auc_vals = np.array([run[j]["auc"] for run in all_runs], dtype=float)
        f1_vals = np.array([run[j]["f1"] for run in all_runs], dtype=float)
        tpr_vals = np.array([run[j]["recall"] for run in all_runs], dtype=float)
        fpr_vals = np.array([run[j]["fpr"] for run in all_runs], dtype=float)
        fnr_vals = np.array([run[j]["fnr"] for run in all_runs], dtype=float)
        tau_vals = np.array([run[j]["best_threshold"] for run in all_runs], dtype=float)
        acc_vals = np.array([run[j]["accuracy"] for run in all_runs], dtype=float)
        prec_vals = np.array([run[j]["precision"] for run in all_runs], dtype=float)
        spec_vals = np.array([run[j]["specificity"] for run in all_runs], dtype=float)

        summary_rows.append({
            "T": int(T),
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


def plot_T_sweep_variance(summary_rows):
    """
    Plot mean ± std for repeated T-sweep experiments.
    """
    T_vals = np.array([r["T"] for r in summary_rows], dtype=int)

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
    plt.errorbar(T_vals, auc_mean, yerr=auc_std, marker="o", linewidth=2, capsize=4)
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("AUC")
    plt.title("T-sweep: AUC mean ± std")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(T_vals, f1_mean, yerr=f1_std, marker="o", linewidth=2, capsize=4, label="F1")
    plt.errorbar(T_vals, tpr_mean, yerr=tpr_std, marker="o", linewidth=2, capsize=4, label="TPR")
    plt.errorbar(T_vals, 1 - fpr_mean, yerr=fpr_std, marker="o", linewidth=2, capsize=4, label="TNR")
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Score")
    plt.title("T-sweep: classification metrics mean ± std")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(T_vals, fpr_mean, yerr=fpr_std, marker="o", linewidth=2, capsize=4, label="FPR")
    plt.errorbar(T_vals, fnr_mean, yerr=fnr_std, marker="o", linewidth=2, capsize=4, label="FNR")
    plt.errorbar(T_vals, tpr_mean, yerr=tpr_std, marker="o", linewidth=2, capsize=4, linestyle="--", label="TPR")
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Rate")
    plt.title("T-sweep: error rates mean ± std")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(T_vals, tau_mean, yerr=tau_std, marker="o", linewidth=2, capsize=4)
    plt.xlabel("Effective sample size T = n - p")
    plt.ylabel("Best threshold")
    plt.title("T-sweep: best threshold mean ± std")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_T_sweep_summary_table(sweep_results):
    """
    Print single-run T-sweep summary.
    """
    print("\n" + "=" * 70)
    print("T-SWEEP SUMMARY TABLE")
    print("=" * 70)
    for row in sweep_results:
        print(
            f"T={row['T']:>4d} | "
            f"n={row['n']:>4d} | "
            f"AUC={row['auc']:.4f} | "
            f"F1={row['f1']:.4f} | "
            f"TPR={row['recall']:.4f} | "
            f"FPR={row['fpr']:.4f} | "
            f"tau*={row['best_threshold']:.4f}"
        )


def print_T_sweep_variance_summary_table(summary_rows, n_repeats: int):
    """
    Print repeated-run T-sweep summary with mean ± std.
    """
    print("\n" + "=" * 90)
    print(f"T-SWEEP SUMMARY TABLE ({n_repeats} REPEATS)")
    print("=" * 90)
    for row in summary_rows:
        print(
            f"T={row['T']:>4d} | "
            f"AUC mean={row['auc_mean']:.4f} | AUC std={row['auc_std']:.4f} | "
            f"F1 mean={row['f1_mean']:.4f} | F1 std={row['f1_std']:.4f}"
        )


# Main
if __name__ == "__main__":


    # Choose which experiment to run
    RUN_SINGLE_SWEEP = True
    RUN_REPEATED_SWEEP = False

    # Shared configuration
    T_values = [100, 200, 400, 600, 800, 1000, 1200, 1498]

    regime_name = "short (within-regime diff)"
    rho_min = 0.75
    rho_max = 0.85

    realizations = 25
    trials = 40
    p_fixed = 2
    d_x = 5
    d_y = 5
    e_scale = 0.2
    K_ir = 25
    best_by = "f1"

    diff_regime_y3 = (0.3, 0.4)
    diff_regime_y4 = (0.9, 0.98)

    
    # Single sweep
    if RUN_SINGLE_SWEEP:
        sweep_results = run_T_sweep(
            T_values=T_values,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            realizations=realizations,
            trials=trials,
            p=p_fixed,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            seed=0,
            diff_regime_y3=diff_regime_y3,
            diff_regime_y4=diff_regime_y4,
            K_ir=K_ir,
            best_by=best_by,
        )

        print_T_sweep_summary_table(sweep_results)
        plot_T_sweep_results(sweep_results)

    
    # Repeated sweep for variance study
    if RUN_REPEATED_SWEEP:
        n_repeats = 100

        all_runs, summary_rows = run_T_sweep_repeated(
            T_values=T_values,
            n_repeats=n_repeats,
            regime_name=regime_name,
            rho_min=rho_min,
            rho_max=rho_max,
            realizations=realizations,
            trials=trials,
            p=p_fixed,
            d_x=d_x,
            d_y=d_y,
            e_scale=e_scale,
            base_seed=0,
            diff_regime_y3=diff_regime_y3,
            diff_regime_y4=diff_regime_y4,
            K_ir=K_ir,
            best_by=best_by,
        )

        print_T_sweep_variance_summary_table(summary_rows, n_repeats=n_repeats)
        plot_T_sweep_variance(summary_rows)