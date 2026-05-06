import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed, parallel_config

from lds import simulate_y_only
from var_model import fit_var_and_components
from metrics import isotropic_var_distance
from threshold import summarize_threshold_analysis
from linalg_utils import spectral_radius, effective_F
from sampling_utils import sample_system_with_F_in_band
from IR_utils import impulse_response_distance

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
5. Phi_i      ∈ R^{d_y, d_y}
6. pi_hat     ∈ R^{(p * d_y^2), 1}

Theory-aligned effective memory matrix:
1. F = A - L C        ∈ R^{d_x, d_x}
2. rho(F)             controls decay of the VAR coefficients
3. Phi_i              = C F^{i-1} L
4. H_k                = C F^{k-1} L

Parallelization notes
---------------------
To preserve EXACTLY the same numerical results as the serial version,
we only parallelize over:
1. p values in run_p_sweep(...)
2. repeat index r in run_p_sweep_with_variance(...)

We do NOT parallelize the inner realization/trial loops, because those
consume one shared RNG stream in sequence.

Also fixed from the original:
1. The variance aggregation indentation bug
2. np.smean(...) typo -> np.mean(...)
"""




# ============================================================================
# KDE fallback and plotting
# ============================================================================

def _kde_fallback(xs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Simple Gaussian-mixture KDE fallback for near-singular cases.
    """
    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    n = dist.size

    if n == 0:
        return np.zeros_like(xs)

    if n == 1:
        h = 1e-3
    else:
        std = np.std(dist, ddof=1) + 1e-12
        h = 1.06 * std * n ** (-1 / 5)

    ys = np.mean(
        np.exp(-0.5 * ((xs[:, None] - dist[None, :]) / h) ** 2),
        axis=1
    ) / (h * np.sqrt(2 * np.pi))

    return ys


def plot_kde_overlay(distributions, title: str, gridsize: int = 400, bw_method=None, xlim=None, tau=None):
    """
    Plot one KDE per realization on the same graph.
    """
    plt.figure(figsize=(8, 5))

    dists = []
    for d in distributions:
        arr = np.asarray(d, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            dists.append(arr)

    if len(dists) == 0:
        print(f"[plot_kde_overlay] Nothing to plot for: {title}")
        return

    all_vals = np.concatenate(dists)

    if xlim is None:
        lo, hi = np.quantile(all_vals, [0.001, 0.999])
        xs = np.linspace(lo, hi, gridsize)
        plt.xlim(lo, hi)
    else:
        xs = np.linspace(xlim[0], xlim[1], gridsize)
        plt.xlim(xlim[0], xlim[1])

    for i, dist in enumerate(distributions):
        dist = np.asarray(dist, dtype=float)
        dist = dist[np.isfinite(dist)]
        if dist.size < 2 or np.std(dist) == 0:
            continue

        try:
            ys = gaussian_kde(dist, bw_method=bw_method)(xs)
        except Exception:
            ys = _kde_fallback(xs, dist)

        plt.plot(xs, ys, alpha=0.75, label=f"Realization {i+1}")

    plt.title(title)
    plt.xlabel("Isotropic Distance")
    plt.ylabel("Estimated Density")
    plt.grid(alpha=0.3)
    if tau is not None:
        plt.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_p_sweep_results(sweep_results):
    """
    Plot key classification metrics as functions of VAR order p.
    """
    p_vals = np.array([r["p"] for r in sweep_results], dtype=int)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    fpr_vals = np.array([r["fpr"] for r in sweep_results], dtype=float)
    fnr_vals = np.array([r["fnr"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, auc_vals, marker="o", linewidth=2)
    plt.xlabel("VAR order p")
    plt.ylabel("AUC")
    plt.title("AUC vs VAR order p (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, f1_vals, marker="o", linewidth=2, label="F1")
    plt.plot(p_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(p_vals, 1 - fpr_vals, marker="o", linewidth=2, label="TNR")
    plt.xlabel("VAR order p")
    plt.ylabel("Score")
    plt.title("Classification metrics vs VAR order p (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, fpr_vals, marker="o", linewidth=2, label="FPR")
    plt.plot(p_vals, fnr_vals, marker="o", linewidth=2, label="FNR")
    plt.plot(p_vals, tpr_vals, marker="o", linewidth=2, linestyle="--", label="TPR")
    plt.xlabel("VAR order p")
    plt.ylabel("Rate")
    plt.title("Error Rates vs VAR order p (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel("VAR order p")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs VAR order p (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_with_variance(summary):
    """
    Plot mean ± std across repeated p-sweeps.
    """
    p_vals = np.array([r["p"] for r in summary])

    auc_mean = np.array([r["auc_mean"] for r in summary])
    auc_std = np.array([r["auc_std"] for r in summary])

    f1_mean = np.array([r["f1_mean"] for r in summary])
    f1_std = np.array([r["f1_std"] for r in summary])

    thresh_mean = np.array([r["threshold_mean"] for r in summary])
    thresh_std = np.array([r["threshold_std"] for r in summary])

    plt.figure(figsize=(7, 4))
    plt.errorbar(p_vals, auc_mean, yerr=auc_std, marker="o", capsize=4)
    plt.xlabel("VAR order p")
    plt.ylabel("AUC")
    plt.title("AUC vs p (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(p_vals, f1_mean, yerr=f1_std, marker="o", capsize=4)
    plt.xlabel("VAR order p")
    plt.ylabel("F1")
    plt.title("F1 vs p (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(p_vals, thresh_mean, yerr=thresh_std, marker="o", capsize=4)
    plt.xlabel("VAR order p")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs p (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# Core experiment

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
    a_rho_max: float = 0.98,
    c_scale: float = 1.0,
    l_scale: float = 1.0,
):
    """
    Outer loop over fixed realizations.
    Inner loop over Monte Carlo trials.

    SAME condition:
        y1 vs y2 from the same fixed realization (A, C, L)

    DIFFERENT condition:
        y3 vs y4 from two fixed but distinct and independent LDS realizations

    If diff_regime_y3 is None:
        y3 is drawn from [rho_min, rho_max] in rho(F)

    If diff_regime_y4 is None:
        y4 is drawn from [rho_min, rho_max] in rho(F)
    """
    rng = np.random.default_rng(seed)

    all_same_M, all_diff_M = [], []
    all_same_delta_pi = []
    all_diff_delta_pi = []

    print(f"\n--- Regime: {regime_name}  rho(F) in [{rho_min}, {rho_max}] ---\n")

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
        A, C, L = sample_system_with_F_in_band(
            d_x=d_x,
            d_y=d_y,
            rho_min=rho_min,
            rho_max=rho_max,
            rng=rng,
            a_rho_max=a_rho_max,
            c_scale=c_scale,
            l_scale=l_scale,
        )

        # DIFF system y3
        if diff_regime_y3 is None:
            rho_min3, rho_max3 = rho_min, rho_max
        else:
            rho_min3, rho_max3 = diff_regime_y3

        A3, C3, L3 = sample_system_with_F_in_band(
            d_x=d_x,
            d_y=d_y,
            rho_min=rho_min3,
            rho_max=rho_max3,
            rng=rng,
            a_rho_max=a_rho_max,
            c_scale=c_scale,
            l_scale=l_scale,
        )

        # DIFF system y4
        if diff_regime_y4 is None:
            rho_min4, rho_max4 = rho_min, rho_max
        else:
            rho_min4, rho_max4 = diff_regime_y4

        A4, C4, L4 = sample_system_with_F_in_band(
            d_x=d_x,
            d_y=d_y,
            rho_min=rho_min4,
            rho_max=rho_max4,
            rng=rng,
            a_rho_max=a_rho_max,
            c_scale=c_scale,
            l_scale=l_scale,
        )

        systems.append((A, C, L))
        diff_pairs.append(((A3, C3, L3), (A4, C4, L4)))

        print(
            f"  rho(F_same)={spectral_radius(effective_F(A, C, L)):.4f}, "
            f"rho(A_same)={spectral_radius(A):.4f}"
        )
        print(
            f"  rho(F_y3)={spectral_radius(effective_F(A3, C3, L3)):.4f}, "
            f"rho(A_y3)={spectral_radius(A3):.4f}"
        )
        print(
            f"  rho(F_y4)={spectral_radius(effective_F(A4, C4, L4)):.4f}, "
            f"rho(A_y4)={spectral_radius(A4):.4f}"
        )

        D_same_M, D_diff_M = [], []

        for _ in range(trials):
            # SAME
            try:
                y1 = simulate_y_only(n, A, C, L, rng, e_scale)
                y2 = simulate_y_only(n, A, C, L, rng, e_scale)

                pi1 = fit_var_and_components(y1, p)
                pi2 = fit_var_and_components(y2, p)

                delta_same = (pi1 - pi2).ravel()
                d_same = isotropic_var_distance(pi1, pi2)

                if np.all(np.isfinite(delta_same)):
                    all_same_delta_pi.append(delta_same)
                if np.isfinite(d_same):
                    D_same_M.append(float(d_same))

            except Exception:
                pass

            # DIFFERENT
            try:
                y3 = simulate_y_only(n, A3, C3, L3, rng, e_scale)
                y4 = simulate_y_only(n, A4, C4, L4, rng, e_scale)

                pi3 = fit_var_and_components(y3, p)
                pi4 = fit_var_and_components(y4, p)

                delta_diff = (pi3 - pi4).ravel()
                d_diff = isotropic_var_distance(pi3, pi4)

                if np.all(np.isfinite(delta_diff)):
                    all_diff_delta_pi.append(delta_diff)
                if np.isfinite(d_diff):
                    D_diff_M.append(float(d_diff))

            except Exception:
                pass

        if len(D_same_M) == 0 or len(D_diff_M) == 0:
            print("  Warning: one of the distance lists is empty for this realization.")
            print()
            all_same_M.append(D_same_M)
            all_diff_M.append(D_diff_M)
            continue

        print("  isotropic same mean:", float(np.mean(D_same_M)))
        print("  isotropic same std:", float(np.std(D_same_M, ddof=1)) if len(D_same_M) > 1 else 0.0)
        print("  isotropic diff mean:", float(np.mean(D_diff_M)))
        print("  isotropic diff std:", float(np.std(D_diff_M, ddof=1)) if len(D_diff_M) > 1 else 0.0)

        d_ir_pair = impulse_response_distance(
            A3, C3, L3,
            A4, C4, L4,
            K=K_ir,
            normalize=True
        )
        print("  IR distance of fixed DIFF pair:", d_ir_pair)

        same = np.asarray(D_same_M, dtype=float)
        diff = np.asarray(D_diff_M, dtype=float)

        print("  SAME q50/q90/q99:", np.quantile(same, [0.5, 0.9, 0.99]))
        print("  DIFF q50/q90/q99:", np.quantile(diff, [0.5, 0.9, 0.99]))
        print("  Pr(DIFF > SAME):", float(np.mean(diff > same)))
        prob_cross = np.mean(diff[:, None] > same[None, :])
        print("  Cross Prob.:", prob_cross)
        print()

        all_same_M.append(D_same_M)
        all_diff_M.append(D_diff_M)

    # Pairwise IR across SAME realizations
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

    # IR within DIFF pairs
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
        print(
            "  max/min ratio:",
            float(np.max(same_var_per_coord) / (np.min(same_var_per_coord) + 1e-12))
        )

    if len(all_diff_delta_pi) > 0:
        diff_delta_arr = np.asarray(all_diff_delta_pi)
        diff_var_per_coord = np.var(diff_delta_arr, axis=0)

        print("\nDIFF Δπ coordinate variance summary:")
        print("  min:", float(np.min(diff_var_per_coord)))
        print("  median:", float(np.median(diff_var_per_coord)))
        print("  max:", float(np.max(diff_var_per_coord)))
        print(
            "  max/min ratio:",
            float(np.max(diff_var_per_coord) / (np.min(diff_var_per_coord) + 1e-12))
        )

    return all_same_M, all_diff_M, ir_dists, diff_ir_dists, systems, diff_pairs



# Parallel-safe wrappers

def _run_single_p_value(
    p,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    diff_regime_y3: tuple[float, float] | None = None,
    diff_regime_y4: tuple[float, float] | None = None,
    K_ir: int = 25,
    best_by: str = "f1",
    a_rho_max: float = 0.98,
    c_scale: float = 1.0,
    l_scale: float = 1.0,
):
    print("\n" + "=" * 70)
    print(f"Running isotropic p-sweep experiment for p = {p}")
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
        a_rho_max=a_rho_max,
        c_scale=c_scale,
        l_scale=l_scale,
    )

    summary = summarize_threshold_analysis(
        same_M,
        diff_M,
        best_by=best_by,
        make_plots=False,
    )

    best = summary["best"]

    row = {
        "p": p,
        "auc": summary["auc"],
        "best_threshold": best["threshold"],
        "accuracy": best["accuracy"],
        "precision": best["precision"],
        "recall": best["recall"],
        "specificity": best["specificity"],
        "fpr": best["fpr"],
        "fnr": best["fnr"],
        "f1": best["f1"],
        "TP": best["TP"],
        "FP": best["FP"],
        "TN": best["TN"],
        "FN": best["FN"],
    }

    print("\nSummary for p =", p)
    print(f"  AUC            = {row['auc']:.6f}")
    print(f"  Best threshold = {row['best_threshold']:.6f}")
    print(f"  Accuracy       = {row['accuracy']:.6f}")
    print(f"  Precision      = {row['precision']:.6f}")
    print(f"  Recall / TPR   = {row['recall']:.6f}")
    print(f"  Specificity    = {row['specificity']:.6f}")
    print(f"  FPR            = {row['fpr']:.6f}")
    print(f"  FNR            = {row['fnr']:.6f}")
    print(f"  F1             = {row['f1']:.6f}")

    return row


def _run_single_repeat_p(
    r: int,
    p_values,
    **kwargs
):
    print(f"\n=== Repeat {r+1} ===")
    return r, run_p_sweep(
        p_values=p_values,
        seed=r,
        **kwargs
    )



# p-sweep

def run_p_sweep(
    p_values,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    diff_regime_y3: tuple[float, float] | None = None,
    diff_regime_y4: tuple[float, float] | None = None,
    K_ir: int = 25,
    best_by: str = "f1",
    n_jobs: int = 1,
    verbose_joblib: int = 0,
    a_rho_max: float = 0.98,
    c_scale: float = 1.0,
    l_scale: float = 1.0,
):
    """
    Sweep over VAR orders p and record pooled classification performance.

    Parallel over p values only.

    Regimes are defined by rho(F), where F = A - L C.
    """
    if n_jobs == 1:
        sweep_results = []
        for p in p_values:
            row = _run_single_p_value(
                p=p,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                d_x=d_x,
                d_y=d_y,
                e_scale=e_scale,
                seed=seed,
                diff_regime_y3=diff_regime_y3,
                diff_regime_y4=diff_regime_y4,
                K_ir=K_ir,
                best_by=best_by,
                a_rho_max=a_rho_max,
                c_scale=c_scale,
                l_scale=l_scale,
            )
            sweep_results.append(row)
        return sweep_results

    with parallel_config(backend="loky", inner_max_num_threads=1):
        sweep_results = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
            delayed(_run_single_p_value)(
                p=p,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                d_x=d_x,
                d_y=d_y,
                e_scale=e_scale,
                seed=seed,
                diff_regime_y3=diff_regime_y3,
                diff_regime_y4=diff_regime_y4,
                K_ir=K_ir,
                best_by=best_by,
                a_rho_max=a_rho_max,
                c_scale=c_scale,
                l_scale=l_scale,
            )
            for p in p_values
        )

    sweep_results.sort(key=lambda row: p_values.index(row["p"]))
    return sweep_results


def run_p_sweep_with_variance(
    p_values,
    n_repeats=100,
    n_jobs: int = 1,
    verbose_joblib: int = 0,
    **kwargs
):
    """
    Repeat the full p-sweep experiment multiple times and compute mean + std.

    Parallel over repeat index r only.
    """
    all_results = {p: {"auc": [], "f1": [], "best_threshold": []} for p in p_values}

    if n_jobs == 1:
        repeat_outputs = []
        for r in range(n_repeats):
            print(f"\n=== Repeat {r+1}/{n_repeats} ===")
            results = run_p_sweep(
                p_values=p_values,
                seed=r,
                n_jobs=1,
                **kwargs
            )
            repeat_outputs.append((r, results))
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            repeat_outputs = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
                delayed(_run_single_repeat_p)(
                    r=r,
                    p_values=p_values,
                    n_jobs=1,
                    **kwargs
                )
                for r in range(n_repeats)
            )

    repeat_outputs.sort(key=lambda x: x[0])

    for _, results in repeat_outputs:
        for row in results:
            p = row["p"]
            all_results[p]["auc"].append(row["auc"])
            all_results[p]["f1"].append(row["f1"])
            all_results[p]["best_threshold"].append(row["best_threshold"])

    summary = []

    for p in p_values:
        auc_arr = np.array(all_results[p]["auc"])
        f1_arr = np.array(all_results[p]["f1"])
        thresh_arr = np.array(all_results[p]["best_threshold"])

        summary.append({
            "p": p,
            "auc_mean": np.mean(auc_arr),
            "auc_std": np.std(auc_arr),
            "f1_mean": np.mean(f1_arr),
            "f1_std": np.std(f1_arr),
            "threshold_mean": np.mean(thresh_arr),
            "threshold_std": np.std(thresh_arr)
        })

    return summary



# Main toggle

if __name__ == "__main__":

    USE_VARIANCE_SWEEP = False

    N_JOBS_SINGLE = 6
    N_JOBS_VARIANCE = 6

    p_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    common_kwargs = dict(
        regime_name="short (within-regime diff, F-band)",
        rho_min=0.75,
        rho_max=0.85,
        realizations=25,
        trials=40,
        n=200,
        d_x=5,
        d_y=5,
        e_scale=0.2,
        diff_regime_y3=(0.75, 0.85),
        diff_regime_y4=(0.8, 0.9),
        K_ir=25,
        best_by="f1",
        a_rho_max=0.9995,
        c_scale=1.0,
        l_scale=1.0,
    )

    if USE_VARIANCE_SWEEP:
        summary = run_p_sweep_with_variance(
            p_values=p_values,
            n_repeats=100,
            n_jobs=N_JOBS_VARIANCE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 90)
        print("ISOTROPIC P-SWEEP SUMMARY TABLE (100 REPEATS, F-BAND)")
        print("=" * 90)
        for row in summary:
            print(
                f"p={row['p']:>2d} | "
                f"AUC mean={row['auc_mean']:.4f} | "
                f"AUC std={row['auc_std']:.4f} | "
                f"F1 mean={row['f1_mean']:.4f} | "
                f"F1 std={row['f1_std']:.4f} | "
                f"tau mean={row['threshold_mean']:.4f} | "
                f"tau std={row['threshold_std']:.4f}"
            )

        plot_with_variance(summary)

    else:
        sweep_results = run_p_sweep(
            p_values=p_values,
            seed=0,
            n_jobs=N_JOBS_SINGLE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 90)
        print("ISOTROPIC P-SWEEP SUMMARY TABLE (SINGLE RUN, F-BAND)")
        print("=" * 90)
        for row in sweep_results:
            print(
                f"p={row['p']:>2d} | "
                f"AUC={row['auc']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"TPR={row['recall']:.4f} | "
                f"FPR={row['fpr']:.4f} | "
                f"tau*={row['best_threshold']:.4f}"
            )

        plot_p_sweep_results(sweep_results)