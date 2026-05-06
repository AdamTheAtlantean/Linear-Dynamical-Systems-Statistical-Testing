import numpy as np
import matplotlib.pyplot as plt
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
2. T      = effective VAR sample size, i.e. T = n - p
3. d_x    = latent state dimension
4. d_y    = observation dimension
5. p      = VAR order

Theory-aligned effective memory matrix:
1. F = A - L C        ∈ R^{d_x, d_x}
2. rho(F)             controls decay of the VAR coefficients
3. Phi_i              = C F^{i-1} L
4. H_k                = C F^{k-1} L

Parallelization notes
---------------------
To preserve EXACTLY the same numerical results as the serial version,
we only parallelize over:
1. d_x values in run_dx_sweep(...)
2. repeat index r in run_dx_sweep_with_variance(...)

We do NOT parallelize the inner realization/trial loops, because those
consume one shared RNG stream in sequence.
"""


# Core experiment

def run_realization_sensitivity(
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
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
    rng = np.random.default_rng(seed)

    all_same_M, all_diff_M = [], []
    systems = []
    diff_pairs = []

    print(f"\n--- Regime: {regime_name}  rho(F) in [{rho_min}, {rho_max}] | d_x = {d_x} ---\n")

    for r in range(realizations):
        print(f"Realization {r+1}/{realizations}")

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

        if diff_regime_y3 is None:
            rho_min3, rho_max3 = rho_min, rho_max
        else:
            rho_min3, rho_max3 = diff_regime_y3

        if diff_regime_y4 is None:
            rho_min4, rho_max4 = rho_min, rho_max
        else:
            rho_min4, rho_max4 = diff_regime_y4

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

                d_same = isotropic_var_distance(pi1, pi2)
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

                d_diff = isotropic_var_distance(pi3, pi4)
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

        d_ir_pair = impulse_response_distance(A3, C3, L3, A4, C4, L4, K=K_ir, normalize=True)
        print("  IR distance of fixed DIFF pair:", d_ir_pair)

        same = np.asarray(D_same_M, dtype=float)
        diff = np.asarray(D_diff_M, dtype=float)

        print("  SAME q50/q90/q99:", np.quantile(same, [0.5, 0.9, 0.99]))
        print("  DIFF q50/q90/q99:", np.quantile(diff, [0.5, 0.9, 0.99]))
        print("  Pr(DIFF > SAME):", float(np.mean(diff > same)))
        print("  Cross Prob.:", float(np.mean(diff[:, None] > same[None, :])))
        print()

        all_same_M.append(D_same_M)
        all_diff_M.append(D_diff_M)

    ir_dists = []
    for i in range(len(systems)):
        A1, C1, L1 = systems[i]
        for j in range(i + 1, len(systems)):
            A2, C2, L2 = systems[j]
            ir_dists.append(
                impulse_response_distance(A1, C1, L1, A2, C2, L2, K=K_ir, normalize=True)
            )

    diff_ir_dists = []
    for r in range(len(diff_pairs)):
        (A3, C3, L3), (A4, C4, L4) = diff_pairs[r]
        diff_ir_dists.append(
            impulse_response_distance(A3, C3, L3, A4, C4, L4, K=K_ir, normalize=True)
        )

    return all_same_M, all_diff_M, ir_dists, diff_ir_dists, systems, diff_pairs


# ============================================================================
# Parallel-safe wrappers
# ============================================================================

def _run_single_dx_value(
    d_x,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 10,
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
    print(f"Running isotropic d_x-sweep experiment for d_x = {d_x}")
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
        "d_x": d_x,
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

    print("\nSummary for d_x =", d_x)
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


def _run_single_repeat_dx(
    r: int,
    dx_values,
    **kwargs
):
    print(f"\n=== Repeat {r+1} ===")
    return r, run_dx_sweep(
        dx_values=dx_values,
        seed=r,
        **kwargs
    )


# ============================================================================
# d_x-sweep
# ============================================================================

def run_dx_sweep(
    dx_values,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 10,
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
    Sweep over latent state dimension d_x.

    Parallel over d_x values only.

    Regimes are defined by rho(F), where F = A - L C.
    """
    if n_jobs == 1:
        sweep_results = []
        for d_x in dx_values:
            row = _run_single_dx_value(
                d_x=d_x,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                p=p,
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
            delayed(_run_single_dx_value)(
                d_x=d_x,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                p=p,
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
            for d_x in dx_values
        )

    sweep_results.sort(key=lambda row: dx_values.index(row["d_x"]))
    return sweep_results


def run_dx_sweep_with_variance(
    dx_values,
    n_repeats=100,
    n_jobs: int = 1,
    verbose_joblib: int = 0,
    **kwargs
):
    all_results = {d_x: {"auc": [], "f1": [], "best_threshold": []} for d_x in dx_values}

    if n_jobs == 1:
        repeat_outputs = []
        for r in range(n_repeats):
            print(f"\n=== Repeat {r+1}/{n_repeats} ===")
            results = run_dx_sweep(
                dx_values=dx_values,
                seed=r,
                n_jobs=1,
                **kwargs
            )
            repeat_outputs.append((r, results))
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            repeat_outputs = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
                delayed(_run_single_repeat_dx)(
                    r=r,
                    dx_values=dx_values,
                    n_jobs=1,
                    **kwargs
                )
                for r in range(n_repeats)
            )

    repeat_outputs.sort(key=lambda x: x[0])

    for _, results in repeat_outputs:
        for row in results:
            d_x = row["d_x"]
            all_results[d_x]["auc"].append(row["auc"])
            all_results[d_x]["f1"].append(row["f1"])
            all_results[d_x]["best_threshold"].append(row["best_threshold"])

    summary = []

    for d_x in dx_values:
        auc_arr = np.array(all_results[d_x]["auc"])
        f1_arr  = np.array(all_results[d_x]["f1"])
        thresh_arr = np.array(all_results[d_x]["best_threshold"])

        summary.append({
            "d_x": d_x,
            "auc_mean": np.mean(auc_arr),
            "auc_std": np.std(auc_arr),
            "f1_mean": np.mean(f1_arr),
            "f1_std": np.std(f1_arr),
            "threshold_mean": np.mean(thresh_arr),
            "threshold_std": np.std(thresh_arr)
        })

    return summary


# ============================================================================
# Plotting
# ============================================================================

def plot_dx_sweep_results(sweep_results):
    dx_vals = np.array([r["d_x"] for r in sweep_results], dtype=int)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    fpr_vals = np.array([r["fpr"] for r in sweep_results], dtype=float)
    fnr_vals = np.array([r["fnr"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(dx_vals, auc_vals, marker="o", linewidth=2)
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("AUC")
    plt.title(r"AUC vs $d_x$ (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(dx_vals, f1_vals, marker="o", linewidth=2, label="F1")
    plt.plot(dx_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(dx_vals, 1 - fpr_vals, marker="o", linewidth=2, label="TNR")
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("Score")
    plt.title(r"Classification metrics vs $d_x$ (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(dx_vals, fpr_vals, marker="o", linewidth=2, label="FPR")
    plt.plot(dx_vals, fnr_vals, marker="o", linewidth=2, label="FNR")
    plt.plot(dx_vals, tpr_vals, marker="o", linewidth=2, linestyle="--", label="TPR")
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("Rate")
    plt.title(r"Error rates vs $d_x$ (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(dx_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("Best threshold")
    plt.title(r"Best threshold vs $d_x$ (Isotropic Metric, F-band)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dx_with_variance(summary):
    dx_vals = np.array([r["d_x"] for r in summary])

    auc_mean = np.array([r["auc_mean"] for r in summary])
    auc_std  = np.array([r["auc_std"] for r in summary])

    f1_mean = np.array([r["f1_mean"] for r in summary])
    f1_std  = np.array([r["f1_std"] for r in summary])

    thresh_mean = np.array([r["threshold_mean"] for r in summary])
    thresh_std = np.array([r["threshold_std"] for r in summary])

    plt.figure(figsize=(7, 4))
    plt.errorbar(dx_vals, auc_mean, yerr=auc_std, marker="o", capsize=4)
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("AUC")
    plt.title(r"AUC vs $d_x$ (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(dx_vals, f1_mean, yerr=f1_std, marker="o", capsize=4)
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("F1")
    plt.title(r"F1 vs $d_x$ (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(dx_vals, thresh_mean, yerr=thresh_std, marker="o", capsize=4)
    plt.xlabel(r"Latent dimension $d_x$")
    plt.ylabel("Best threshold")
    plt.title(r"Best threshold vs $d_x$ (mean ± std) — Isotropic Metric, F-band")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main toggle
# ============================================================================

if __name__ == "__main__":

    USE_VARIANCE_SWEEP = True

    N_JOBS_SINGLE = 6
    N_JOBS_VARIANCE = 6

    dx_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    common_kwargs = dict(
        regime_name="short (within-regime diff, F-band)",
        rho_min=0.75,
        rho_max=0.85,
        realizations=25,
        trials=40,
        n=200,
        p=10,
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
        summary = run_dx_sweep_with_variance(
            dx_values=dx_values,
            n_repeats=100,
            n_jobs=N_JOBS_VARIANCE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 90)
        print("ISOTROPIC d_x-SWEEP SUMMARY TABLE (100 REPEATS, F-BAND)")
        print("=" * 90)
        for row in summary:
            print(
                f"d_x={row['d_x']:>2d} | "
                f"AUC mean={row['auc_mean']:.4f} | "
                f"AUC std={row['auc_std']:.4f} | "
                f"F1 mean={row['f1_mean']:.4f} | "
                f"F1 std={row['f1_std']:.4f} | "
                f"tau mean={row['threshold_mean']:.4f} | "
                f"tau std={row['threshold_std']:.4f}"
            )

        plot_dx_with_variance(summary)

    else:
        sweep_results = run_dx_sweep(
            dx_values=dx_values,
            seed=0,
            n_jobs=N_JOBS_SINGLE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 90)
        print("ISOTROPIC d_x-SWEEP SUMMARY TABLE (SINGLE RUN, F-BAND)")
        print("=" * 90)
        for row in sweep_results:
            print(
                f"d_x={row['d_x']:>2d} | "
                f"AUC={row['auc']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"TPR={row['recall']:.4f} | "
                f"FPR={row['fpr']:.4f} | "
                f"tau*={row['best_threshold']:.4f}"
            )

        plot_dx_sweep_results(sweep_results)