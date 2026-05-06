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
1. e_scale values in run_e_scale_sweep(...)
2. repeat index r in run_e_scale_sweep_with_variance(...)

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

    print(
        f"\n--- Regime: {regime_name}  rho(F) in [{rho_min}, {rho_max}] | "
        f"e_scale = {e_scale} ---\n"
    )

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

def _run_single_e_scale_value(
    e_scale,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 10,
    d_x: int = 5,
    d_y: int = 5,
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
    print(f"Running isotropic e_scale-sweep experiment for e_scale = {e_scale}")
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

    same_flat = np.concatenate([np.asarray(x, dtype=float) for x in same_M]) if len(same_M) > 0 else np.array([])
    diff_flat = np.concatenate([np.asarray(x, dtype=float) for x in diff_M]) if len(diff_M) > 0 else np.array([])

    same_flat = same_flat[np.isfinite(same_flat)]
    diff_flat = diff_flat[np.isfinite(diff_flat)]

    row = {
        "e_scale": float(e_scale),
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
        "mean_diff_ir": float(np.mean(diff_ir_dists)) if len(diff_ir_dists) > 0 else np.nan,
        "mean_same_distance": float(np.mean(same_flat)) if same_flat.size > 0 else np.nan,
        "median_same_distance": float(np.median(same_flat)) if same_flat.size > 0 else np.nan,
        "mean_diff_distance": float(np.mean(diff_flat)) if diff_flat.size > 0 else np.nan,
        "median_diff_distance": float(np.median(diff_flat)) if diff_flat.size > 0 else np.nan,
    }

    print(f"\nSummary for e_scale = {e_scale}")
    print(f"  mean SAME distance  = {row['mean_same_distance']:.6f}")
    print(f"  mean DIFF distance  = {row['mean_diff_distance']:.6f}")
    print(f"  mean DIFF IR        = {row['mean_diff_ir']:.6f}")
    print(f"  AUC                 = {row['auc']:.6f}")
    print(f"  Best threshold      = {row['best_threshold']:.6f}")
    print(f"  Accuracy            = {row['accuracy']:.6f}")
    print(f"  Precision           = {row['precision']:.6f}")
    print(f"  Recall / TPR        = {row['recall']:.6f}")
    print(f"  Specificity         = {row['specificity']:.6f}")
    print(f"  FPR                 = {row['fpr']:.6f}")
    print(f"  FNR                 = {row['fnr']:.6f}")
    print(f"  F1                  = {row['f1']:.6f}")

    return row


def _run_single_repeat_e_scale(
    r: int,
    e_scale_values,
    **kwargs
):
    print(f"\n=== Repeat {r+1} ===")
    return r, run_e_scale_sweep(
        e_scale_values=e_scale_values,
        seed=r,
        **kwargs
    )


# ============================================================================
# e_scale sweep
# ============================================================================

def run_e_scale_sweep(
    e_scale_values,
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 25,
    trials: int = 40,
    n: int = 1500,
    p: int = 10,
    d_x: int = 5,
    d_y: int = 5,
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
    Sweep over observation-noise scale e_scale.

    Parallel over e_scale values only.

    Regimes are defined by rho(F), where F = A - L C.
    """
    if n_jobs == 1:
        sweep_results = []
        for e_scale in e_scale_values:
            row = _run_single_e_scale_value(
                e_scale=e_scale,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                p=p,
                d_x=d_x,
                d_y=d_y,
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
            delayed(_run_single_e_scale_value)(
                e_scale=e_scale,
                regime_name=regime_name,
                rho_min=rho_min,
                rho_max=rho_max,
                realizations=realizations,
                trials=trials,
                n=n,
                p=p,
                d_x=d_x,
                d_y=d_y,
                seed=seed,
                diff_regime_y3=diff_regime_y3,
                diff_regime_y4=diff_regime_y4,
                K_ir=K_ir,
                best_by=best_by,
                a_rho_max=a_rho_max,
                c_scale=c_scale,
                l_scale=l_scale,
            )
            for e_scale in e_scale_values
        )

    sweep_results.sort(key=lambda row: e_scale_values.index(row["e_scale"]))
    return sweep_results


def run_e_scale_sweep_with_variance(
    e_scale_values,
    n_repeats=100,
    n_jobs: int = 1,
    verbose_joblib: int = 0,
    **kwargs
):
    all_results = {
        float(e_scale): {
            "auc": [],
            "f1": [],
            "best_threshold": [],
            "recall": [],
            "fnr": [],
            "mean_same_distance": [],
            "mean_diff_distance": [],
            "mean_diff_ir": [],
        }
        for e_scale in e_scale_values
    }

    if n_jobs == 1:
        repeat_outputs = []
        for r in range(n_repeats):
            print(f"\n=== Repeat {r+1}/{n_repeats} ===")
            results = run_e_scale_sweep(
                e_scale_values=e_scale_values,
                seed=r,
                n_jobs=1,
                **kwargs
            )
            repeat_outputs.append((r, results))
    else:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            repeat_outputs = Parallel(n_jobs=n_jobs, verbose=verbose_joblib)(
                delayed(_run_single_repeat_e_scale)(
                    r=r,
                    e_scale_values=e_scale_values,
                    n_jobs=1,
                    **kwargs
                )
                for r in range(n_repeats)
            )

    repeat_outputs.sort(key=lambda x: x[0])

    for _, results in repeat_outputs:
        for row in results:
            e_scale = float(row["e_scale"])
            all_results[e_scale]["auc"].append(row["auc"])
            all_results[e_scale]["f1"].append(row["f1"])
            all_results[e_scale]["best_threshold"].append(row["best_threshold"])
            all_results[e_scale]["recall"].append(row["recall"])
            all_results[e_scale]["fnr"].append(row["fnr"])
            all_results[e_scale]["mean_same_distance"].append(row["mean_same_distance"])
            all_results[e_scale]["mean_diff_distance"].append(row["mean_diff_distance"])
            all_results[e_scale]["mean_diff_ir"].append(row["mean_diff_ir"])

    summary = []

    for e_scale in e_scale_values:
        e_scale = float(e_scale)

        auc_arr = np.array(all_results[e_scale]["auc"])
        f1_arr = np.array(all_results[e_scale]["f1"])
        thresh_arr = np.array(all_results[e_scale]["best_threshold"])
        recall_arr = np.array(all_results[e_scale]["recall"])
        fnr_arr = np.array(all_results[e_scale]["fnr"])
        same_dist_arr = np.array(all_results[e_scale]["mean_same_distance"])
        diff_dist_arr = np.array(all_results[e_scale]["mean_diff_distance"])
        diff_ir_arr = np.array(all_results[e_scale]["mean_diff_ir"])

        summary.append({
            "e_scale": e_scale,
            "auc_mean": np.mean(auc_arr),
            "auc_std": np.std(auc_arr),
            "f1_mean": np.mean(f1_arr),
            "f1_std": np.std(f1_arr),
            "threshold_mean": np.mean(thresh_arr),
            "threshold_std": np.std(thresh_arr),
            "recall_mean": np.mean(recall_arr),
            "recall_std": np.std(recall_arr),
            "fnr_mean": np.mean(fnr_arr),
            "fnr_std": np.std(fnr_arr),
            "mean_same_distance_mean": np.mean(same_dist_arr),
            "mean_same_distance_std": np.std(same_dist_arr),
            "mean_diff_distance_mean": np.mean(diff_dist_arr),
            "mean_diff_distance_std": np.std(diff_dist_arr),
            "mean_diff_ir_mean": np.mean(diff_ir_arr),
            "mean_diff_ir_std": np.std(diff_ir_arr),
        })

    return summary


# ============================================================================
# Plotting (V2 style)
# ============================================================================

def plot_e_scale_sweep_results_v2(sweep_results, show_auc=True):
    e_vals = np.array([r["e_scale"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    fnr_vals = np.array([r["fnr"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)
    same_mean_vals = np.array([r["mean_same_distance"] for r in sweep_results], dtype=float)
    diff_mean_vals = np.array([r["mean_diff_distance"] for r in sweep_results], dtype=float)
    diff_ir_vals = np.array([r["mean_diff_ir"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(e_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(e_vals, fnr_vals, marker="o", linewidth=2, label="FNR")
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Rate")
    plt.title(r"H$_1$ detection rates vs observation noise")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(e_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs observation noise")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(e_vals, same_mean_vals, marker="o", linewidth=2, linestyle="--", label=r"Mean H$_0$ distance")
    plt.plot(e_vals, diff_mean_vals, marker="o", linewidth=2, label=r"Mean H$_1$ distance")
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Mean isotropic distance")
    plt.title(r"Mean H$_0$/H$_1$ distances vs observation noise")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(e_vals, diff_ir_vals, marker="o", linewidth=2)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel(r"Mean H$_1$ IR distance")
    plt.title(r"Mean H$_1$ IR distance vs observation noise")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(e_vals, f1_vals, marker="o", linewidth=2)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("F1")
    plt.title("F1 vs observation noise")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if show_auc:
        plt.figure(figsize=(7, 4))
        plt.plot(e_vals, auc_vals, marker="o", linewidth=2)
        plt.xlabel("Noise scale e_scale")
        plt.ylabel("AUC")
        plt.title("AUC vs observation noise")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_e_scale_with_variance_v2(summary, show_auc=True):
    e_vals = np.array([r["e_scale"] for r in summary], dtype=float)

    recall_mean = np.array([r["recall_mean"] for r in summary], dtype=float)
    recall_std = np.array([r["recall_std"] for r in summary], dtype=float)

    fnr_mean = np.array([r["fnr_mean"] for r in summary], dtype=float)
    fnr_std = np.array([r["fnr_std"] for r in summary], dtype=float)

    thresh_mean = np.array([r["threshold_mean"] for r in summary], dtype=float)
    thresh_std = np.array([r["threshold_std"] for r in summary], dtype=float)

    same_dist_mean = np.array([r["mean_same_distance_mean"] for r in summary], dtype=float)
    same_dist_std = np.array([r["mean_same_distance_std"] for r in summary], dtype=float)

    diff_dist_mean = np.array([r["mean_diff_distance_mean"] for r in summary], dtype=float)
    diff_dist_std = np.array([r["mean_diff_distance_std"] for r in summary], dtype=float)

    diff_ir_mean = np.array([r["mean_diff_ir_mean"] for r in summary], dtype=float)
    diff_ir_std = np.array([r["mean_diff_ir_std"] for r in summary], dtype=float)

    f1_mean = np.array([r["f1_mean"] for r in summary], dtype=float)
    f1_std = np.array([r["f1_std"] for r in summary], dtype=float)

    auc_mean = np.array([r["auc_mean"] for r in summary], dtype=float)
    auc_std = np.array([r["auc_std"] for r in summary], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.errorbar(e_vals, recall_mean, yerr=recall_std, marker="o", capsize=4, label="TPR")
    plt.errorbar(e_vals, fnr_mean, yerr=fnr_std, marker="o", capsize=4, label="FNR")
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Rate")
    plt.title(r"H$_1$ detection rates vs observation noise (mean ± std)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(e_vals, thresh_mean, yerr=thresh_std, marker="o", capsize=4)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs observation noise (mean ± std)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(e_vals, same_dist_mean, yerr=same_dist_std, marker="o", capsize=4, linestyle="--", label=r"Mean H$_0$ distance")
    plt.errorbar(e_vals, diff_dist_mean, yerr=diff_dist_std, marker="o", capsize=4, label=r"Mean H$_1$ distance")
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("Mean isotropic distance")
    plt.title(r"Mean H$_0$ /H$_1$ distances vs observation noise (mean ± std)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(e_vals, diff_ir_mean, yerr=diff_ir_std, marker="o", capsize=4)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel(r"Mean H$_1$ IR distance")
    plt.title(r"Mean H$_1$ IR distance vs observation noise (mean ± std)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.errorbar(e_vals, f1_mean, yerr=f1_std, marker="o", capsize=4)
    plt.xlabel("Noise scale e_scale")
    plt.ylabel("F1")
    plt.title("F1 vs observation noise (mean ± std)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if show_auc:
        plt.figure(figsize=(7, 4))
        plt.errorbar(e_vals, auc_mean, yerr=auc_std, marker="o", capsize=4)
        plt.xlabel("Noise scale e_scale")
        plt.ylabel("AUC")
        plt.title("AUC vs observation noise (mean ± std)")
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

    e_scale_values = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

    common_kwargs = dict(
        regime_name="short (within-regime diff, F-band)",
        rho_min=0.75,
        rho_max=0.85,
        realizations=25,
        trials=40,
        n=200,
        p=10,
        d_x=5,
        d_y=5,
        diff_regime_y3=(0.75, 0.85),
        diff_regime_y4=(0.80, 0.90),
        K_ir=25,
        best_by="f1",
        a_rho_max=0.9995,
        c_scale=1.0,
        l_scale=1.0,
    )

    if USE_VARIANCE_SWEEP:
        summary = run_e_scale_sweep_with_variance(
            e_scale_values=e_scale_values,
            n_repeats=100,
            n_jobs=N_JOBS_VARIANCE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 110)
        print("ISOTROPIC E_SCALE-SWEEP V2 SUMMARY TABLE (100 REPEATS, F-BAND)")
        print("=" * 110)
        for row in summary:
            print(
                f"e_scale={row['e_scale']:.2f} | "
                f"TPR mean={row['recall_mean']:.4f} | "
                f"FNR mean={row['fnr_mean']:.4f} | "
                f"F1 mean={row['f1_mean']:.4f} | "
                f"tau mean={row['threshold_mean']:.4f} | "
                f"SAME dist mean={row['mean_same_distance_mean']:.4f} | "
                f"DIFF dist mean={row['mean_diff_distance_mean']:.4f}"
            )

        plot_e_scale_with_variance_v2(summary, show_auc=True)

    else:
        sweep_results = run_e_scale_sweep(
            e_scale_values=e_scale_values,
            seed=0,
            n_jobs=N_JOBS_SINGLE,
            verbose_joblib=10,
            **common_kwargs
        )

        print("\n" + "=" * 110)
        print("ISOTROPIC E_SCALE-SWEEP V2 SUMMARY TABLE (SINGLE RUN, F-BAND)")
        print("=" * 110)
        for row in sweep_results:
            print(
                f"e_scale={row['e_scale']:.2f} | "
                f"TPR={row['recall']:.4f} | "
                f"FNR={row['fnr']:.4f} | "
                f"F1={row['f1']:.4f} | "
                f"tau*={row['best_threshold']:.4f} | "
                f"mean SAME dist={row['mean_same_distance']:.4f} | "
                f"mean DIFF dist={row['mean_diff_distance']:.4f}"
            )

        plot_e_scale_sweep_results_v2(sweep_results, show_auc=True)