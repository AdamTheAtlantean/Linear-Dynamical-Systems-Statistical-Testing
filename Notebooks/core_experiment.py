import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from metrics import isotropic_var_distance
from evaluate_H0 import evaluate_H0, print_H0_summary
from evaluate_H1 import evaluate_H1, print_H1_summary
from threshold import summarize_threshold_analysis
from linalg_utils import spectral_radius, effective_F
from sampling_utils import sample_system_with_F_in_band
from lds import simulate_y_only
from var_model import fit_var_and_components

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
5. pi_hat     ∈ R^{(p * d_y^2),}

Theory-aligned effective memory matrix:
1. F = A - L C     ∈ R^{d_x, d_x}
2. rho(F)          controls decay of the VAR coefficients
3. Phi_i           = C F^{i-1} L
4. H_k             = C F^{k-1} L

Important numerical note:
- We sample F first so that rho(F) lies in the requested band.
- We then construct A = F + L C.
- We reject only bad embeddings (C, L) for a fixed F before resampling F.
- We also reject systems whose rho(A) is too large, so simulate_lds
  remains numerically stable.
"""


# Impulse-response helpers (theory-aligned with F = A - L C)

def impulse_response_matrices(
    A: np.ndarray,
    C: np.ndarray,
    L: np.ndarray,
    K: int,
) -> list[np.ndarray]:
    """
    Returns [H_1, ..., H_K] where H_k = C F^{k-1} L and F = A - L C.
    """
    F = effective_F(A, C, L)
    d_x = A.shape[0]

    Fk = np.eye(d_x)
    H = []

    for _ in range(K):
        H.append(C @ Fk @ L)
        Fk = Fk @ F

    return H


def impulse_response_distance(
    A1: np.ndarray,
    C1: np.ndarray,
    L1: np.ndarray,
    A2: np.ndarray,
    C2: np.ndarray,
    L2: np.ndarray,
    K: int = 25,
    normalize: bool = True,
) -> float:
    """
    D_H = sum_{k=1..K} ||H_k^{(1)} - H_k^{(2)}||_F^2
    where H_k = C (A - L C)^{k-1} L.

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



# Plotting helpers

def compute_global_xlim(same_list, diff_list, lower_q=0.001, upper_q=0.999):
    """
    Compute one shared x-axis range across all SAME and DIFFERENT realizations.
    """
    all_vals = []

    for d in list(same_list) + list(diff_list):
        arr = np.asarray(d, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            all_vals.append(arr)

    if len(all_vals) == 0:
        return (0.0, 1.0)

    all_vals = np.concatenate(all_vals)

    if all_vals.size == 1:
        x = float(all_vals[0])
        return (max(0.0, x - 1.0), x + 1.0)

    x_min, x_max = np.quantile(all_vals, [lower_q, upper_q])

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        if x_min == x_max:
            x_min = max(0.0, x_min - 1.0)
            x_max = x_max + 1.0

    return (float(x_min), float(x_max))


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


def plot_kde_per_realization(same_list, diff_list, tau=None, xlim=None):
    """
    For each realization, plot SAME KDE vs DIFFERENT KDE.
    """
    n_realizations = len(same_list)

    for i in range(n_realizations):
        same_vals = np.asarray(same_list[i], dtype=float)
        diff_vals = np.asarray(diff_list[i], dtype=float)

        same_vals = same_vals[np.isfinite(same_vals)]
        diff_vals = diff_vals[np.isfinite(diff_vals)]

        if same_vals.size < 2 or diff_vals.size < 2:
            continue

        if xlim is not None:
            xmin, xmax = xlim
        else:
            xmin = min(same_vals.min(), diff_vals.min())
            xmax = max(same_vals.max(), diff_vals.max())

        x = np.linspace(xmin, xmax, 500)

        try:
            kde_same = gaussian_kde(same_vals)
            ys_same = kde_same(x)
        except Exception:
            ys_same = _kde_fallback(x, same_vals)

        try:
            kde_diff = gaussian_kde(diff_vals)
            ys_diff = kde_diff(x)
        except Exception:
            ys_diff = _kde_fallback(x, diff_vals)

        plt.figure(figsize=(6, 4))
        plt.plot(x, ys_same, label="H$_0$", linewidth=2)
        plt.plot(x, ys_diff, label="H$_1$", linewidth=2)

        plt.title(f"Probability Distribution Comparison: Realization {i + 1}")
        plt.xlabel("Isotropic Distance")
        plt.ylabel("Probability Density")
        plt.xlim(xmin, xmax)
        if tau is not None:
            plt.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


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

    if xlim is not None:
        xs = np.linspace(xlim[0], xlim[1], gridsize)
        plt.xlim(xlim[0], xlim[1])
    else:
        lo, hi = np.quantile(all_vals, [0.001, 0.999])
        xs = np.linspace(lo, hi, gridsize)
        plt.xlim(lo, hi)

    for i, dist in enumerate(distributions):
        dist = np.asarray(dist, dtype=float)
        dist = dist[np.isfinite(dist)]
        if dist.size < 2 or np.std(dist) == 0:
            continue

        try:
            ys = gaussian_kde(dist, bw_method=bw_method)(xs)
        except Exception:
            ys = _kde_fallback(xs, dist)

        plt.plot(xs, ys, alpha=0.75, label=f"Realization {i + 1}")

    plt.title(title)
    plt.xlabel("Isotropic Distance")
    plt.ylabel("Estimated Density")
    plt.grid(alpha=0.3)
    if tau is not None:
        plt.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$")
    plt.tight_layout()
    plt.show()


def plot_histograms_shared_xlim(same_list, diff_list, bins=30, xlim=None, tau=None):
    """
    Shared-range histogram for pooled SAME and DIFFERENT distances.
    """
    same = np.concatenate([np.asarray(d, dtype=float) for d in same_list]) if len(same_list) > 0 else np.array([])
    diff = np.concatenate([np.asarray(d, dtype=float) for d in diff_list]) if len(diff_list) > 0 else np.array([])

    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]

    if same.size == 0 or diff.size == 0:
        print("[plot_histograms_shared_xlim] Not enough finite pooled values to plot.")
        return

    if xlim is None:
        xlim = compute_global_xlim(same_list, diff_list)

    plt.figure(figsize=(7, 4))
    plt.hist(same, bins=bins, range=xlim, alpha=0.6, density=True, label=r"H$_0$")
    plt.hist(diff, bins=bins, range=xlim, alpha=0.6, density=True, label=r"H$_1$")
    if tau is not None:
        plt.axvline(tau, color="red", linestyle="--", linewidth=1.5, label=r"$\tau$")
    plt.xlim(*xlim)
    plt.xlabel("Isotropic Distance")
    plt.ylabel("Density")
    plt.title(r"Pooled H$_0$ vs H$_1$ Histograms")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



# IR profile for one realization

def ir_profile_for_realization(
    systems,
    idx: int,
    K_ir: int = 25,
    normalize: bool = True,
    top_k: int = 5,
):
    """
    Compare one realization against all others via theory-aligned IR distance.
    """
    R = len(systems)
    if not (0 <= idx < R):
        raise ValueError(f"idx must be in [0, {R - 1}]")

    A1, C1, L1 = systems[idx]

    dists = []
    js = []

    for j in range(R):
        if j == idx:
            continue
        A2, C2, L2 = systems[j]
        d = impulse_response_distance(A1, C1, L1, A2, C2, L2, K=K_ir, normalize=normalize)
        dists.append(d)
        js.append(j)

    dists = np.asarray(dists, dtype=float)
    js = np.asarray(js, dtype=int)
    order = np.argsort(dists)

    summary = {
        "realization": idx + 1,
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "nearest_neighbor": int(js[order[0]] + 1),
        "nearest_neighbor_dist": float(dists[order[0]]),
    }

    print(f"\nIR profile for Realization {idx + 1} (K={K_ir}, normalize={normalize})")
    print(
        f"  mean={summary['mean']:.6g}, median={summary['median']:.6g}, "
        f"min={summary['min']:.6g}, max={summary['max']:.6g}"
    )
    print(
        f"  nearest neighbor: Realization {summary['nearest_neighbor']} "
        f"with D_H={summary['nearest_neighbor_dist']:.6g}"
    )

    print(f"  {top_k} closest realizations:")
    for k in range(min(top_k, len(order))):
        j = js[order[k]]
        print(f"    -> Realization {j + 1}: D_H = {dists[order[k]]:.6g}")

    return dists, summary



# Main experiment

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
    c_scale: float = 0.15,
    l_scale: float = 0.15,
):
    """
    Outer loop over fixed realizations.
    Inner loop over Monte Carlo trials.

    SAME condition:
        y1 vs y2 from the same fixed realization (A, C, L)

    DIFFERENT condition:
        y3 vs y4 from two independently sampled LDS realizations

    IMPORTANT:
    The regime band [rho_min, rho_max] refers to rho(F), where F = A - L C.
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
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ]
    ] = []

    for r in range(realizations):
        print(f"Realization {r + 1}/{realizations}")

        A, C, L = sample_system_with_F_in_band(
            d_x, d_y, rho_min, rho_max, rng,
            a_rho_max=a_rho_max, c_scale=c_scale, l_scale=l_scale
        )

        if diff_regime_y3 is None:
            rho_min3, rho_max3 = rho_min, rho_max
        else:
            rho_min3, rho_max3 = diff_regime_y3

        A3, C3, L3 = sample_system_with_F_in_band(
            d_x, d_y, rho_min3, rho_max3, rng,
            a_rho_max=a_rho_max, c_scale=c_scale, l_scale=l_scale
        )

        if diff_regime_y4 is None:
            rho_min4, rho_max4 = rho_min, rho_max
        else:
            rho_min4, rho_max4 = diff_regime_y4

        A4, C4, L4 = sample_system_with_F_in_band(
            d_x, d_y, rho_min4, rho_max4, rng,
            a_rho_max=a_rho_max, c_scale=c_scale, l_scale=l_scale
        )

        systems.append((A, C, L))
        diff_pairs.append(((A3, C3, L3), (A4, C4, L4)))

        rho_F_same = spectral_radius(effective_F(A, C, L))
        rho_F_3 = spectral_radius(effective_F(A3, C3, L3))
        rho_F_4 = spectral_radius(effective_F(A4, C4, L4))
        rho_A_same = spectral_radius(A)
        rho_A_3 = spectral_radius(A3)
        rho_A_4 = spectral_radius(A4)

        print(f"  target rho(F) band SAME = [{rho_min}, {rho_max}]")
        print(f"  target rho(F) band y3   = [{rho_min3}, {rho_max3}]")
        print(f"  target rho(F) band y4   = [{rho_min4}, {rho_max4}]")
        print(f"  rho(F_same) = {rho_F_same:.6f}, rho(A_same) = {rho_A_same:.6f}")
        print(f"  rho(F_y3)   = {rho_F_3:.6f}, rho(A_y3)   = {rho_A_3:.6f}")
        print(f"  rho(F_y4)   = {rho_F_4:.6f}, rho(A_y4)   = {rho_A_4:.6f}")

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
        print("  Cross Prob.:", float(np.mean(diff[:, None] > same[None, :])))
        print()

        all_same_M.append(D_same_M)
        all_diff_M.append(D_diff_M)

    # Pairwise IR distances across SAME outer-loop realizations
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
    a_rho_max: float = 0.98,
    c_scale: float = 0.15,
    l_scale: float = 0.15,
):
    """
    Sweep over VAR orders p and record pooled classification performance.
    The regime is defined by rho(F), where F = A - L C.
    """
    sweep_results = []

    for p in p_values:
        print("\n" + "=" * 70)
        print(f"Running p-sweep experiment for p = {p}")
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

        sweep_results.append(row)

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

    return sweep_results


def plot_p_sweep_results(sweep_results):
    """
    Plot key classification metrics as functions of VAR order p.
    """
    p_vals = np.array([r["p"] for r in sweep_results], dtype=int)
    auc_vals = np.array([r["auc"] for r in sweep_results], dtype=float)
    f1_vals = np.array([r["f1"] for r in sweep_results], dtype=float)
    fpr_vals = np.array([r["fpr"] for r in sweep_results], dtype=float)
    tpr_vals = np.array([r["recall"] for r in sweep_results], dtype=float)
    thresh_vals = np.array([r["best_threshold"] for r in sweep_results], dtype=float)

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, auc_vals, marker="o", linewidth=2)
    plt.xlabel("VAR order p")
    plt.ylabel("AUC")
    plt.title("AUC vs VAR order p")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, f1_vals, marker="o", linewidth=2, label="F1")
    plt.plot(p_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.plot(p_vals, 1 - fpr_vals, marker="o", linewidth=2, label="TNR")
    plt.xlabel("VAR order p")
    plt.ylabel("Score")
    plt.title("Classification metrics vs VAR order p")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, fpr_vals, marker="o", linewidth=2, label="FPR")
    plt.plot(p_vals, tpr_vals, marker="o", linewidth=2, label="TPR")
    plt.xlabel("VAR order p")
    plt.ylabel("Rate")
    plt.title("FPR and TPR vs VAR order p")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.plot(p_vals, thresh_vals, marker="o", linewidth=2)
    plt.xlabel("VAR order p")
    plt.ylabel("Best threshold")
    plt.title("Best threshold vs VAR order p")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Run

if __name__ == "__main__":

    same_M, diff_M, ir_dists, diff_ir_dists, systems, diff_pairs = run_realization_sensitivity(
        regime_name="F-band experiment",
        rho_min=0.75,
        rho_max=0.85,
        realizations=25,
        trials=40,
        n=200,
        p=25,
        d_x=5,
        d_y=5,
        e_scale=0.2,
        seed=0,
        diff_regime_y3=(0.30, 0.40),
        diff_regime_y4=(0.85, 0.95),
        K_ir=25,
        a_rho_max=0.98,
        c_scale=1,
        l_scale=1,
    )

    # Shared x-limits across all KDE plots
    shared_xlim = compute_global_xlim(same_M, diff_M, lower_q=0.001, upper_q=0.999)
    print("\nShared xlim used for plots:", shared_xlim)

    plot_kde_overlay(
        same_M,
        r"H$_0$ Probability Distributions across realizations",
        bw_method="silverman",
        xlim=shared_xlim,
        tau=0.911169,
    )

    plot_kde_overlay(
        diff_M,
        r"H$_1$ Probability Distributions across realizations",
        bw_method="silverman",
        xlim=shared_xlim,
        tau=0.911169,
    )

    plot_kde_per_realization(
        same_M,
        diff_M,
        tau=0.911169,
        xlim=shared_xlim,
    )

    plot_histograms_shared_xlim(
        same_M,
        diff_M,
        bins=30,
        xlim=shared_xlim,
        tau=0.911169,
    )

    # Keep summarize_threshold_analysis non-plotting to avoid NaN/hist-range issues
    # in external plotting code. We use our own shared-range plots above instead.
    summary = summarize_threshold_analysis(
        same_M,
        diff_M,
        best_by="f1",
        make_plots=False,
    )

    print("\nThreshold summary:")
    print(f"  AUC            = {summary['auc']:.6f}")
    print(f"  Best threshold = {summary['best']['threshold']:.6f}")
    print(f"  Accuracy       = {summary['best']['accuracy']:.6f}")
    print(f"  Precision      = {summary['best']['precision']:.6f}")
    print(f"  Recall / TPR   = {summary['best']['recall']:.6f}")
    print(f"  Specificity    = {summary['best']['specificity']:.6f}")
    print(f"  FPR            = {summary['best']['fpr']:.6f}")
    print(f"  FNR            = {summary['best']['fnr']:.6f}")
    print(f"  F1             = {summary['best']['f1']:.6f}")

    tau = 0.911169

    summary_H0 = evaluate_H0(same_M, tau)
    print_H0_summary(summary_H0)

    summary_H1 = evaluate_H1(diff_M, tau)
    print_H1_summary(summary_H1)