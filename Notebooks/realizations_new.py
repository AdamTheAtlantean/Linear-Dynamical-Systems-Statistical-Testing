import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import mahalanobis_var_distance



# Helpers: sampling new realizations of A, C, and L
def sample_stable_A(
    d_x: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample A and rescale so its spectral radius lies in [rho_min, rho_max].
    """
    A_raw = rng.normal(size=(d_x, d_x))
    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)
    return (rho_target / (rho_raw + 1e-12)) * A_raw


def sample_stable_A_identity_centered(
    d_x: int,
    rho_min: float,
    rho_max: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample A around a scaled identity, then rescale so its spectral radius
    lies in [rho_min, rho_max].
    """
    A_raw = 10 * np.eye(d_x) + rng.normal(size=(d_x, d_x))
    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)
    return (rho_target / (rho_raw + 1e-12)) * A_raw


def sample_C(d_y: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    """Sample observation matrix C ∈ R^{d_y x d_x}."""
    return rng.normal(size=(d_y, d_x))


def sample_L(d_x: int, d_y: int, rng: np.random.Generator) -> np.ndarray:
    """Sample noise injection matrix L ∈ R^{d_x x d_y}."""
    return rng.normal(size=(d_x, d_y))



# Impulse-response distance across realizations
def impulse_response_matrices(A: np.ndarray, C: np.ndarray, L: np.ndarray, K: int):
    """
    Returns [H_1, ..., H_K] where H_k = C A^{k-1} L.
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
    return out[1] if isinstance(out, tuple) else out


# Fit VAR(p) and return objects needed for Mahalanobis metric
def fit_var_and_components(y: np.ndarray, p: int):
    """
    Fit VAR(p) by LS and return:
      - pi_hat     : vec(Phi_1, ..., Phi_p) in Fortran order
      - Sigma_hat  : residual covariance = (U^T U)/T
      - QX_hat     : regressor covariance = (X^T X)/T
    """
    X, Y = build_var_xy(y, p=p)

    T = X.shape[0]
    if T <= 0:
        raise ValueError(f"Empty VAR design matrix: got T={T}. Increase n or decrease p.")

    B_hat = fit_ls(Y, X)
    U_hat = Y - X @ B_hat

    Sigma_hat = (U_hat.T @ U_hat) / T
    QX_hat = (X.T @ X) / T

    d_y = Y.shape[1]
    Phi_list = unpack_B_to_Phi(B_hat, p=p, d_y=d_y)

    pi_hat = np.concatenate([Phi.flatten(order="F") for Phi in Phi_list])
    return pi_hat, Sigma_hat, QX_hat



# KDE utilities
def _kde_fallback(xs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Simple Gaussian-mixture KDE fallback for near-singular cases.
    """
    dist = np.asarray(dist)
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


def plot_kde_per_realization(same_list, diff_list):
    """
    For each realization, plot SAME KDE vs DIFFERENT KDE.
    """
    n_realizations = len(same_list)

    for i in range(n_realizations):
        same_vals = np.asarray(same_list[i])
        diff_vals = np.asarray(diff_list[i])

        if same_vals.size < 2 or diff_vals.size < 2:
            continue

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
        plt.plot(x, ys_same, label="Same KDE", linewidth=2)
        plt.plot(x, ys_diff, label="Different KDE", linewidth=2)

        plt.title(f"Probability Distribution Comparison: Realization {i+1}")
        plt.xlabel("Mahalanobis Distance")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_kde_overlay(distributions, title: str, gridsize: int = 400, bw_method=None, xlim=None):
    """
    Plot one KDE per realization on the same graph.
    """
    plt.figure(figsize=(8, 5))

    dists = [np.asarray(d) for d in distributions if np.asarray(d).size > 0]
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
        dist = np.asarray(dist)
        if dist.size < 2 or np.std(dist) == 0:
            continue

        try:
            ys = gaussian_kde(dist, bw_method=bw_method)(xs)
        except Exception:
            ys = _kde_fallback(xs, dist)

        plt.plot(xs, ys, alpha=0.75, label=f"Realization {i+1}")

    plt.title(title)
    plt.xlabel("Mahalanobis Distance")
    plt.ylabel("Estimated Density")
    plt.grid(alpha=0.3)
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
    Compare one realization against all others via IR distance.
    """
    R = len(systems)
    if not (0 <= idx < R):
        raise ValueError(f"idx must be in [0, {R-1}]")

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

    print(f"\nIR profile for Realization {idx+1} (K={K_ir}, normalize={normalize})")
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
        print(f"    -> Realization {j+1}: D_H = {dists[order[k]]:.6g}")

    return dists, summary



# Main experiment: realization sensitivity
def run_realization_sensitivity(
    regime_name: str,
    rho_min: float,
    rho_max: float,
    realizations: int = 5,
    trials: int = 40,
    n: int = 1500,
    p: int = 10,
    d_x: int = 5,
    d_y: int = 5,
    e_scale: float = 0.2,
    seed: int = 0,
    diff_regime: tuple[float, float] | None = None,
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

    print(f"\n--- Regime: {regime_name}  rho in [{rho_min}, {rho_max}] ---\n")

    systems: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for r in range(realizations):
        print(f"Realization {r+1}/{realizations}")

        # Fixed realization for SAME comparisons in this outer loop
        A = sample_stable_A(d_x, rho_min, rho_max, rng)
        C = sample_C(d_y, d_x, rng)
        L = sample_L(d_x, d_y, rng)

        systems.append((A, C, L))

        D_same_M, D_diff_M = [], []
        cond_QX_list = []

        for _ in range(trials):
            
            # SAME LDS: y1 vs y2 from the same fixed system
            y1 = simulate_y_only(n, A, C, L, rng, e_scale)
            y2 = simulate_y_only(n, A, C, L, rng, e_scale)

            pi1, Sigma1, QX1 = fit_var_and_components(y1, p)
            pi2, Sigma2, QX2 = fit_var_and_components(y2, p)

            cond_QX_list.append(np.linalg.cond(QX1))
            cond_QX_list.append(np.linalg.cond(QX2))

            d_same = mahalanobis_var_distance(pi1, pi2, Sigma1, Sigma2, QX1, QX2)
            D_same_M.append(d_same)

            
            # DIFFERENT LDS: y3 vs y4 from two independent systems
           
            # System for y3
            A3 = sample_stable_A_identity_centered(d_x, rho_min, rho_max, rng)
            C3 = sample_C(d_y, d_x, rng)
            L3 = sample_L(d_x, d_y, rng)

            # System for y4
            if diff_regime is None:
                rho_min4, rho_max4 = rho_min, rho_max
            else:
                rho_min4, rho_max4 = diff_regime

            A4 = sample_stable_A_identity_centered(d_x, rho_min4, rho_max4, rng)
            C4 = sample_C(d_y, d_x, rng)
            L4 = sample_L(d_x, d_y, rng)

            y3 = simulate_y_only(n, A3, C3, L3, rng, e_scale)
            y4 = simulate_y_only(n, A4, C4, L4, rng, e_scale)

            pi3, Sigma3, QX3 = fit_var_and_components(y3, p)
            pi4, Sigma4, QX4 = fit_var_and_components(y4, p)

            cond_QX_list.append(np.linalg.cond(QX3))
            cond_QX_list.append(np.linalg.cond(QX4))

            d_diff = mahalanobis_var_distance(pi3, pi4, Sigma3, Sigma4, QX3, QX4)
            D_diff_M.append(d_diff)

        # Realization summary
        print("  Mahalanobis same mean:", float(np.mean(D_same_M)))
        print("  Mahalanobis same std:", float(np.std(D_same_M, ddof=1)))
        print("  Mahalanobis diff mean:", float(np.mean(D_diff_M)))
        print("  Mahalanobis diff std:", float(np.std(D_diff_M, ddof=1)))

        cond_arr = np.array(cond_QX_list)
        print("  QX condition number (median):", np.median(cond_arr))
        print("  QX condition number (max):", np.max(cond_arr))

        same = np.asarray(D_same_M)
        diff = np.asarray(D_diff_M)

        print("  SAME q50/q90/q99:", np.quantile(same, [0.5, 0.9, 0.99]))
        print("  DIFF q50/q90/q99:", np.quantile(diff, [0.5, 0.9, 0.99]))
        print("  Pr(DIFF > SAME):", float(np.mean(diff > same)))
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

    return all_same_M, all_diff_M, ir_dists, systems



# Run


if __name__ == "__main__":

    same_M, diff_M, ir_dists, systems = run_realization_sensitivity(
        regime_name="short (within-regime diff)",
        rho_min=0.75,
        rho_max=0.80,
        realizations=25,
        trials=100,
        n=1500,
        p=10,
        d_x=5,
        d_y=5,
        e_scale=0.2,
        seed=0,
        diff_regime=None,
        K_ir=25,
    )

    # IR profile per realization
    for r in range(1, len(systems) + 1):
        ir_profile_for_realization(
            systems,
            idx=r - 1,
            K_ir=25,
            normalize=True,
            top_k=5,
        )

    # Histogram of pairwise IR distances
    if len(ir_dists) > 0:
        plt.figure()
        plt.hist(ir_dists, bins=15)
        plt.title("Pairwise impulse-response distances across realizations")
        plt.xlabel("D_H")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.show()

    # KDE overlays across realizations
    plot_kde_overlay(
        same_M,
        "Mahalanobis SAME KDE across realizations (short)",
        bw_method="silverman"
    )

    plot_kde_overlay(
        diff_M,
        "Mahalanobis DIFFERENT KDE across realizations (short)",
        bw_method="silverman"
    )

    # Per-realization SAME vs DIFFERENT KDE
    plot_kde_per_realization(same_M, diff_M)