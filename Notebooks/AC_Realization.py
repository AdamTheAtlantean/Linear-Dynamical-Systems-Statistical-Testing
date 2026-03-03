import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from lds import simulate_lds
from var_model import build_var_xy, fit_ls, unpack_B_to_Phi
from metrics import mahalanobis_var_distance



# Helpers: sampling new realizations of A, C (and L)
def sample_stable_A(d_x: int, rho_min: float, rho_max: float, rng: np.random.Generator) -> np.ndarray:
    """Sample A and rescale so its spectral radius lies in [rho_min, rho_max]."""
    A_raw = rng.normal(size=(d_x, d_x))
    rho_raw = np.max(np.abs(np.linalg.eigvals(A_raw)))
    rho_target = rng.uniform(rho_min, rho_max)
    return (rho_target / (rho_raw + 1e-12)) * A_raw


def sample_C(d_y: int, d_x: int, rng: np.random.Generator) -> np.ndarray:
    """Sample observation matrix C ∈ R^{d_y x d_x}."""
    return rng.normal(size=(d_y, d_x))


def sample_L(d_x: int, d_y: int, rng: np.random.Generator) -> np.ndarray:
    """Sample noise injection matrix L ∈ R^{d_x x d_y}."""
    return rng.normal(size=(d_x, d_y))


# Impulse response distance across realizations
def impulse_response_matrices(A: np.ndarray, C: np.ndarray, L: np.ndarray, K: int):
    """
    Returns [H_1,...,H_K] where H_k = C A^{k-1} L.
    """
    d_x = A.shape[0]
    Ak = np.eye(d_x)  # A^0
    H = []
    for _ in range(K):
        H.append(C @ Ak @ L)   # C A^{k-1} L
        Ak = Ak @ A            # update to next power
    return H


def impulse_response_distance(A1, C1, L1, A2, C2, L2, K: int = 25, normalize: bool = True) -> float:
    """
    D_H = sum_{k=1..K} ||H_k^{(1)} - H_k^{(2)}||_F^2

    If normalize=True: divide by sum_k ||H_k^{(1)}||_F^2 to make it scale-invariant.
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
    Robustly extract y from simulate_lds return.
    simulate_lds returns (x, y, e), so y is index 1.
    """
    out = simulate_lds(n, A, C, L, rng, e_scale)
    return out[1] if isinstance(out, tuple) else out



# Fit VAR(p) and return everything needed for Mahalanobis metric
def fit_var_and_components(y: np.ndarray, p: int):
    """
    Fit VAR(p) by LS and return:
      - pi_hat     : vec(Phi_1,...,Phi_p) (Fortran order)
      - Sigma_hat  : (U^T U)/T residual covariance
      - QX_hat     : (X^T X)/T regressor covariance
    """
    X, Y = build_var_xy(y, p=p)     # X: (T, d_y*p), Y: (T, d_y)

    T = X.shape[0]
    if T <= 0:
        raise ValueError(f"Empty VAR design matrix: got T={T}. Increase n or decrease p.")

    B_hat = fit_ls(Y, X)            # B_hat: (d_y*p, d_y)
    U_hat = Y - X @ B_hat

    Sigma_hat = (U_hat.T @ U_hat) / T
    QX_hat = (X.T @ X) / T

    d_y = Y.shape[1]
    Phi_list = unpack_B_to_Phi(B_hat, p=p, d_y=d_y)  # list of p matrices (d_y, d_y)

    pi_hat = np.concatenate([Phi.flatten(order="F") for Phi in Phi_list])
    return pi_hat, Sigma_hat, QX_hat


def euclidean_distance(pi1: np.ndarray, pi2: np.ndarray) -> float:
    """Unnormalized baseline: ||pi1 - pi2||_2^2."""
    d = pi1 - pi2
    return float(d @ d)


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
    Outer loop over A,C realizations (fixed per realization).
    Inner loop over Monte Carlo trials (different noise seeds).

    Returns:
      all_same_M, all_diff_M, all_same_E, all_diff_E, ir_dists, systems

    where:
      - all_* are list-of-lists (one list per realization)
      - ir_dists is the pairwise impulse-response distance across ALL realizations
      - systems is the list of (A, C, L) per realization
    """
    rng = np.random.default_rng(seed)

    # Mahalanobis
    all_same_M, all_diff_M = [], []
    # Euclidean
    all_same_E, all_diff_E = [], []

    print(f"\n--- Regime: {regime_name}  rho in [{rho_min}, {rho_max}] ---\n")

    systems: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for r in range(realizations):
        print(f"Realization {r+1}/{realizations}")

        # Fix ONE system for this realization
        A = sample_stable_A(d_x, rho_min, rho_max, rng)
        C = sample_C(d_y, d_x, rng)
        L = sample_L(d_x, d_y, rng)

        systems.append((A, C, L))

        D_same_M, D_diff_M = [], []
        D_same_E, D_diff_E = [], []

        cond_QX_list = []

        for _ in range(trials):
            # ----- SAME LDS -----
            y1 = simulate_y_only(n, A, C, L, rng, e_scale)
            y2 = simulate_y_only(n, A, C, L, rng, e_scale)

            pi1, Sigma1, QX1 = fit_var_and_components(y1, p)
            pi2, Sigma2, QX2 = fit_var_and_components(y2, p)

            cond_QX_list.append(np.linalg.cond(QX1))
            cond_QX_list.append(np.linalg.cond(QX2))

            D_same_M.append(
                mahalanobis_var_distance(pi1, pi2, Sigma1, Sigma2, QX1, QX2)
            )
            D_same_E.append(euclidean_distance(pi1, pi2))

            # ----- DIFFERENT LDS -----
            if diff_regime is None:
                rho_min2, rho_max2 = rho_min, rho_max   # different realization, same regime
            else:
                rho_min2, rho_max2 = diff_regime        # different realization, different regime

            A2 = sample_stable_A(d_x, rho_min2, rho_max2, rng)
            C2 = sample_C(d_y, d_x, rng)
            L2 = sample_L(d_x, d_y, rng)

            y3 = simulate_y_only(n, A2, C2, L2, rng, e_scale)
            pi3, Sigma3, QX3 = fit_var_and_components(y3, p)

            D_diff_M.append(
                mahalanobis_var_distance(pi1, pi3, Sigma1, Sigma3, QX1, QX3)
            )
            D_diff_E.append(euclidean_distance(pi1, pi3))

        # Print realization summary
        print("  Mahalanobis same mean:", float(np.mean(D_same_M)))
        print("  Mahalanobis same std:", float(np.std(D_same_M, ddof=1)))
        print("  Mahalanobis diff mean:", float(np.mean(D_diff_M)))
        print("  Mahalanobis diff std:", float(np.std(D_diff_M, ddof=1)))
        #print("  Euclidean same mean:", float(np.mean(D_same_E)))
        #print("  Euclidean diff mean:", float(np.mean(D_diff_E)))
        #print("  Euclidean same std:", float(np.std(D_same_E)))
        #print("  Euclidean diff std:", float(np.std(D_diff_E)))
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
        all_same_E.append(D_same_E)
        all_diff_E.append(D_diff_E)

    # ----- Impulse response variability across realizations -----
    # Compute ONCE, after all systems exist.
    ir_dists: list[float] = []
    for i in range(len(systems)):
        A1, C1, L1 = systems[i]
        for j in range(i + 1, len(systems)):
            A2, C2, L2 = systems[j]
            ir_dists.append(
                impulse_response_distance(A1, C1, L1, A2, C2, L2, K=K_ir, normalize=True)
            )

    if len(ir_dists) == 0:
        print(f"IR response distance across realizations (K={K_ir}): (need ≥ 2 realizations)")
    else:
        ir_arr = np.asarray(ir_dists)
        std = np.std(ir_arr, ddof=1) if ir_arr.size >= 2 else 0.0
        print(f"IR response distance across realizations (K={K_ir}): "
              f"mean={np.mean(ir_arr):.4g}, std={std:.4g}, "
              f"min={np.min(ir_arr):.4g}, max={np.max(ir_arr):.4g}")

    return all_same_M, all_diff_M, all_same_E, all_diff_E, ir_dists, systems



# KDE plotting (with safety guards)
def _kde_fallback(xs: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Simple Gaussian-mixture KDE fallback (Silverman bandwidth) that works even
    when scipy's gaussian_kde can fail (e.g., near-singular variance).
    """
    dist = np.asarray(dist)
    n = dist.size
    if n == 0:
        return np.zeros_like(xs)

    if n == 1:
        # single spike: approximate with a tiny Gaussian
        h = 1e-3
    else:
        std = np.std(dist, ddof=1) + 1e-12
        h = 1.06 * std * n ** (-1 / 5)  # Silverman rule

    ys = np.mean(
        np.exp(-0.5 * ((xs[:, None] - dist[None, :]) / h) ** 2),
        axis=1
    ) / (h * np.sqrt(2 * np.pi))
    return ys


def plot_kde_overlay(distributions, title: str, gridsize: int = 400, bw_method=None, xlim=None):
    """
    Plot one KDE per realization (overlay).
    distributions: list of 1D arrays/lists (one per realization)
    """
    plt.figure(figsize=(8, 5))

    # Filter out empty dists
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
            # not enough variability for KDE — skip (or use fallback)
            continue

        try:
            ys = gaussian_kde(dist, bw_method=bw_method)(xs)
        except Exception:
            ys = _kde_fallback(xs, dist)

        plt.plot(xs, ys, alpha=0.75, label=f"Realization {i+1}")

    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Estimated density")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    #plt.legend()
    plt.show()


def plot_kde_same_vs_diff(same_dists, diff_dists, title: str, gridsize: int = 500, bw_method=None, xlim=None):
    """
    Plot pooled SAME vs pooled DIFFERENT KDE (one curve each).
    same_dists, diff_dists: list-of-lists (one list per realization)
    """
    plt.figure(figsize=(8, 5))

    same = np.concatenate([np.asarray(d) for d in same_dists if np.asarray(d).size > 0])
    diff = np.concatenate([np.asarray(d) for d in diff_dists if np.asarray(d).size > 0])

    if same.size < 2 or diff.size < 2:
        print(f"[plot_kde_same_vs_diff] Not enough data to KDE for: {title}")
        return

    all_vals = np.concatenate([same, diff])

    if xlim is None:
        lo, hi = np.quantile(all_vals, [0.001, 0.999])
        xs = np.linspace(lo, hi, gridsize)
        plt.xlim(lo, hi)
    else:
        xs = np.linspace(xlim[0], xlim[1], gridsize)
        plt.xlim(xlim[0], xlim[1])

    try:
        ys_same = gaussian_kde(same, bw_method=bw_method)(xs)
    except Exception:
        ys_same = _kde_fallback(xs, same)

    try:
        ys_diff = gaussian_kde(diff, bw_method=bw_method)(xs)
    except Exception:
        ys_diff = _kde_fallback(xs, diff)

    plt.plot(xs, ys_same, alpha=0.85, label="SAME (pooled)")
    plt.plot(xs, ys_diff, alpha=0.85, label="DIFFERENT (pooled)")

    plt.title(title)
    plt.xlabel("Distance")
    plt.ylabel("Estimated density")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    plt.legend()
    plt.show()


# Run
if __name__ == "__main__":

    # Within-regime "different" (short regime)
    same_M, diff_M, same_E, diff_E, ir_dists, systems = run_realization_sensitivity(
        regime_name="short (within-regime diff)",
        rho_min=0.75,
        rho_max=0.80,
        realizations=25,
        trials=40,
        n=1500,
        p=10,
        d_x=5,
        d_y=5,
        e_scale=0.2,
        seed=0,
        diff_regime=None,
        K_ir=25,
    )

    # --- impulse-response variability plot ---
    if len(ir_dists) > 0:
        plt.figure()
        plt.hist(ir_dists, bins=15)
        plt.title("Pairwise impulse-response distances across realizations")
        plt.xlabel("D_H")
        plt.ylabel("count")
        plt.show()

    # KDE overlays across realizations
    plot_kde_overlay(same_M, "Mahalanobis SAME KDE across realizations (short)", bw_method="silverman")
    plot_kde_overlay(diff_M, "Mahalanobis DIFFERENT KDE across realizations (short, within-regime)", bw_method="silverman")

    # Pooled SAME vs DIFFERENT KDE (most readable)
    plot_kde_same_vs_diff(same_M, diff_M, "Mahalanobis KDE: SAME vs DIFFERENT (short, pooled)", bw_method="silverman")

    # Also visualize Euclidean baseline
    plot_kde_same_vs_diff(same_E, diff_E, "Euclidean KDE: SAME vs DIFFERENT (short, pooled)", bw_method="silverman")

    # DIFFERENT regime example (short vs long):
    # same_M2, diff_M2, _, _, ir2, _ = run_realization_sensitivity(
    #     regime_name="short vs long",
    #     rho_min=0.75,
    #     rho_max=0.80,
    #     realizations=4,
    #     trials=40,
    #     n=1500,
    #     p=10,
    #     d_x=5,
    #     d_y=5,
    #     e_scale=0.2,
    #     seed=0,
    #     diff_regime=(0.95, 0.98),
    #     K_ir=25,
    # )
    # plot_kde_same_vs_diff(same_M2, diff_M2, "Mahalanobis KDE: SAME vs DIFFERENT (short vs long, pooled)", bw_method="silverman")


    