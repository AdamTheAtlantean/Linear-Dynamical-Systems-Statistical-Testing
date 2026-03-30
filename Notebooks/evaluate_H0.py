import numpy as np


def flatten_same_distances(same_list):
    """
    Pool SAME-LDS distance samples across realizations into one 1D array.

    Parameters
    ----------
    same_list : list of array-like
        same_list[r] contains distance values from trials where both
        time series were generated from the same LDS in realization r.

    Returns
    -------
    same : np.ndarray
        Flattened array of pooled SAME-LDS distances.
    """
    same = np.concatenate([np.asarray(x, dtype=float) for x in same_list])

    if same.size == 0:
        raise ValueError("same_list contains no distance samples.")

    return same


def evaluate_H0(same_list, tau):
    """
    Evaluate the fixed-threshold decision rule under H0.

    Hypotheses
    ----------
    H0: LDS^(1) = LDS^(2)
    H1: LDS^(1) != LDS^(2)

    Decision rule
    -------------
    Reject H0 if distance > tau.
    Otherwise fail to reject H0.

    Since this function uses SAME-LDS samples only, H0 is true for all samples.
    Therefore:
      - distance > tau: false positive
      - distance <= tau: true negative

    Parameters
    ----------
    same_list : list of array-like
        SAME-LDS distance samples grouped by realization.
    tau : float
        Fixed decision threshold.

    Returns
    -------
    summary : dict
        Dictionary containing counts and rates under H0.
    """
    same = flatten_same_distances(same_list)

    FP = int(np.sum(same > tau))
    TN = int(np.sum(same <= tau))

    eps = 1e-12
    fpr = FP / (same.size + eps)
    tnr = TN / (same.size + eps)

    return {
        "tau": float(tau),
        "n_same": int(same.size),
        "FP": FP,
        "TN": TN,
        "fpr": float(fpr),
        "tnr": float(tnr),
        "same": same,
    }


def print_H0_summary(summary):
    """
    H0 evaluation summary.
    """
    print("\n--- H0 Evaluation (SAME LDS) ---")
    print("H0: LDS^(1) = LDS^(2)")
    print(f"tau:      {summary['tau']:.6f}")
    print(f"n_same:   {summary['n_same']}")
    print(f"FP:       {summary['FP']}")
    print(f"TN:       {summary['TN']}")
    print(f"FPR:      {summary['fpr']:.6f}")
    print(f"TNR:      {summary['tnr']:.6f}")
