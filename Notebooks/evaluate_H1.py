import numpy as np


def flatten_diff_distances(diff_list):
    """
    Pool DIFFERENT-LDS distance samples across realizations into one 1D array.

    Parameters
    ----------
    diff_list : list of array-like
        diff_list[r] contains distance values from trials where the two
        time series were generated from different LDSs.

    Returns
    -------
    diff : np.ndarray
        Flattened array of pooled DIFFERENT-LDS distances.
    """
    diff = np.concatenate([np.asarray(x, dtype=float) for x in diff_list])

    if diff.size == 0:
        raise ValueError("diff_list contains no distance samples.")

    return diff


def evaluate_H1(diff_list, tau):
    """
    Evaluate the fixed-threshold decision rule under H1.

    Hypotheses
    ----------
    H0: LDS^(1) = LDS^(2)
    H1: LDS^(1) != LDS^(2)

    Decision rule
    -------------
    Reject H0 if distance > tau.
    Otherwise fail to reject H0.

    Since this function uses DIFFERENT-LDS samples only, H1 is true for all samples.
    Therefore:
      - distance > tau: true positive
      - distance <= tau: false negative

    Parameters
    ----------
    diff_list : list of array-like
        DIFFERENT-LDS distance samples grouped by realization.
    tau : float
        Fixed decision threshold.

    Returns
    -------
    summary : dict
        Dictionary containing counts and rates under H1.
    """
    diff = flatten_diff_distances(diff_list)

    TP = int(np.sum(diff > tau))
    FN = int(np.sum(diff <= tau))

    eps = 1e-12
    tpr = TP / (diff.size + eps)
    fnr = FN / (diff.size + eps)

    return {
        "tau": float(tau),
        "n_diff": int(diff.size),
        "TP": TP,
        "FN": FN,
        "tpr": float(tpr),
        "fnr": float(fnr),
        "diff": diff,
    }


def print_H1_summary(summary):
    """
    H1 evaluation summary.
    """
    print("\n--- H1 Evaluation (DIFFERENT LDS) ---")
    print("H1: LDS^(1) != LDS^(2)")
    print(f"tau:      {summary['tau']:.6f}")
    print(f"n_diff:   {summary['n_diff']}")
    print(f"TP:       {summary['TP']}")
    print(f"FN:       {summary['FN']}")
    print(f"TPR:      {summary['tpr']:.6f}")
    print(f"FNR:      {summary['fnr']:.6f}")


if __name__ == "__main__":
    from realizations_new import diff_M

    tau = 163.0

    summary = evaluate_H1(diff_M, tau)
    print_H1_summary(summary)