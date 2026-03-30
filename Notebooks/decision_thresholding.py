import numpy as np
import matplotlib.pyplot as plt

def flatten_same_diff(same_list, diff_list):
    """
    """

    same = np.concatenate([np.asarray(x, dtype=float) for x in same_list])
    diff = np.concatenate([np.asarray(x, dtype=float) for x in diff_list])

    scores = np.concatenate([same, diff])
    y_true = np.concatenate([
        np.zeros_like(same, dtype=int),
        np.ones_like(diff, dtype=int)
    ])

    return same, diff, scores, y_true


def confusion_from_threshold(scores, y_true, threshold):
    """

    """

    y_pred = (scores > threshold).astype(int)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    return TP, FP, TN, FN

def metrics_from_confusion(TP, FP, TN, FN):
    """

    """

    eps = 1e-12 # tiny number added to denom. to ensure no division by zero

    accuracy    = (TP + TN) / (TP + FP + TN + FN + eps)
    precision   = TP / (TP + FP + eps)
    recall      = TP / (TP + FN + eps) #TRP aka sensitivity
    specificity = TN / (TN + FP + eps)
    fpr         = FP / (FP + TN + eps)
    fnr         = FN / (FN + TP + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)

    return {
        "accuracy ": accuracy,
        "precision ": precision,
        "recall ": recall,
        "specificity ": specificity,
        "fpr ": fpr,
        "fnr ": fnr,
        "f1": f1,
    }


def evaluate_threshold(scores, y_true, threshold):
    """

    """
    TP, FP, TN, FN = confusion_from_threshold(scores, y_true, threshold)
    out = metrics_from_confusion(TP, FP, TN, FN,)
    out.update({
        "threshold ": float(threshold),
        "TP ": TP,
        "FP ": FP,
        "TN ": TN,
        "FN ": FN,
    })
    return out

def threshold_grid_from_scores(scores):
    """
    """
    s = np.unique(np.asarray(scores, dtype=float))
    if s.size == 0:
        raise ValueError("score is empty")
    
    eps_lo = 1e-9 * max(1.0, abs(s[0]))
    eps_hi = 1e-9 * max(1.0, abs(s[-1]))

    thresholds = np.concatenate([
        [s[0] - eps_lo],
        s,
        [s[-1] + eps_hi]
    ])
    return thresholds


def sweep_thresholds(scores, y_true):
    """
    """

    thresholds = threshold_grid_from_scores(scores)
    results = [evaluate_threshold(scores, y_true, tau) for tau in thresholds]
    return results

def roc_curve_from_results(results):
    """
    """

    fpr = np.array([r["fpr"] for r in results], dtype=float)
    tpr = np.array([r["reacall"] for r in results], dtype=float)
    thresholds = np.array([r["threshold"] for r in results], dtype=float)

    order = np.argsort(fpr)
    return fpr[order], tpr[order], thresholds[order]

def auc_trapezoid(x, y):
    """
    """
    return float(np.trapezoid(y, x))

def precision_recall_curve_from_results(results):
    """
    """

    recall = np.array([r["recall"] for r in results], dtype=float)
    precision = np.array([r["precision"] for r in results], dtype=float)
    thresholds = np.array([r["threshold"] for r in results], dtype=float)
    
    order = np.argsort(recall)
    return recall[order], precision[order], thresholds[order]

def find_best_threshold(results, criterion="f1"):
    """
    """

    if criterion == "f1":
        vals = np.array([r["f1"] for r in results], dtype=float)
    elif criterion == "accuracy":
        vals = np.array([r["accuracy"] for r in results], dtype=float)
    elif criterion == "youden":
        vals = np.array([r["recall"] - r["fpr"] for r in results], dtype=float)
    else:
        raise ValueError("criterion must be one of {'f1', 'accuracy', 'youden'}")

    idx = int(np.argmax(vals))
    best = dict(results[idx])  # copy
    best["criterion"] = criterion
    best["criterion_value"] = float(vals[idx])
    return best


def summarize_threshold_analysis(same_list, diff_list, best_by="f1", make_plots=True):
    """
    """

    same, diff, scores, y_true = flatten_same_diff(same_list, diff_list)
    results = sweep_thresholds(scores, y_true)

    fpr, tpr, roc_thresholds = roc_curve_from_results(results)
    auc = auc_trapezoid(fpr, tpr)

    best = find_best_threshold(results, criterion=best_by)



    print("\n--- Threshold Classification Summary ---")
    print(f"Pooled SAME count:      {same.size}")
    print(f"Pooled DIFFERENT count: {diff.size}")
    print(f"AUC:                    {auc:.6f}")
    print(f"Best threshold by {best_by}: {best['threshold']:.6f}")
    print(f"Accuracy:               {best['accuracy']:.6f}")
    print(f"Precision:              {best['precision']:.6f}")
    print(f"Recall / TPR:           {best['recall']:.6f}")
    print(f"Specificity:            {best['specificity']:.6f}")
    print(f"FPR:                    {best['fpr']:.6f}")
    print(f"FNR:                    {best['fnr']:.6f}")
    print(f"F1 score:               {best['f1']:.6f}")
    print(f"TP={best['TP']}  FP={best['FP']}  TN={best['TN']}  FN={best['FN']}")

    if make_plots:
        # ROC
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve: SAME vs DIFFERENT")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # F1 vs threshold
        thresholds = np.array([r["threshold"] for r in results], dtype=float)
        f1_vals = np.array([r["f1"] for r in results], dtype=float)

        plt.figure(figsize=(7, 4))
        plt.plot(thresholds, f1_vals, linewidth=2)
        plt.axvline(best["threshold"], linestyle="--", linewidth=1,
                    label=f"Best threshold = {best['threshold']:.4f}")
        plt.xlabel("Threshold")
        plt.ylabel("F1 score")
        plt.title("F1 Score vs Threshold")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Histogram with threshold
        plt.figure(figsize=(7, 4))
        plt.hist(same, bins=30, alpha=0.6, density=True, label="SAME")
        plt.hist(diff, bins=30, alpha=0.6, density=True, label="DIFFERENT")
        plt.axvline(best["threshold"], linestyle="--", linewidth=2,
                    label=f"Best threshold = {best['threshold']:.4f}")
        plt.xlabel("Mahalanobis Distance")
        plt.ylabel("Density")
        plt.title("Pooled Distance Distributions with Decision Threshold")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        "same": same,
        "diff": diff,
        "scores": scores,
        "y_true": y_true,
        "results": results,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "auc": auc,
        "best": best,
    }


