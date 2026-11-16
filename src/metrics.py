import numpy as np
import pandas as pd


def _validate_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D.")
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1.")

    return y_true, y_pred


def confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    y_true, y_pred = _validate_labels(y_true, y_pred)

    return {
        "TP": int(np.sum((y_true == 1) & (y_pred == 1))),
        "TN": int(np.sum((y_true == 0) & (y_pred == 0))),
        "FP": int(np.sum((y_true == 0) & (y_pred == 1))),
        "FN": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    counts = confusion_counts(y_true, y_pred)
    return np.array([
        [counts["TN"], counts["FP"]],
        [counts["FN"], counts["TP"]],
    ])


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tp, tn = counts["TP"], counts["TN"]
    fp, fn = counts["FP"], counts["FN"]

    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1_score = _safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
    }


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    if y_true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and scores must be 1D.")
    if y_true.size == 0 or y_true.shape != scores.shape:
        raise ValueError("y_true and scores must have the same length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain finite values.")

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty 1D array.")
    if not np.isfinite(thresholds).all():
        raise ValueError("thresholds must contain finite values.")

    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(np.int64)
        rows.append({
            "threshold": float(threshold),
            **confusion_counts(y_true, y_pred),
            **classification_metrics(y_true, y_pred),
        })

    return pd.DataFrame(rows)
