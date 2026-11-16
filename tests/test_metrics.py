import numpy as np
import pytest

from src.metrics import (
    classification_metrics,
    confusion_counts,
    confusion_matrix,
    threshold_metrics,
)


def test_confusion_counts():
    """Confusion-matrix counts should match a known example."""

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    confusion_components = confusion_counts(
        y_true,
        y_pred,
    )
    matrix = confusion_matrix(y_true, y_pred)

    assert confusion_components == {
        "TP": 2,
        "TN": 2,
        "FP": 1,
        "FN": 1,
    }
    assert np.array_equal(
        matrix,
        np.array([
            [2, 1],
            [1, 2],
        ]),
    )


def test_classification_metrics():
    """All metrics should match their manually calculated values."""

    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    metrics = classification_metrics(y_true, y_pred)

    expected_value = 2.0 / 3.0

    assert np.isclose(metrics["accuracy"], expected_value)
    assert np.isclose(metrics["precision"], expected_value)
    assert np.isclose(metrics["recall"], expected_value)
    assert np.isclose(metrics["specificity"], expected_value)
    assert np.isclose(metrics["f1_score"], expected_value)


def test_zero_denominator():
    """Undefined positive-class metrics should safely return zero."""

    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])

    metrics = classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["specificity"] == 1.0
    assert metrics["f1_score"] == 0.0


def test_threshold_metrics():
    """Threshold rows should reflect the expected precision-recall changes."""

    actual_labels = np.array([0, 0, 1, 1])
    decision_scores = np.array([0.1, 0.4, 0.6, 0.9])
    threshold_grid = np.array([0.3, 0.5, 0.7])

    performance_table = threshold_metrics(
        actual_labels,
        decision_scores,
        threshold_grid,
    )

    assert np.allclose(
        performance_table["precision"],
        np.array([2.0 / 3.0, 1.0, 1.0]),
    )
    assert np.allclose(
        performance_table["recall"],
        np.array([1.0, 1.0, 0.5]),
    )
    assert np.array_equal(
        performance_table["TP"],
        np.array([2, 2, 1]),
    )


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (
            np.array([0, 1]),
            np.array([0]),
            "same length",
        ),
        (
            np.array([0, 2]),
            np.array([0, 1]),
            "y_true",
        ),
        (
            np.array([0, 1]),
            np.array([0, -1]),
            "y_pred",
        ),
    ],
)
def test_invalid_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    message: str,
):
    """Invalid labels and unequal lengths should be rejected."""

    with pytest.raises(ValueError, match=message):
        confusion_counts(y_true, y_pred)
