import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "05_final_model_comparison"
    / "final_model_comparison.py"
)

module_spec = importlib.util.spec_from_file_location(
    "final_model_comparison",
    EXPERIMENT_PATH,
)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("Could not load the final comparison module.")

final_comparison = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(final_comparison)


def test_polynomial_feature_values():
    """Degree-two output should contain originals, squares, and interactions."""

    X = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    transformed = final_comparison.create_polynomial_features(X)
    expected = np.array([
        [2.0, 3.0, 4.0, 9.0, 6.0],
        [4.0, 5.0, 16.0, 25.0, 20.0],
    ])

    assert np.array_equal(transformed, expected)


def test_prepared_split_shapes_and_balance():
    """The fixed split should be stratified and use the expected feature counts."""

    feature_sets, y_train, y_validation, y_test = (
        final_comparison.prepare_datasets()
    )

    X_train_a, X_validation_a, X_test_a = feature_sets[
        "A: Scaled original features"
    ]
    X_train_b, X_validation_b, X_test_b = feature_sets[
        "B: Degree-two features"
    ]

    assert X_train_a.shape == (540, 7)
    assert X_validation_a.shape == (180, 7)
    assert X_test_a.shape == (180, 7)

    assert X_train_b.shape == (540, 35)
    assert X_validation_b.shape == (180, 35)
    assert X_test_b.shape == (180, 35)

    assert np.bincount(y_train).tolist() == [270, 270]
    assert np.bincount(y_validation).tolist() == [90, 90]
    assert np.bincount(y_test).tolist() == [90, 90]

    assert np.allclose(X_train_a.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(X_train_b.mean(axis=0), 0.0, atol=1e-12)
    assert np.isfinite(X_validation_b).all()
    assert np.isfinite(X_test_b).all()


def test_validation_candidates_cover_every_final_experiment():
    """Each model-feature pair should have a small candidate set."""

    experiment_counts = {}
    for candidate in final_comparison.VALIDATION_CANDIDATES:
        key = (candidate["feature_set"], candidate["model_name"])
        experiment_counts[key] = experiment_counts.get(key, 0) + 1

    assert experiment_counts == {
        ("A: Scaled original features", "Linear"): 4,
        ("A: Scaled original features", "Logistic"): 4,
        ("B: Degree-two features", "Linear"): 4,
        ("B: Degree-two features", "Logistic"): 4,
    }
