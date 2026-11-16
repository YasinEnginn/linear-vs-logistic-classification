import numpy as np
import pytest

from src.linear_regression_classifier import (
    LinearRegressionClassifier,
)


def test_linear_training():
    """The model should learn a simple binary pattern."""

    X = np.array([
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ])

    y = np.array([0, 0, 1, 1])

    model = LinearRegressionClassifier(
        learning_rate=0.1,
        num_iterations=500,
        lambda_=0.0,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == y.shape

    initial_cost = model.cost_history[0]
    final_cost = model.cost_history[-1]

    assert initial_cost > final_cost

    assert np.array_equal(predictions, y)
    assert np.isfinite(model.cost_history).all()


def test_linear_predict_before_fit():
    """Prediction should not be allowed before training."""

    model = LinearRegressionClassifier()

    X = np.array([
        [1.0],
        [2.0],
    ])

    with pytest.raises(
        RuntimeError,
        match="fitted",
    ):
        model.predict(X)


def test_linear_gradient():
    """Analytic gradients should agree with finite differences."""

    X = np.array([
        [0.2, -0.4],
        [1.0, 0.5],
        [-0.3, 0.8],
    ])
    y = np.array([0.0, 1.0, 1.0])

    model = LinearRegressionClassifier(lambda_=0.1)
    model.weights = np.array([0.3, -0.2])
    model.bias = 0.1

    analytic_weight_gradient, analytic_bias_gradient = (
        model.compute_gradient(X, y)
    )

    epsilon = 1e-6
    numerical_weight_gradient = np.zeros_like(model.weights)

    for index in range(model.weights.size):
        original_weight = model.weights[index]

        model.weights[index] = original_weight + epsilon
        cost_plus = model.compute_cost(X, y)

        model.weights[index] = original_weight - epsilon
        cost_minus = model.compute_cost(X, y)

        model.weights[index] = original_weight
        numerical_weight_gradient[index] = (
            cost_plus - cost_minus
        ) / (2.0 * epsilon)

    original_bias = model.bias

    model.bias = original_bias + epsilon
    cost_plus = model.compute_cost(X, y)

    model.bias = original_bias - epsilon
    cost_minus = model.compute_cost(X, y)

    model.bias = original_bias
    numerical_bias_gradient = (
        cost_plus - cost_minus
    ) / (2.0 * epsilon)

    assert np.allclose(
        analytic_weight_gradient,
        numerical_weight_gradient,
        atol=1e-7,
    )
    assert np.isclose(
        analytic_bias_gradient,
        numerical_bias_gradient,
        atol=1e-7,
    )
