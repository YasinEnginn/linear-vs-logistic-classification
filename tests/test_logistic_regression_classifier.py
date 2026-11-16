import numpy as np

from src.logistic_regression_classifier import (
    LogisticRegressionClassifier,
)


def test_sigmoid():
    """The sigmoid function should return known probabilities."""

    logits = np.array([
        -1000.0,
        -2.0,
        0.0,
        2.0,
        1000.0,
    ])
    expected = np.array([
        0.0,
        0.11920292202211755,
        0.5,
        0.8807970779778823,
        1.0,
    ])

    probabilities = LogisticRegressionClassifier.sigmoid(logits)

    assert np.allclose(probabilities, expected)


def test_logistic_training():
    """Gradient descent should reduce the training cost."""

    X = np.array([
        [-2.0],
        [-1.0],
        [1.0],
        [2.0],
    ])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionClassifier(
        learning_rate=0.1,
        num_iterations=500,
    )
    model.fit(X, y)

    assert model.cost_history[-1] < model.cost_history[0]
    assert np.isfinite(model.cost_history).all()
    assert np.array_equal(model.predict(X), y)


def test_probability_range():
    """Every predicted probability should remain between zero and one."""

    X = np.array([
        [-1.0, 0.5],
        [0.0, 0.0],
        [1.0, -0.5],
    ])
    y = np.array([0, 0, 1])

    model = LogisticRegressionClassifier(
        learning_rate=0.1,
        num_iterations=200,
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == y.shape
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()


def test_logistic_gradient():
    """Analytic gradients should agree with finite differences."""

    X = np.array([
        [0.2, -0.4],
        [1.0, 0.5],
        [-0.3, 0.8],
    ])
    y = np.array([0.0, 1.0, 1.0])

    model = LogisticRegressionClassifier(lambda_=0.1)
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


def test_extreme_sigmoid_values():
    """Extreme logits should not produce NaN or infinite values."""

    logits = np.array([
        -1_000_000.0,
        1_000_000.0,
    ])

    probabilities = LogisticRegressionClassifier.sigmoid(logits)

    assert np.isfinite(probabilities).all()
    assert np.array_equal(probabilities, np.array([0.0, 1.0]))
