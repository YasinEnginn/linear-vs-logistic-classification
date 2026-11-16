import numpy as np


class LinearRegressionClassifier:
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_iterations: int = 1000,
        lambda_: float = 0.0,
    ):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not isinstance(num_iterations, (int, np.integer)) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        if not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("lambda_ cannot be negative.")

        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_

        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.cost_history: list[float] = []

    @staticmethod
    def _validate_data(
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")
        if X.shape[0] != y.shape[0] or 0 in X.shape:
            raise ValueError("X and y must contain the same samples.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain finite values.")
        if not np.isin(y, [0.0, 1.0]).all():
            raise ValueError("y must contain only 0 and 1.")

        return X, y

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)

        if self.weights is None:
            raise RuntimeError("The model must be fitted first.")
        if X.ndim != 2 or X.shape[1] != self.weights.shape[0]:
            raise ValueError("X has an invalid shape.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain finite values.")

        return X

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        X = self._validate_X(X)
        return np.dot(X, self.weights) + self.bias

    def compute_cost(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        X, y = self._validate_data(X, y)
        errors = self.predict_score(X) - y
        m = X.shape[0]

        data_cost = np.sum(errors**2) / (2 * m)
        reg_cost = (
            self.lambda_
            * np.sum(self.weights**2)
            / (2 * m)
        )

        return float(data_cost + reg_cost)

    def compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        X, y = self._validate_data(X, y)
        errors = self.predict_score(X) - y
        m = X.shape[0]

        dj_dw = np.dot(X.T, errors) / m
        dj_dw += (self.lambda_ / m) * self.weights
        dj_db = float(np.mean(errors))

        return dj_dw, dj_db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "LinearRegressionClassifier":
        X, y = self._validate_data(X, y)

        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0
        self.cost_history = [self.compute_cost(X, y)]

        for _ in range(self.num_iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)

            self.weights -= self.learning_rate * dj_dw
            self.bias -= self.learning_rate * dj_db

            self.cost_history.append(self.compute_cost(X, y))

        return self

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")

        return (self.predict_score(X) >= threshold).astype(np.int64)
