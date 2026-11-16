"""Create sigmoid, contour, and 3D logistic-cost visualizations.

The demonstration uses standardized Area as a single input and the encoded
Raisin class as the target. It visualizes the binary cross-entropy objective
used by LogisticRegressionClassifier.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raisin" / "Raisin_Dataset.xlsx"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "figures" / "optimization"

TARGET_ENCODING = {"Kecimen": 0, "Besni": 1}
CLASS_COLORS = {"Kecimen": "#0072B2", "Besni": "#E69F00"}
RANDOM_SEED = 42


def load_training_data():
    """Return standardized Area, binary target, and class names."""

    raisin_df = pd.read_excel(DATA_PATH)
    training_df, _ = train_test_split(
        raisin_df,
        test_size=0.40,
        random_state=RANDOM_SEED,
        stratify=raisin_df["Class"],
    )

    area_raw = training_df["Area"].to_numpy(dtype=np.float64)
    area_mean = area_raw.mean()
    area_std = area_raw.std()
    area_scaled = (area_raw - area_mean) / area_std

    target = (
        training_df["Class"]
        .map(TARGET_ENCODING)
        .to_numpy(dtype=np.float64)
    )
    class_names = training_df["Class"].to_numpy()

    return area_scaled, target, class_names, area_mean, area_std


def sigmoid(z):
    """Calculate sigmoid probabilities without numerical overflow."""

    z = np.asarray(z, dtype=np.float64)
    probabilities = np.empty_like(z)

    positive_mask = z >= 0
    probabilities[positive_mask] = 1.0 / (
        1.0 + np.exp(-z[positive_mask])
    )

    exp_z = np.exp(z[~positive_mask])
    probabilities[~positive_mask] = exp_z / (1.0 + exp_z)

    return probabilities


def compute_cost(feature, target, weight, bias):
    """Calculate binary cross-entropy J(w, b)."""

    logits = feature * weight + bias
    probabilities = np.clip(sigmoid(logits), 1e-15, 1 - 1e-15)

    return -np.mean(
        target * np.log(probabilities)
        + (1 - target) * np.log(1 - probabilities)
    )


def run_gradient_descent(
    feature,
    target,
    learning_rate=0.20,
    number_of_iterations=200,
):
    """Train one-feature logistic regression and retain its path."""

    weight = 0.0
    bias = 0.0
    number_of_samples = feature.shape[0]

    weight_history = [weight]
    bias_history = [bias]
    cost_history = [compute_cost(feature, target, weight, bias)]

    for _ in range(number_of_iterations):
        probabilities = sigmoid(feature * weight + bias)
        errors = probabilities - target

        weight_gradient = np.dot(feature, errors) / number_of_samples
        bias_gradient = np.sum(errors) / number_of_samples

        weight = weight - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        weight_history.append(weight)
        bias_history.append(bias)
        cost_history.append(compute_cost(feature, target, weight, bias))

    return (
        np.asarray(weight_history),
        np.asarray(bias_history),
        np.asarray(cost_history),
    )


def calculate_cost_surface(feature, target, weight_history, bias_history):
    """Evaluate logistic cost on a grid of w and b values."""

    weight_span = max(weight_history.max() - weight_history.min(), 1.0)
    bias_span = max(bias_history.max() - bias_history.min(), 1.0)

    weight_values = np.linspace(
        weight_history.min() - 0.35 * weight_span,
        weight_history.max() + 0.35 * weight_span,
        220,
    )
    bias_values = np.linspace(
        bias_history.min() - 0.85 * bias_span,
        bias_history.max() + 0.85 * bias_span,
        220,
    )
    weight_grid, bias_grid = np.meshgrid(weight_values, bias_values)
    cost_grid = np.empty_like(weight_grid)

    for row_index, bias in enumerate(bias_values):
        logits = (
            feature[:, np.newaxis] * weight_values[np.newaxis, :]
            + bias
        )
        probabilities = np.clip(sigmoid(logits), 1e-15, 1 - 1e-15)
        sample_costs = -(
            target[:, np.newaxis] * np.log(probabilities)
            + (1 - target[:, np.newaxis]) * np.log(1 - probabilities)
        )
        cost_grid[row_index] = sample_costs.mean(axis=0)

    return weight_grid, bias_grid, cost_grid


def draw_sigmoid_graph(
    axis,
    feature,
    target,
    class_names,
    final_weight,
    final_bias,
):
    """Draw observations, fitted probability curve, and threshold."""

    for class_name in ["Kecimen", "Besni"]:
        class_mask = class_names == class_name
        axis.scatter(
            feature[class_mask],
            target[class_mask],
            label=class_name,
            color=CLASS_COLORS[class_name],
            s=27,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.35,
        )

    curve_x = np.linspace(feature.min(), feature.max(), 400)
    curve_y = sigmoid(curve_x * final_weight + final_bias)

    axis.plot(
        curve_x,
        curve_y,
        color="#222222",
        linewidth=2.6,
        label="Predicted probability",
    )
    axis.axhline(
        0.5,
        color="#D55E00",
        linestyle="--",
        linewidth=1.5,
        label="Decision threshold = 0.5",
    )

    if final_weight != 0:
        decision_location = -final_bias / final_weight
        axis.axvline(
            decision_location,
            color="#D55E00",
            linestyle=":",
            linewidth=1.3,
        )

    axis.set_title("A. One-feature logistic regression", loc="left")
    axis.set_xlabel("Standardized Area")
    axis.set_ylabel(r"Probability $P(y=1\mid x)$")
    axis.set_ylim(-0.10, 1.10)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
    axis.legend(frameon=False, fontsize=8.5)


def draw_contour_graph(
    axis,
    weight_grid,
    bias_grid,
    cost_grid,
    weight_history,
    bias_history,
):
    """Draw logistic-cost contours and the gradient descent path."""

    minimum_cost = cost_grid.min()
    cost_range = cost_grid.max() - minimum_cost
    contour_levels = minimum_cost + np.geomspace(
        max(cost_range * 0.001, 1e-7),
        cost_range * 0.92,
        12,
    )

    contours = axis.contour(
        weight_grid,
        bias_grid,
        cost_grid,
        levels=contour_levels,
        cmap="Blues",
        linewidths=1.35,
    )
    axis.clabel(contours, inline=True, fontsize=7, fmt="%.3f")

    axis.plot(
        weight_history,
        bias_history,
        color="#D55E00",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
        markevery=12,
        label="Gradient descent path",
        zorder=5,
    )
    axis.scatter(
        weight_history[0],
        bias_history[0],
        color="#222222",
        s=42,
        label="Start",
        zorder=6,
    )
    axis.scatter(
        weight_history[-1],
        bias_history[-1],
        marker="*",
        color="#009E73",
        edgecolor="white",
        linewidth=0.7,
        s=170,
        label="Minimum",
        zorder=7,
    )

    axis.set_title("B. Contour plot of logistic cost", loc="left")
    axis.set_xlabel(r"Weight ($w$)")
    axis.set_ylabel(r"Bias ($b$)")
    axis.grid(False)
    axis.legend(frameon=False, fontsize=8.5)


def draw_3d_cost_surface(
    axis,
    weight_grid,
    bias_grid,
    cost_grid,
    weight_history,
    bias_history,
    cost_history,
):
    """Draw the 3D logistic-cost surface and optimization path."""

    axis.plot_surface(
        weight_grid,
        bias_grid,
        cost_grid,
        cmap="Blues",
        alpha=0.82,
        linewidth=0,
        antialiased=True,
    )
    axis.plot(
        weight_history,
        bias_history,
        cost_history,
        color="#D55E00",
        linewidth=2.7,
        label="Gradient descent",
        zorder=8,
    )
    axis.scatter(
        weight_history[-1],
        bias_history[-1],
        cost_history[-1],
        marker="*",
        color="#009E73",
        edgecolor="white",
        linewidth=0.7,
        s=130,
        label="Minimum",
        zorder=9,
    )

    axis.set_title("C. Three-dimensional logistic-cost surface", loc="left")
    axis.set_xlabel(r"Weight ($w$)", labelpad=8)
    axis.set_ylabel(r"Bias ($b$)", labelpad=8)
    axis.set_zlabel(r"Logistic cost $J(w,b)$", labelpad=8)
    axis.view_init(elev=28, azim=-125)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")


def create_figure():
    """Create, save, and report the logistic-regression visualization."""

    feature, target, class_names, area_mean, area_std = load_training_data()
    weight_history, bias_history, cost_history = run_gradient_descent(
        feature,
        target,
    )
    weight_grid, bias_grid, cost_grid = calculate_cost_surface(
        feature,
        target,
        weight_history,
        bias_history,
    )

    figure = plt.figure(figsize=(18, 6.2), constrained_layout=True)
    grid = figure.add_gridspec(1, 3)

    sigmoid_axis = figure.add_subplot(grid[0, 0])
    contour_axis = figure.add_subplot(grid[0, 1])
    surface_axis = figure.add_subplot(grid[0, 2], projection="3d")

    draw_sigmoid_graph(
        sigmoid_axis,
        feature,
        target,
        class_names,
        weight_history[-1],
        bias_history[-1],
    )
    draw_contour_graph(
        contour_axis,
        weight_grid,
        bias_grid,
        cost_grid,
        weight_history,
        bias_history,
    )
    draw_3d_cost_surface(
        surface_axis,
        weight_grid,
        bias_grid,
        cost_grid,
        weight_history,
        bias_history,
        cost_history,
    )

    figure.suptitle(
        "Logistic Regression Cost Geometry and Gradient Descent",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.025,
        (
            "Training split only (n = 540). Area is standardized using training "
            "statistics. Kecimen = 0 and Besni = 1."
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / "logistic_regression_cost_visualization.png"
    pdf_path = OUTPUT_DIRECTORY / "logistic_regression_cost_visualization.pdf"

    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Area mean: {area_mean:.6f}")
    print(f"Area standard deviation: {area_std:.6f}")
    print(f"Final weight: {weight_history[-1]:.6f}")
    print(f"Final bias: {bias_history[-1]:.6f}")
    print(f"Initial cost: {cost_history[0]:.6f}")
    print(f"Final cost: {cost_history[-1]:.6f}")


if __name__ == "__main__":
    create_figure()
