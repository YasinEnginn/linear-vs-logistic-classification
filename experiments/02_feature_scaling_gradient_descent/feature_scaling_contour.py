"""Visualize how feature scaling changes gradient descent geometry.

The figure uses two Raisin Dataset features and the same mean-squared-error
objective as the linear regression classifier. Only the training split is used.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raisin" / "Raisin_Dataset.xlsx"
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "figures" / "optimization"

FEATURE_NAMES = ["Area", "Eccentricity"]
TARGET_ENCODING = {"Kecimen": 0, "Besni": 1}
RANDOM_SEED = 42
TEST_SIZE = 0.40

CLASS_COLORS = {
    "Kecimen": "#0072B2",
    "Besni": "#E69F00",
}


def load_training_data():
    """Load and split the data, then return raw and standardized features."""

    raisin_df = pd.read_excel(DATA_PATH)
    training_df, _ = train_test_split(
        raisin_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=raisin_df["Class"],
    )

    X_raw = training_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = (
        training_df["Class"]
        .map(TARGET_ENCODING)
        .to_numpy(dtype=np.float64)
    )

    training_means = X_raw.mean(axis=0)
    training_stds = X_raw.std(axis=0)

    X_centered = X_raw - training_means
    X_scaled = X_centered / training_stds
    y_centered = y - y.mean()

    return (
        training_df,
        X_raw,
        X_centered,
        X_scaled,
        y_centered,
        training_means,
        training_stds,
    )


def calculate_quadratic_terms(X, y):
    """Return the quadratic MSE terms for J(w)."""

    number_of_samples = X.shape[0]
    hessian = np.dot(X.T, X) / number_of_samples
    linear_term = np.dot(X.T, y) / number_of_samples
    target_term = np.dot(y, y) / number_of_samples

    return hessian, linear_term, target_term


def trace_gradient_descent(
    X,
    y,
    number_of_iterations=25,
    step_fraction=1.5,
):
    """Run gradient descent and retain every two-weight position."""

    hessian, linear_term, _ = calculate_quadratic_terms(X, y)
    eigenvalues = np.linalg.eigvalsh(hessian)
    learning_rate = step_fraction / eigenvalues.max()

    weights = np.zeros(X.shape[1], dtype=np.float64)
    weight_history = [weights.copy()]

    for _ in range(number_of_iterations):
        gradient = np.dot(hessian, weights) - linear_term
        weights = weights - learning_rate * gradient
        weight_history.append(weights.copy())

    optimum_weights = np.linalg.solve(hessian, linear_term)
    condition_number = eigenvalues.max() / eigenvalues.min()

    return (
        np.asarray(weight_history),
        optimum_weights,
        learning_rate,
        condition_number,
    )


def calculate_plot_limits(weight_history, optimum_weights):
    """Create readable limits containing the path and the optimum."""

    important_points = np.vstack([weight_history, optimum_weights])
    lower_limits = important_points.min(axis=0)
    upper_limits = important_points.max(axis=0)
    spans = upper_limits - lower_limits
    spans = np.where(spans == 0, 1.0, spans)

    return lower_limits - 0.20 * spans, upper_limits + 0.20 * spans


def calculate_cost_surface(X, y, weight_history, optimum_weights):
    """Evaluate the MSE cost on a grid of w1 and w2 values."""

    lower_limits, upper_limits = calculate_plot_limits(
        weight_history,
        optimum_weights,
    )

    weight_1_values = np.linspace(lower_limits[0], upper_limits[0], 260)
    weight_2_values = np.linspace(lower_limits[1], upper_limits[1], 260)
    weight_1_grid, weight_2_grid = np.meshgrid(
        weight_1_values,
        weight_2_values,
    )

    hessian, linear_term, target_term = calculate_quadratic_terms(X, y)

    cost_grid = 0.5 * (
        hessian[0, 0] * weight_1_grid**2
        + 2 * hessian[0, 1] * weight_1_grid * weight_2_grid
        + hessian[1, 1] * weight_2_grid**2
        - 2 * linear_term[0] * weight_1_grid
        - 2 * linear_term[1] * weight_2_grid
        + target_term
    )

    return weight_1_grid, weight_2_grid, cost_grid


def draw_feature_scatter(
    axis,
    training_df,
    X_values,
    title,
    x_label,
    y_label,
):
    """Draw a class-colored two-feature scatter plot."""

    for class_name in ["Kecimen", "Besni"]:
        class_mask = training_df["Class"].to_numpy() == class_name
        axis.scatter(
            X_values[class_mask, 0],
            X_values[class_mask, 1],
            label=class_name,
            color=CLASS_COLORS[class_name],
            s=26,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.35,
        )

    axis.set_title(title, loc="left")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(title="Raisin class", frameon=False)
    axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)


def draw_descent_path(axis, weight_history):
    """Draw the early gradient descent steps with directional arrows."""

    visible_steps = min(12, len(weight_history))
    visible_path = weight_history[:visible_steps]

    axis.plot(
        visible_path[:, 0],
        visible_path[:, 1],
        color="#D55E00",
        linewidth=1.6,
        marker="o",
        markersize=3.2,
        zorder=5,
    )

    for step_index in range(1, visible_steps):
        axis.annotate(
            "",
            xy=visible_path[step_index],
            xytext=visible_path[step_index - 1],
            arrowprops={
                "arrowstyle": "->",
                "color": "#D55E00",
                "linewidth": 1.2,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            zorder=6,
        )

    axis.scatter(
        weight_history[0, 0],
        weight_history[0, 1],
        color="#222222",
        s=38,
        zorder=7,
        label="Start",
    )
    axis.scatter(
        weight_history[-1, 0],
        weight_history[-1, 1],
        color="#D55E00",
        s=45,
        zorder=7,
        label="After 25 steps",
    )


def draw_cost_contour(
    axis,
    X,
    y,
    weight_history,
    optimum_weights,
    learning_rate,
    condition_number,
    title,
    scaled,
):
    """Draw cost contours, the optimum, and the gradient descent path."""

    weight_1_grid, weight_2_grid, cost_grid = calculate_cost_surface(
        X,
        y,
        weight_history,
        optimum_weights,
    )

    minimum_cost = cost_grid.min()
    cost_range = cost_grid.max() - minimum_cost
    contour_levels = minimum_cost + np.geomspace(
        max(cost_range * 0.0015, 1e-7),
        cost_range * 0.85,
        10,
    )

    axis.contour(
        weight_1_grid,
        weight_2_grid,
        cost_grid,
        levels=contour_levels,
        cmap="Blues",
        linewidths=1.35,
    )

    draw_descent_path(axis, weight_history)

    axis.scatter(
        optimum_weights[0],
        optimum_weights[1],
        marker="*",
        s=150,
        color="#009E73",
        edgecolor="white",
        linewidth=0.7,
        zorder=8,
        label="Minimum",
    )

    axis.set_title(title, loc="left")
    axis.set_xlabel(r"Weight for Area ($w_1$)")
    axis.set_ylabel(r"Weight for Eccentricity ($w_2$)")
    axis.grid(False)
    axis.legend(frameon=False, loc="upper right")

    axis.text(
        0.03,
        0.04,
        (
            f"Condition number: {condition_number:.2e}\n"
            f"Learning rate: {learning_rate:.2e}"
        ),
        transform=axis.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#BDBDBD",
            "alpha": 0.92,
        },
    )

    if not scaled:
        axis.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def create_figure():
    """Create and save the complete feature-scaling comparison figure."""

    (
        training_df,
        X_raw,
        X_centered,
        X_scaled,
        y_centered,
        training_means,
        training_stds,
    ) = load_training_data()

    raw_results = trace_gradient_descent(
        X_centered,
        y_centered,
        step_fraction=1.5,
    )
    scaled_results = trace_gradient_descent(
        X_scaled,
        y_centered,
        step_fraction=1.0,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
        constrained_layout=True,
    )

    draw_feature_scatter(
        axes[0, 0],
        training_df,
        X_raw,
        "A. Original feature scales",
        "Area (pixels)",
        "Eccentricity",
    )
    draw_cost_contour(
        axes[0, 1],
        X_centered,
        y_centered,
        *raw_results,
        title="B. Cost surface without standardization",
        scaled=False,
    )

    draw_feature_scatter(
        axes[1, 0],
        training_df,
        X_scaled,
        "C. Standardized feature scales",
        "Standardized Area",
        "Standardized Eccentricity",
    )
    draw_cost_contour(
        axes[1, 1],
        X_scaled,
        y_centered,
        *scaled_results,
        title="D. Cost surface after standardization",
        scaled=True,
    )

    figure.suptitle(
        "Feature Scaling and Gradient Descent on the Raisin Dataset",
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        (
            "Training split only (n = 540). Target encoding: Kecimen = 0, "
            "Besni = 1. Contours show the linear-regression MSE objective; "
            "the intercept is handled by centering."
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIRECTORY / "feature_scaling_gradient_descent.png"
    pdf_path = OUTPUT_DIRECTORY / "feature_scaling_gradient_descent.pdf"

    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Training means: {training_means}")
    print(f"Training standard deviations: {training_stds}")
    print(f"Raw condition number: {raw_results[3]:.6e}")
    print(f"Scaled condition number: {scaled_results[3]:.6f}")


if __name__ == "__main__":
    create_figure()
