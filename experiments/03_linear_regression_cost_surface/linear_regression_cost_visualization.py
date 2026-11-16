"""Create regression, contour, and 3D cost-surface visualizations.

The demonstration uses standardized Area as a single input and the encoded
Raisin class as the target. It visualizes the same squared-error objective used
by LinearRegressionClassifier.
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


def compute_cost(feature, target, weight, bias):
    """Calculate J(w, b) using the squared-error objective."""

    predictions = feature * weight + bias
    errors = predictions - target
    return np.sum(errors**2) / (2 * feature.shape[0])


def run_gradient_descent(
    feature,
    target,
    learning_rate=0.15,
    number_of_iterations=80,
):
    """Train the one-feature model and retain the optimization path."""

    weight = 0.0
    bias = 0.0
    number_of_samples = feature.shape[0]

    weight_history = [weight]
    bias_history = [bias]
    cost_history = [compute_cost(feature, target, weight, bias)]

    for _ in range(number_of_iterations):
        predictions = feature * weight + bias
        errors = predictions - target

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
    """Evaluate J(w, b) on a two-dimensional parameter grid."""

    weight_span = weight_history.max() - weight_history.min()
    bias_span = bias_history.max() - bias_history.min()

    weight_values = np.linspace(
        weight_history.min() - 0.35 * weight_span,
        weight_history.max() + 0.35 * weight_span,
        240,
    )
    bias_values = np.linspace(
        bias_history.min() - 0.25 * bias_span,
        bias_history.max() + 0.25 * bias_span,
        240,
    )
    weight_grid, bias_grid = np.meshgrid(weight_values, bias_values)

    mean_x_squared = np.mean(feature**2)
    mean_x = np.mean(feature)
    mean_xy = np.mean(feature * target)
    mean_y = np.mean(target)
    mean_y_squared = np.mean(target**2)

    cost_grid = 0.5 * (
        mean_x_squared * weight_grid**2
        + 2 * mean_x * weight_grid * bias_grid
        + bias_grid**2
        - 2 * mean_xy * weight_grid
        - 2 * mean_y * bias_grid
        + mean_y_squared
    )

    return weight_grid, bias_grid, cost_grid


def draw_regression_graph(
    axis,
    feature,
    target,
    class_names,
    final_weight,
    final_bias,
):
    """Draw the data, fitted line, and classification threshold."""

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

    line_x = np.linspace(feature.min(), feature.max(), 300)
    line_y = line_x * final_weight + final_bias

    axis.plot(
        line_x,
        line_y,
        color="#222222",
        linewidth=2.4,
        label="Fitted linear score",
    )
    axis.axhline(
        0.5,
        color="#D55E00",
        linestyle="--",
        linewidth=1.5,
        label="Decision threshold = 0.5",
    )

    if final_weight != 0:
        decision_location = (0.5 - final_bias) / final_weight
        axis.axvline(
            decision_location,
            color="#D55E00",
            linestyle=":",
            linewidth=1.3,
        )

    axis.set_title("A. One-feature linear regression classifier", loc="left")
    axis.set_xlabel("Standardized Area")
    axis.set_ylabel("Encoded class / predicted score")
    axis.set_ylim(-0.15, 1.15)
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
    """Draw J(w, b) contours and the gradient descent path."""

    minimum_cost = cost_grid.min()
    cost_range = cost_grid.max() - minimum_cost
    contour_levels = minimum_cost + np.geomspace(
        max(cost_range * 0.001, 1e-7),
        cost_range * 0.90,
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
        markersize=3.2,
        markevery=5,
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

    axis.set_title("B. Contour plot of the cost function", loc="left")
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
    """Draw the 3D cost surface and optimization path."""

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

    axis.set_title("C. Three-dimensional cost surface", loc="left")
    axis.set_xlabel(r"Weight ($w$)", labelpad=8)
    axis.set_ylabel(r"Bias ($b$)", labelpad=8)
    axis.set_zlabel(r"Cost $J(w,b)$", labelpad=8)
    axis.view_init(elev=28, azim=-125)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left")


def create_figure():
    """Create, save, and report the three-panel visualization."""

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

    regression_axis = figure.add_subplot(grid[0, 0])
    contour_axis = figure.add_subplot(grid[0, 1])
    surface_axis = figure.add_subplot(grid[0, 2], projection="3d")

    draw_regression_graph(
        regression_axis,
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
        "Linear Regression Cost Geometry and Gradient Descent",
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
    png_path = OUTPUT_DIRECTORY / "linear_regression_cost_visualization.png"
    pdf_path = OUTPUT_DIRECTORY / "linear_regression_cost_visualization.pdf"

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
