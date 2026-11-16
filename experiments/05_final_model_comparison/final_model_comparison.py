"""Run the final four-model comparison on the held-out test split."""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.linear_regression_classifier import LinearRegressionClassifier
from src.logistic_regression_classifier import LogisticRegressionClassifier
from src.metrics import classification_metrics, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "raisin" / "Raisin_Dataset.xlsx"
RESULT_DIRECTORY = PROJECT_ROOT / "reports" / "results"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures" / "evaluation"

TARGET_COLUMN = "Class"
TARGET_ENCODING = {"Kecimen": 0, "Besni": 1}
RANDOM_SEED = 42


VALIDATION_CANDIDATES = [
    {
        "candidate_id": "A-LIN-1",
        "feature_set": "A: Scaled original features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.005,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "A-LIN-2",
        "feature_set": "A: Scaled original features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "A-LIN-3",
        "feature_set": "A: Scaled original features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.02,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "A-LIN-4",
        "feature_set": "A: Scaled original features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.01,
        "threshold": 0.45,
    },
    {
        "candidate_id": "A-LOG-1",
        "feature_set": "A: Scaled original features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "A-LOG-2",
        "feature_set": "A: Scaled original features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.50,
    },
    {
        "candidate_id": "A-LOG-3",
        "feature_set": "A: Scaled original features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.40,
    },
    {
        "candidate_id": "A-LOG-4",
        "feature_set": "A: Scaled original features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 1000,
        "lambda_": 0.01,
        "threshold": 0.50,
    },
    {
        "candidate_id": "B-LIN-1",
        "feature_set": "B: Degree-two features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.005,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "B-LIN-2",
        "feature_set": "B: Degree-two features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.50,
    },
    {
        "candidate_id": "B-LIN-3",
        "feature_set": "B: Degree-two features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.45,
    },
    {
        "candidate_id": "B-LIN-4",
        "feature_set": "B: Degree-two features",
        "model_name": "Linear",
        "model_class": LinearRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.01,
        "threshold": 0.50,
    },
    {
        "candidate_id": "B-LOG-1",
        "feature_set": "B: Degree-two features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.01,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.50,
    },
    {
        "candidate_id": "B-LOG-2",
        "feature_set": "B: Degree-two features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.40,
    },
    {
        "candidate_id": "B-LOG-3",
        "feature_set": "B: Degree-two features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 1000,
        "lambda_": 0.0,
        "threshold": 0.50,
    },
    {
        "candidate_id": "B-LOG-4",
        "feature_set": "B: Degree-two features",
        "model_name": "Logistic",
        "model_class": LogisticRegressionClassifier,
        "learning_rate": 0.05,
        "num_iterations": 2000,
        "lambda_": 0.01,
        "threshold": 0.45,
    },
]

PRIMARY_VALIDATION_METRIC = "validation_accuracy"
TIE_BREAK_VALIDATION_METRIC = "validation_f1_score"


def encode_target(target_series):
    """Encode Kecimen as 0 and Besni as 1."""

    encoded_target = target_series.map(TARGET_ENCODING)
    if encoded_target.isna().any():
        raise ValueError("Unexpected target class.")
    return encoded_target.to_numpy(dtype=np.int64)


def standardize(X, means, standard_deviations):
    """Apply training-derived standardization statistics."""

    return (X - means) / standard_deviations


def create_polynomial_features(X):
    """Create original, squared, and pairwise interaction features."""

    feature_blocks = [X, X**2]
    interactions = []

    for first_index in range(X.shape[1]):
        for second_index in range(first_index + 1, X.shape[1]):
            interactions.append(
                X[:, first_index] * X[:, second_index]
            )

    feature_blocks.append(np.column_stack(interactions))
    return np.column_stack(feature_blocks)


def prepare_datasets():
    """Create the fixed 60/20/20 split and both feature sets."""

    raisin_df = pd.read_excel(DATA_PATH)
    training_df, temporary_df = train_test_split(
        raisin_df,
        test_size=0.40,
        random_state=RANDOM_SEED,
        stratify=raisin_df[TARGET_COLUMN],
    )
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_df[TARGET_COLUMN],
    )

    feature_names = [
        column
        for column in raisin_df.columns
        if column != TARGET_COLUMN
    ]

    X_train_raw = training_df[feature_names].to_numpy(dtype=np.float64)
    X_validation_raw = validation_df[feature_names].to_numpy(dtype=np.float64)
    X_test_raw = test_df[feature_names].to_numpy(dtype=np.float64)

    y_train = encode_target(training_df[TARGET_COLUMN])
    y_validation = encode_target(validation_df[TARGET_COLUMN])
    y_test = encode_target(test_df[TARGET_COLUMN])

    original_means = X_train_raw.mean(axis=0)
    original_stds = X_train_raw.std(axis=0)
    safe_original_stds = np.where(original_stds == 0, 1.0, original_stds)

    X_train_a = standardize(
        X_train_raw,
        original_means,
        safe_original_stds,
    )
    X_validation_a = standardize(
        X_validation_raw,
        original_means,
        safe_original_stds,
    )
    X_test_a = standardize(
        X_test_raw,
        original_means,
        safe_original_stds,
    )

    X_train_polynomial = create_polynomial_features(X_train_a)
    X_validation_polynomial = create_polynomial_features(X_validation_a)
    X_test_polynomial = create_polynomial_features(X_test_a)

    polynomial_means = X_train_polynomial.mean(axis=0)
    polynomial_stds = X_train_polynomial.std(axis=0)
    safe_polynomial_stds = np.where(
        polynomial_stds == 0,
        1.0,
        polynomial_stds,
    )

    X_train_b = standardize(
        X_train_polynomial,
        polynomial_means,
        safe_polynomial_stds,
    )
    X_validation_b = standardize(
        X_validation_polynomial,
        polynomial_means,
        safe_polynomial_stds,
    )
    X_test_b = standardize(
        X_test_polynomial,
        polynomial_means,
        safe_polynomial_stds,
    )

    feature_sets = {
        "A: Scaled original features": (
            X_train_a,
            X_validation_a,
            X_test_a,
        ),
        "B: Degree-two features": (
            X_train_b,
            X_validation_b,
            X_test_b,
        ),
    }

    return feature_sets, y_train, y_validation, y_test


def predict_with_threshold(model, X, threshold):
    """Use the correct continuous output and apply the selected threshold."""

    if isinstance(model, LogisticRegressionClassifier):
        scores = model.predict_proba(X)
    else:
        scores = model.predict_score(X)

    return (scores >= threshold).astype(np.int64)


def select_final_configurations(feature_sets, y_train, y_validation):
    """Select one candidate per model and feature set using validation data."""

    validation_rows = []

    for candidate in VALIDATION_CANDIDATES:
        X_train, X_validation, _ = feature_sets[candidate["feature_set"]]
        model = candidate["model_class"](
            learning_rate=candidate["learning_rate"],
            num_iterations=candidate["num_iterations"],
            lambda_=candidate["lambda_"],
        )
        model.fit(X_train, y_train)

        if not model.cost_history[-1] < model.cost_history[0]:
            raise RuntimeError(
                f'Cost did not decrease for {candidate["candidate_id"]}.'
            )

        validation_predictions = predict_with_threshold(
            model,
            X_validation,
            candidate["threshold"],
        )
        validation_metrics = classification_metrics(
            y_validation,
            validation_predictions,
        )

        validation_rows.append({
            "candidate_id": candidate["candidate_id"],
            "feature_set": candidate["feature_set"],
            "model": candidate["model_name"],
            "learning_rate": candidate["learning_rate"],
            "num_iterations": candidate["num_iterations"],
            "lambda": candidate["lambda_"],
            "threshold": candidate["threshold"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_precision": validation_metrics["precision"],
            "validation_recall": validation_metrics["recall"],
            "validation_f1_score": validation_metrics["f1_score"],
            "initial_cost": model.cost_history[0],
            "final_cost": model.cost_history[-1],
            "cost_decreased": True,
            "selected": False,
        })

    validation_results = pd.DataFrame(validation_rows)
    selected_ids = []

    grouped_results = validation_results.groupby(
        ["feature_set", "model"],
        sort=False,
    )
    for _, group in grouped_results:
        ranked_group = group.sort_values(
            [
                PRIMARY_VALIDATION_METRIC,
                TIE_BREAK_VALIDATION_METRIC,
                "candidate_id",
            ],
            ascending=[False, False, True],
        )
        selected_ids.append(ranked_group.iloc[0]["candidate_id"])

    validation_results.loc[
        validation_results["candidate_id"].isin(selected_ids),
        "selected",
    ] = True

    candidate_lookup = {
        candidate["candidate_id"]: candidate
        for candidate in VALIDATION_CANDIDATES
    }
    selected_configurations = [
        candidate_lookup[candidate_id]
        for candidate_id in selected_ids
    ]

    return selected_configurations, validation_results


def run_final_comparison():
    """Fit selected configurations and evaluate the test split once."""

    feature_sets, y_train, y_validation, y_test = prepare_datasets()
    final_configurations, validation_results = select_final_configurations(
        feature_sets,
        y_train,
        y_validation,
    )

    result_rows = []
    trained_models = []

    for configuration in final_configurations:
        X_train, X_validation, X_test = feature_sets[
            configuration["feature_set"]
        ]

        model = configuration["model_class"](
            learning_rate=configuration["learning_rate"],
            num_iterations=configuration["num_iterations"],
            lambda_=configuration["lambda_"],
        )
        model.fit(X_train, y_train)

        training_predictions = predict_with_threshold(
            model,
            X_train,
            configuration["threshold"],
        )
        validation_predictions = predict_with_threshold(
            model,
            X_validation,
            configuration["threshold"],
        )
        test_predictions = predict_with_threshold(
            model,
            X_test,
            configuration["threshold"],
        )

        training_metrics = classification_metrics(
            y_train,
            training_predictions,
        )
        validation_metrics = classification_metrics(
            y_validation,
            validation_predictions,
        )
        test_metrics = classification_metrics(y_test, test_predictions)
        test_confusion = confusion_matrix(y_test, test_predictions)

        result_rows.append({
            "feature_set": configuration["feature_set"],
            "model": configuration["model_name"],
            "selected_candidate": configuration["candidate_id"],
            "learning_rate": configuration["learning_rate"],
            "num_iterations": configuration["num_iterations"],
            "lambda": configuration["lambda_"],
            "threshold": configuration["threshold"],
            "training_accuracy": training_metrics["accuracy"],
            "validation_accuracy": validation_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_specificity": test_metrics["specificity"],
            "test_f1_score": test_metrics["f1_score"],
            "TN": int(test_confusion[0, 0]),
            "FP": int(test_confusion[0, 1]),
            "FN": int(test_confusion[1, 0]),
            "TP": int(test_confusion[1, 1]),
            "initial_cost": model.cost_history[0],
            "final_cost": model.cost_history[-1],
        })
        trained_models.append((configuration, model, test_confusion))

    results_df = pd.DataFrame(result_rows)
    return results_df, trained_models, validation_results


def save_confusion_matrices(trained_models):
    """Save four test confusion matrices in one figure."""

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for axis, (configuration, _, matrix) in zip(
        axes.flat,
        trained_models,
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=90)

        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                    color=(
                        "white"
                        if matrix[row_index, column_index] > 45
                        else "#222222"
                    ),
                )

        axis.set_title(
            f'{configuration["feature_set"][0]} — {configuration["model_name"]}'
        )
        axis.set_xticks([0, 1], ["Kecimen", "Besni"])
        axis.set_yticks([0, 1], ["Kecimen", "Besni"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")

    figure.colorbar(image, ax=axes, shrink=0.75, label="Number of samples")
    figure.suptitle(
        "Final Test Confusion Matrices",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "final_test_confusion_matrices.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "final_test_confusion_matrices.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def save_cost_curves(trained_models):
    """Save the four final training cost histories."""

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for axis, (configuration, model, _) in zip(
        axes.flat,
        trained_models,
    ):
        axis.plot(model.cost_history, color="#0072B2", linewidth=2.0)
        axis.set_title(
            f'{configuration["feature_set"][0]} — {configuration["model_name"]}'
        )
        axis.set_xlabel("Gradient descent iteration")
        axis.set_ylabel("Cost")
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)

    figure.suptitle(
        "Training Cost Curves for the Final Configurations",
        fontsize=16,
        fontweight="bold",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "final_training_cost_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        FIGURE_DIRECTORY / "final_training_cost_curves.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    """Execute and save the final evaluation artifacts."""

    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    results_df, trained_models, validation_results = run_final_comparison()
    validation_path = RESULT_DIRECTORY / "validation_model_selection.csv"
    results_path = RESULT_DIRECTORY / "final_model_comparison.csv"
    validation_results.to_csv(
        validation_path,
        index=False,
        float_format="%.6f",
    )
    results_df.to_csv(results_path, index=False, float_format="%.6f")

    save_confusion_matrices(trained_models)
    save_cost_curves(trained_models)

    display_columns = [
        "feature_set",
        "model",
        "validation_accuracy",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    print(results_df[display_columns].to_string(index=False))
    print(f"\nSaved: {validation_path}")
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
