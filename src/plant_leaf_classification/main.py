
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from plant_leaf_classification.dataset_manager import DatasetManager
from plant_leaf_classification.evaluator import Evaluator
from plant_leaf_classification.experiment_runner import ExperimentRunner
from plant_leaf_classification.feature_extractor import FeatureExtractor

# Dataset prerequisite: Download PlantVillage before running the experiment.
# Run from the project root directory: git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/plantvillage
# Expected image directory: data/plantvillage/raw/color/


def save_features(images: pd.DataFrame, extract_method, file_path: str):
    """Extract features from each image and combine them with its original information, such as species and class. Save the results in a CSV file so the features can be reused without processing the images again."""
    
    records = []

    for image in images.itertuples(index=False):
        features = extract_method(image.image_path)

        records.append({
            "image_path": image.image_path,
            "species": image.species,
            "original_class": image.original_class,
            "binary_label": image.binary_label,
            **features,
        })

        if len(records) % 5000 == 0:
            print(f"Processed {len(records)} images")

    data = pd.DataFrame(records)
    data.to_csv(file_path, index=False)

    print(f"Saved {len(data)} images with {len(data.columns)} columns")


def validate_features():
    """Check whether the three feature datasets were generated correctly. Verify the number of images and features, check for missing or invalid values, and confirm that all datasets contain the same images and labels."""

    datasets = {
        "rgb_basic": 6,
        "rgb_percentiles": 15,
        "rgb_histogram": 54,
    }

    reference = None

    for name, feature_count in datasets.items():
        data = pd.read_csv(f"data/features/{name}.csv")

        assert len(data) == 54305
        assert len(data.columns) == feature_count + 4
        assert data["image_path"].is_unique
        assert not data.isna().any().any()
        assert data["binary_label"].isin([0, 1]).all()

        features = data.iloc[:, 4:]
        assert np.isfinite(features.to_numpy(dtype=float)).all()

        metadata = data[
            ["image_path", "species", "original_class", "binary_label"]
        ].sort_values("image_path").reset_index(drop=True)

        if reference is None:
            reference = metadata
        else:
            assert metadata.equals(reference)

        print(f"{name}: OK ({len(data)} images, {feature_count} features)")


def extract_features():
    """Load the PlantVillage images and assign their binary labels. Extract the three sets of RGB features from the original images and save each set in a separate CSV file."""

    manager = DatasetManager("data/plantvillage/raw/color")
    images = manager.map_binary_labels(manager.load_images())
    extractor = FeatureExtractor()

    save_features(images, extractor.extract_rgb_basic, "data/features/rgb_basic.csv")
    save_features(images, extractor.extract_rgb_percentiles, "data/features/rgb_percentiles.csv")
    save_features(images, extractor.extract_rgb_histogram, "data/features/rgb_histogram.csv")


def run_experiments():
    """Balance the dataset and divide it into five stratified folds. Run the three classifiers with each feature set, always using the same images and folds. Save the evaluation results and calculate the average performance of each model."""

    manager = DatasetManager("data/plantvillage/raw/color")

    base_data = pd.read_csv("data/features/rgb_basic.csv")
    base_data = base_data.sort_values("image_path").reset_index(drop=True)

    base_data = manager.balance_binary_classes(base_data)

    folds = manager.create_stratified_folds(base_data)

    datasets = {
        "rgb_basic": "data/features/rgb_basic.csv",
        "rgb_percentiles": "data/features/rgb_percentiles.csv",
        "rgb_histogram": "data/features/rgb_histogram.csv",
    }

    runner = ExperimentRunner()
    evaluator = Evaluator()

    for name, path in datasets.items():
        data = pd.read_csv(path)
        data = data[data["image_path"].isin(base_data["image_path"])]
        data = data.sort_values("image_path").reset_index(drop=True)

        assert data["image_path"].equals(base_data["image_path"])
        assert data["binary_label"].equals(base_data["binary_label"])

        print(f"\nRunning experiment: {name}")

        results = runner.run_experiment(data, folds)
        results.to_csv(f"results/{name}_folds.csv", index=False)

        summary = evaluator.aggregate_fold_metrics(results)

        print(f"\nAggregated results ({name}):")
        print(summary.round(4))


def create_experiment_summary():
    """Combine the results from all nine experimental configurations. Calculate the mean and standard deviation of each metric across the five folds and save the summary in a single CSV file."""

    datasets = {
        "RGB Basic": "results/rgb_basic_folds.csv",
        "RGB Percentiles": "results/rgb_percentiles_folds.csv",
        "RGB Histogram": "results/rgb_histogram_folds.csv",
    }

    evaluator = Evaluator()
    summaries = []

    for name, path in datasets.items():
        results = pd.read_csv(path)
        summary = evaluator.aggregate_fold_metrics(results)

        summary.columns = [
            f"{metric}_{stat}" for metric, stat in summary.columns
        ]

        summary = summary.reset_index()
        summary.insert(0, "feature_set", name)

        summaries.append(summary)

    final_summary = pd.concat(summaries, ignore_index=True)
    final_summary.to_csv("results/experiment_summary.csv", index=False)

    print("\nExperiment summary:")
    print(final_summary.round(4).to_string(index=False))


def analyze_species():
    """Analyze how each classifier performs on different plant species. Collect predictions from the five folds and balance the healthy and diseased images within each species. Calculate and save the metricsfor each model using the balanced samples."""

    manager = DatasetManager("data/plantvillage/raw/color")

    base_data = pd.read_csv("data/features/rgb_basic.csv")
    base_data = base_data.sort_values("image_path").reset_index(drop=True)

    base_data = manager.balance_binary_classes(base_data)

    folds = manager.create_stratified_folds(base_data)

    datasets = {
        "rgb_basic": "data/features/rgb_basic.csv",
        "rgb_percentiles": "data/features/rgb_percentiles.csv",
        "rgb_histogram": "data/features/rgb_histogram.csv",
    }

    runner = ExperimentRunner()
    evaluator = Evaluator()

    metadata = ["image_path", "species", "original_class", "binary_label"]

    for name, path in datasets.items():
        data = pd.read_csv(path)
        data = data[data["image_path"].isin(base_data["image_path"])]
        data = data.sort_values("image_path").reset_index(drop=True)

        assert data[metadata].equals(base_data[metadata])

        print(f"\nEvaluating species: {name}")

        predictions = runner.collect_species_predictions(data, folds)
        species_results = evaluator.calculate_species_metrics(predictions)

        species_results.to_csv(f"results/{name}_species.csv", index=False)

        print(f"\nSpecies evaluated: {species_results['species'].nunique()}")
        print(f"Results saved: {len(species_results)}")



def plot_confusion_matrices():
    """Generate confusion matrices for each classifier and feature set. Use predictions from the five folds to count correct and incorrect classifications of healthy and diseased leaves. Save the matrices as images and their numerical results in a CSV file."""

    manager = DatasetManager("data/plantvillage/raw/color")

    base_data = pd.read_csv("data/features/rgb_basic.csv")
    base_data = base_data.sort_values("image_path").reset_index(drop=True)

    base_data = manager.balance_binary_classes(base_data)

    folds = manager.create_stratified_folds(base_data)

    datasets = {
        "rgb_basic": "RGB Basic",
        "rgb_percentiles": "RGB Percentiles",
        "rgb_histogram": "RGB Histogram",
    }

    models = ["Logistic Regression", "KNN", "Decision Tree"]
    metadata = ["image_path", "species", "original_class", "binary_label"]

    cell_labels = [
        ["True Negative (TN)", "False Positive (FP)"],
        ["False Negative (FN)", "True Positive (TP)"],
    ]

    runner = ExperimentRunner()
    results = []

    for name, title in datasets.items():
        data = pd.read_csv(f"data/features/{name}.csv")
        data = data[data["image_path"].isin(base_data["image_path"])]
        data = data.sort_values("image_path").reset_index(drop=True)

        assert data[metadata].equals(base_data[metadata])

        print(f"\nGenerating confusion matrices: {title}")

        predictions = runner.collect_species_predictions(data, folds)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for model_name, ax in zip(models, axes):
            model_data = predictions[predictions["model"] == model_name]

            matrix = confusion_matrix(
                model_data["actual"],
                model_data["predicted"],
                labels=[0, 1]
            )

            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Healthy", "Diseased"]
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                colorbar=False,
                include_values=False
            )

            for row in range(2):
                for col in range(2):
                    value = matrix[row, col]
                    percentage = value / matrix[row].sum() * 100
                    color = "white" if value > matrix.max() / 2 else "black"

                    ax.text(
                        col, row,
                        f"{cell_labels[row][col]}\n{value:,} cases\n{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color=color
                    )

            ax.set_title(model_name)

            tn, fp, fn, tp = matrix.ravel()

            results.append({
                "feature_set": name,
                "model": model_name,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            })

        fig.suptitle(f"Confusion Matrices - {title}")

        fig.text(
            0.5, 0.02,
            f"Note: Counts represent all {len(base_data):,} balanced images evaluated across 5 stratified cross-validation folds.\n"
            "Each image is evaluated once. Percentages are calculated relative to each actual class (row).",
            ha="center",
            fontsize=9
        )

        fig.tight_layout(rect=(0, 0.07, 1, 0.95))
        fig.savefig(
            f"results/{name}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    pd.DataFrame(results).to_csv(
        "results/confusion_matrices.csv",
        index=False
    )

    print("\nConfusion matrices saved successfully.")


def plot_f1_scores():
    """Create a bar chart comparing the F1-score of all nine experimental configurations. Show the average score and standard deviation from the five folds to compare model performance and consistency."""

    data = pd.read_csv("results/experiment_summary.csv")

    means = data.pivot(
        index="feature_set",
        columns="model",
        values="f1_score_mean"
    ) * 100

    stds = data.pivot(
        index="feature_set",
        columns="model",
        values="f1_score_std"
    ) * 100

    order = ["RGB Basic", "RGB Percentiles", "RGB Histogram"]
    means = means.loc[order]
    stds = stds.loc[order, means.columns]

    fig, ax = plt.subplots(figsize=(13, 7))

    means.plot(
        kind="bar",
        yerr=stds,
        capsize=4,
        ax=ax
    )

    # Display mean and standard deviation on each bar.
    bars = [c for c in ax.containers if hasattr(c, "patches")]

    for model, container in zip(means.columns, bars):
        labels = [
            f"{mean:.2f}%\n± {std:.2f} pp"
            for mean, std in zip(means[model], stds[model])
        ]
        ax.bar_label(
            container,
            labels=labels,
            padding=4,
            fontsize=9
        )

    ax.set_title("F1-score Comparison")
    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Mean F1-score (%)")
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 10))
    ax.tick_params(axis="x", rotation=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=1, alpha=0.7, color="gray")
    ax.legend(title="Classifier")

    fig.text(
        0.5, 0.03,
        "Note: Results show mean F1-score ± standard deviation across 5 stratified cross-validation folds.\n"
        "Error bars represent ±1 standard deviation; pp = percentage points. Positive class: Diseased.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(
        "results/f1_score_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_species_accuracy():
    """Create a horizontal bar chart comparing model accuracy across plant species using the RGB histogram features. Display the results of the three classifiers for each species after balancing its samples."""
    # Accuracy is calculated from pooled validation predictions across five stratified cross-validation folds.

    data = pd.read_csv("results/rgb_histogram_species.csv")

    accuracy = data.pivot(
        index="species",
        columns="model",
        values="accuracy"
    ) * 100

    accuracy = accuracy.sort_values("KNN")

    fig, ax = plt.subplots(figsize=(13, 9))

    accuracy.plot(kind="barh", ax=ax)

    # Display only the accuracy percentage on each bar.
    for bars in ax.containers:
        if hasattr(bars, "patches"):
            ax.bar_label(
                bars,
                fmt="%.1f%%",
                padding=3,
                fontsize=8
            )

    ax.set_title("Accuracy by Species - RGB Histogram")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Species")
    ax.set_xlim(0, 108)
    ax.set_xticks(range(0, 101, 10))

    # Enable grid lines to facilitate comparisons.
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle="--", linewidth=1.2, alpha=0.7, color="gray")

    ax.legend(
        title="Classifier",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=3
    )

    fig.text(
        0.5, 0.02,
        "Note: Accuracy represents correctly classified images as a percentage of each species' total.\n"
        "Results are based on pooled validation predictions across 5 stratified cross-validation folds.",
        ha="center",
        fontsize=9
    )

    fig.tight_layout(rect=(0, 0.13, 1, 1))

    fig.savefig(
        "results/species_accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    """Run the complete plant leaf classification experiment. Prepare and validate the datasets, train and evaluate the classifiers, analyze the results by species, and generate the comparison charts."""

    # Prepare output directories
    Path("data/features").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    # Data preparation
    extract_features()
    validate_features()

    # Experiments and evaluation
    run_experiments()
    create_experiment_summary()
    analyze_species()

    # Results visualization
    plot_confusion_matrices()
    plot_f1_scores()
    plot_species_accuracy()

    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()
