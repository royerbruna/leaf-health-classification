
import pandas as pd
import numpy as np

from plant_leaf_classification.dataset_manager import DatasetManager
from plant_leaf_classification.feature_extractor import FeatureExtractor
from plant_leaf_classification.model_trainer import ModelTrainer
from plant_leaf_classification.evaluator import Evaluator
from plant_leaf_classification.experiment_runner import ExperimentRunner


def save_features(images: pd.DataFrame, extract_method, file_path: str):
    """Extract and save image features to a CSV file."""
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
    """Validate the consistency of the extracted feature datasets."""
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



def main():
    """Run cross-validation experiments for the selected feature sets."""
    manager = DatasetManager("data/plantvillage/raw/color")

    # Feature extraction - already completed
    # images = manager.map_binary_labels(manager.load_images())
    # extractor = FeatureExtractor()
    # save_features(images, extractor.extract_rgb_basic, "data/features/rgb_basic.csv")
    # save_features(images, extractor.extract_rgb_percentiles, "data/features/rgb_percentiles.csv")
    # save_features(images, extractor.extract_rgb_histogram, "data/features/rgb_histogram.csv")

    # validate_features()

    base_data = pd.read_csv("data/features/rgb_basic.csv")
    base_data = base_data.sort_values("image_path").reset_index(drop=True)

    folds = manager.create_stratified_folds(base_data)

    datasets = {
        "rgb_percentiles": "data/features/rgb_percentiles.csv",
        "rgb_histogram": "data/features/rgb_histogram.csv",
    }

    runner = ExperimentRunner()
    evaluator = Evaluator()

    for name, path in datasets.items():
        data = pd.read_csv(path)
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
    """Combine the aggregated results of all feature sets."""
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


if __name__ == "__main__":
    # main()
    create_experiment_summary()
