
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from plant_leaf_classification.config import RANDOM_STATE


class Evaluator:
    """Evaluate classification models using performance metrics."""

    def calculate_metrics(self, y_true, y_pred) -> dict:
        """Compare the model predictions with the correct labels to calculate accuracy, precision, recall, and F1-score. Diseased leaves are considered the positive class when calculating these metrics."""
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=1),
            "recall": recall_score(y_true, y_pred, pos_label=1),
            "f1_score": f1_score(y_true, y_pred, pos_label=1),
        }


    def aggregate_fold_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """Combine the results from the five folds for each classifier. Calculate the mean and standard deviation of each metric to show the average performance and how much it varies between folds."""

        metrics = ["accuracy", "precision", "recall", "f1_score"]

        return results.groupby("model")[metrics].agg(["mean", "std"])


    def calculate_species_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Evaluate the models separately for each plant species. Species without both healthy and diseased images are excluded, and the remaining ones are balanced by randomly selecting equal numbers from each class. The same selected images are used to calculate the metrics for all models."""

        results = []

        unique_images = predictions[
            ["image_path", "species", "actual"]
        ].drop_duplicates("image_path")

        for species, group in unique_images.groupby("species"):
            counts = group["actual"].value_counts()

            # Skip species without both classes.
            if not {0, 1}.issubset(counts.index):
                continue

            min_count = counts.min()

            # Select equal numbers of healthy and diseased images.
            balanced = group.groupby(
                "actual", group_keys=False
            ).sample(n=min_count, random_state=RANDOM_STATE)

            selected_paths = balanced["image_path"]

            species_predictions = predictions[
                (predictions["species"] == species)
                & (predictions["image_path"].isin(selected_paths))
            ]

            for model, model_group in species_predictions.groupby("model"):
                y_true = model_group["actual"]
                y_pred = model_group["predicted"]

                metrics = self.calculate_metrics(y_true, y_pred)

                results.append({
                    "species": species,
                    "model": model,
                    "images": len(model_group),
                    "healthy_images": (y_true == 0).sum(),
                    "diseased_images": (y_true == 1).sum(),
                    **metrics,
                })

        return pd.DataFrame(results)
