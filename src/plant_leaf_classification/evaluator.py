
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


class Evaluator:
    """Evaluate classification models using performance metrics."""

    def calculate_metrics(self, y_true, y_pred) -> dict:
        """Calculate metrics considering diseased leaves as the positive class."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, pos_label=1),
            "recall": recall_score(y_true, y_pred, pos_label=1),
            "f1_score": f1_score(y_true, y_pred, pos_label=1),
        }


    def aggregate_fold_metrics(self, results: pd.DataFrame) -> pd.DataFrame:
        """Calculate the mean and standard deviation of metrics across folds."""
        metrics = ["accuracy", "precision", "recall", "f1_score"]

        return results.groupby("model")[metrics].agg(["mean", "std"])


    def calculate_species_metrics(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Calculate classification metrics for each species and model."""
        results = []

        for (species, model), group in predictions.groupby(["species", "model"]):
            y_true = group["actual"]
            y_pred = group["predicted"]

            if (y_true == 1).any():
                metrics = self.calculate_metrics(y_true, y_pred)
            else:
                # If there are no diaseased samples: positive class metrics are marked as NaN
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "f1_score": float("nan"),
                }

            results.append({
                "species": species,
                "model": model,
                "images": len(group),
                "healthy_images": (y_true == 0).sum(),
                "diseased_images": (y_true == 1).sum(),
                **metrics,
            })

        return pd.DataFrame(results)
