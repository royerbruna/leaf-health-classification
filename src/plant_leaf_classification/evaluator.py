
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

