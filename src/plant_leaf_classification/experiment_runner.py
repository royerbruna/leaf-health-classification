
import pandas as pd

from plant_leaf_classification.model_trainer import ModelTrainer
from plant_leaf_classification.evaluator import Evaluator


class ExperimentRunner:
    """Run classification experiments using stratified cross-validation."""

    def run_experiment(self, data: pd.DataFrame, folds: list) -> pd.DataFrame:
        """Train and evaluate all three classifiers across five folds."""
        trainer = ModelTrainer()
        evaluator = Evaluator()

        X = data.drop(columns=[
            "image_path", "species", "original_class", "binary_label"
        ])
        y = data["binary_label"]

        models = {
            "Logistic Regression": trainer.train_logistic_regression,
            "KNN": trainer.train_knn,
            "Decision Tree": trainer.train_decision_tree,
        }

        results = []

        for fold_number, (train_indices, val_indices) in enumerate(folds, start=1):
            X_train = X.iloc[train_indices]
            y_train = y.iloc[train_indices]
            X_val = X.iloc[val_indices]
            y_val = y.iloc[val_indices]

            print(f"\nFold {fold_number}")

            for model_name, train_method in models.items():
                model = train_method(X_train, y_train)
                predictions = model.predict(X_val)
                metrics = evaluator.calculate_metrics(y_val, predictions)

                results.append({
                    "fold": fold_number,
                    "model": model_name,
                    **metrics,
                })

                print(f"{model_name}: F1-score = {metrics['f1_score']:.4f}")

        return pd.DataFrame(results)
