
import pandas as pd

from plant_leaf_classification.evaluator import Evaluator
from plant_leaf_classification.model_trainer import ModelTrainer


class ExperimentRunner:
    """Run classification experiments using stratified cross-validation."""

    def run_experiment(self, data: pd.DataFrame, folds: list) -> pd.DataFrame:
        """Run the three classifiers using the five predefined folds. In each round, the models are trained on four folds and tested on the remaining one. The evaluation metrics are calculated and saved for each model and fold."""
        
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


    def collect_species_predictions(self, data: pd.DataFrame, folds: list) -> pd.DataFrame:
        """Collect predictions from the three classifiers across the five folds. For each validation image, save its path, species, correct label, and predicted label. These records are later used to evaluate model performance separately for each species."""

        trainer = ModelTrainer()

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

            print(f"\nFold {fold_number}")

            for model_name, train_method in models.items():
                model = train_method(X_train, y_train)
                predictions = model.predict(X_val)

                results.append(pd.DataFrame({
                    "image_path": data.iloc[val_indices]["image_path"].to_numpy(),
                    "species": data.iloc[val_indices]["species"].to_numpy(),
                    "model": model_name,
                    "actual": y.iloc[val_indices].to_numpy(),
                    "predicted": predictions,
                }))

                print(f"{model_name}: {len(predictions)} predictions")

        return pd.concat(results, ignore_index=True)
