
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from plant_leaf_classification.config import RANDOM_STATE


class ModelTrainer:
    """Train supervised machine learning classifiers."""

    def train_logistic_regression(self, X_train, y_train):
        """Train a Logistic Regression classifier using standardized features. The data is scaled before training so that features with different numerical ranges can be compared fairly. The trained model and scaler are returned together in a pipeline."""
        
        model = make_pipeline(
            StandardScaler(),
            # Defaults: L2 regularization, C=1.0, lbfgs solver, max_iter=100.
            LogisticRegression()
        )

        model.fit(X_train, y_train)

        return model


    def train_knn(self, X_train, y_train):
        """Train a KNN classifier using standardized features. Scaling prevents features with larger values from dominating distance calculations. The model uses five nearest neighbors by default to classify new images."""

        model = make_pipeline(
            StandardScaler(),
            # Defaults: n_neighbors=5, weights='uniform', metric='minkowski', p=2.
            KNeighborsClassifier()
        )

        model.fit(X_train, y_train)

        return model


    def train_decision_tree(self, X_train, y_train):
        """Train a Decision Tree classifier using the extracted image features. The model learns a sequence of rules to separate healthy and diseased leaves. It uses the default parameters and a fixed random seed to make the results reproducible."""

        # Defaults: criterion='gini', splitter='best', max_depth=None.
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)

        model.fit(X_train, y_train)

        return model
