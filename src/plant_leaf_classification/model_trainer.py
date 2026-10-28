
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    """Train supervised machine learning classifiers."""

    def train_logistic_regression(self, X_train, y_train):
        """Train a logistic regression model with standardized features."""
        model = make_pipeline(
            StandardScaler(),
            # Defaults: L2 regularization, C=1.0, lbfgs solver, max_iter=100.
            LogisticRegression()
        )

        model.fit(X_train, y_train)

        return model


    def train_knn(self, X_train, y_train):
        """Train a KNN classifier with standardized features."""
        model = make_pipeline(
            StandardScaler(),
            # Defaults: n_neighbors=5, weights='uniform', metric='minkowski', p=2.
            KNeighborsClassifier()
        )

        model.fit(X_train, y_train)

        return model


    def train_decision_tree(self, X_train, y_train):
        """Train a decision tree classifier."""
        # Defaults: criterion='gini', splitter='best', max_depth=None.
        model = DecisionTreeClassifier(random_state=42) # Fix the random seed to ensure reproducible results across runs.

        model.fit(X_train, y_train)

        return model
