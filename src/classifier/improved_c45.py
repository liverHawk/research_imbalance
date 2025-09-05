from sklearn.tree import DecisionTreeClassifier
import pandas as pd


def check_numeric_features(X: pd.DataFrame):
    """
    Check if all features in the DataFrame are numeric.
    Raises ValueError if any feature is not numeric.
    """
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(
                f"Feature '{col}' is not numeric.",
                "Convert it to numeric before fitting."
            )


class ImprovedC45:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        load_path=None
    ):
        if load_path:
            import joblib
            self.clf = joblib.load(load_path)
        else:
            self.clf = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )

    def fit(self, X: pd.DataFrame, y):
        check_numeric_features(X)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
    def save(self, path):
        import joblib
        joblib.dump(self.clf, path)

    