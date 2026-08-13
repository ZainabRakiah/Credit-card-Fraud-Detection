"""Phase 4: Train Logistic Regression classifier with balanced class weights."""

import joblib
from sklearn.linear_model import LogisticRegression

from preprocess import get_preprocessed_data
from utils import MODELS_DIR, ensure_directories


def train_logistic_regression():
    """
    Train and save a balanced Logistic Regression model.

    Returns:
        Tuple of trained model and test features/target.
    """
    ensure_directories()
    X_train, X_test, y_train, y_test, _ = get_preprocessed_data()

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / "logistic_regression.pkl"
    joblib.dump(model, model_path)
    print(f"Logistic Regression model saved to {model_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_logistic_regression()
