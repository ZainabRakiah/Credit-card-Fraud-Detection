"""Phase 4: Train Decision Tree classifier with balanced class weights."""

import joblib
from sklearn.tree import DecisionTreeClassifier

from preprocess import get_preprocessed_data
from utils import MODELS_DIR, ensure_directories


def train_decision_tree():
    """
    Train and save a balanced Decision Tree model.

    Returns:
        Tuple of trained model and test features/target.
    """
    ensure_directories()
    X_train, X_test, y_train, y_test, _ = get_preprocessed_data()

    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    model_path = MODELS_DIR / "decision_tree.pkl"
    joblib.dump(model, model_path)
    print(f"Decision Tree model saved to {model_path}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_decision_tree()
