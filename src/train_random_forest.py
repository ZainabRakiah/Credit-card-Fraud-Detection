"""Phase 4 & 5: Train and tune Random Forest classifier."""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from preprocess import get_preprocessed_data
from utils import FRAUD_MODEL_PATH, MODELS_DIR, ensure_directories


def train_random_forest(tune: bool = True):
    """
    Train Random Forest model with optional hyperparameter tuning.

    Args:
        tune: Whether to run RandomizedSearchCV tuning.

    Returns:
        Tuple of best model, test features, and test target.
    """
    ensure_directories()
    X_train, X_test, y_train, y_test, _ = get_preprocessed_data()

    base_model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    if tune:
        param_grid = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring="f1",
            random_state=42,
            n_jobs=1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

        print("Best Random Forest parameters:")
        print(search.best_params_)
        print(f"Best CV F1 score: {search.best_score_:.4f}")
    else:
        model = base_model
        model.fit(X_train, y_train)

    rf_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(model, rf_path)
    joblib.dump(model, FRAUD_MODEL_PATH)
    print(f"Random Forest model saved to {rf_path}")
    print(f"Best model also saved to {FRAUD_MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_random_forest(tune=True)
