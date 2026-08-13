"""Phase 3: Feature extraction, scaling, and reusable preprocessing helpers."""

import joblib

from split_data import load_train_test_split
from utils import (
    SCALER_PATH,
    extract_features_and_target,
    fit_scaler,
    load_dataset,
    transform_features,
)


def preprocess_split(
    X_train, X_test, save_scaler: bool = True
):
    """
    Fit scaler on training data and transform train/test sets.

    Args:
        X_train: Training features.
        X_test: Test features.
        save_scaler: Whether to persist the fitted scaler.

    Returns:
        Tuple of scaled X_train and X_test.
    """
    scaler = fit_scaler(X_train)
    X_train_scaled = transform_features(X_train, scaler)
    X_test_scaled = transform_features(X_test, scaler)

    if save_scaler:
        SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)

    return X_train_scaled, X_test_scaled, scaler


def get_preprocessed_data(save_scaler: bool = False):
    """
    Load dataset, apply saved split indices, and return scaled splits.

    Args:
        save_scaler: Whether to overwrite the saved scaler file.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
    """
    from split_data import load_train_test_split

    df = load_dataset()
    X, y = extract_features_and_target(df)
    train_idx, test_idx = load_train_test_split()

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    X_train_scaled, X_test_scaled, scaler = preprocess_split(
        X_train, X_test, save_scaler=save_scaler
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
