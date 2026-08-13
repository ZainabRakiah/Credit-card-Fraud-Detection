"""Shared utility functions for the credit card fraud detection project."""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "creditcard.csv"
SPLIT_INDICES_PATH = DATA_DIR / "split_indices.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FRAUD_MODEL_PATH = MODELS_DIR / "fraud_model.pkl"

TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Amount", "Time"]
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + SCALE_COLUMNS


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load the credit card fraud dataset from disk.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place creditcard.csv in the data/ directory."
        )
    return pd.read_csv(DATA_PATH)


def extract_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Args:
        df: Input dataframe containing the Class column.

    Returns:
        Tuple of feature matrix X and target vector y.
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    return X, y


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on Amount and Time columns using training data.

    Args:
        X_train: Training feature matrix.

    Returns:
        Fitted StandardScaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X_train[SCALE_COLUMNS])
    return scaler


def transform_features(
    X: pd.DataFrame, scaler: StandardScaler
) -> pd.DataFrame:
    """
    Scale Amount and Time columns while preserving other features.

    Args:
        X: Feature matrix to transform.
        scaler: Fitted StandardScaler for Amount and Time.

    Returns:
        Transformed feature matrix.
    """
    X_scaled = X.copy()
    X_scaled[SCALE_COLUMNS] = scaler.transform(X[SCALE_COLUMNS])
    return X_scaled


def class_label(label: int) -> str:
    """Convert numeric class label to readable string."""
    return "Fraud" if label == 1 else "Legitimate"
