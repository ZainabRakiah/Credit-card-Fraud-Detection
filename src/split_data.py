"""Phase 3: Stratified train-test split with persisted indices."""

import joblib
from sklearn.model_selection import train_test_split

from utils import (
    SPLIT_INDICES_PATH,
    ensure_directories,
    extract_features_and_target,
    load_dataset,
)


def create_train_test_split(test_size: float = 0.20, random_state: int = 42):
    """
    Create stratified train/test split and save indices.

    Args:
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of train indices and test indices.
    """
    ensure_directories()
    df = load_dataset()
    _, y = extract_features_and_target(df)

    indices = df.index.to_numpy()
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    split_data = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "random_state": random_state,
        "test_size": test_size,
    }
    joblib.dump(split_data, SPLIT_INDICES_PATH)

    print(f"Train samples: {len(train_idx)} ({100 * (1 - test_size):.0f}%)")
    print(f"Test samples: {len(test_idx)} ({100 * test_size:.0f}%)")
    print(f"Split indices saved to {SPLIT_INDICES_PATH}")

    return train_idx, test_idx


def load_train_test_split():
    """
    Load previously saved train/test indices.

    Returns:
        Tuple of train indices and test indices.
    """
    if not SPLIT_INDICES_PATH.exists():
        raise FileNotFoundError(
            f"Split indices not found at {SPLIT_INDICES_PATH}. "
            "Run split_data.py first."
        )
    split_data = joblib.load(SPLIT_INDICES_PATH)
    return split_data["train_idx"], split_data["test_idx"]


if __name__ == "__main__":
    create_train_test_split()
