"""Phase 10: Terminal-based credit card fraud prediction system."""

import sys

import joblib
import numpy as np
import pandas as pd

from split_data import load_train_test_split
from utils import (
    FRAUD_MODEL_PATH,
    SCALE_COLUMNS,
    SCALER_PATH,
    class_label,
    extract_features_and_target,
    load_dataset,
    transform_features,
)


def load_artifacts():
    """Load saved model and scaler for inference."""
    if not FRAUD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {FRAUD_MODEL_PATH}. Run save_model.py first."
        )
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run save_model.py first."
        )

    model = joblib.load(FRAUD_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_transaction(model, scaler, transaction: pd.DataFrame):
    """
    Predict fraud probability for a single transaction.

    Args:
        model: Trained classifier.
        scaler: Fitted StandardScaler.
        transaction: Single-row feature dataframe.

    Returns:
        Tuple of predicted class and fraud probability.
    """
    scaled = transform_features(transaction, scaler)
    probability = model.predict_proba(scaled)[0, 1]
    prediction = int(probability >= 0.5)
    return prediction, probability


def test_random_transaction(model, scaler):
    """Select and evaluate a random transaction from the test set."""
    df = load_dataset()
    X, y = extract_features_and_target(df)
    _, test_idx = load_train_test_split()

    random_pos = np.random.randint(0, len(test_idx))
    row_idx = test_idx[random_pos]

    transaction = X.loc[[row_idx]]
    actual = int(y.loc[row_idx])
    predicted, probability = predict_transaction(model, scaler, transaction)

    print("\nRandom Test Transaction")
    print("-" * 40)
    print(f"Transaction ID: {row_idx}")
    print(f"Actual Class: {class_label(actual)}")
    print(f"Predicted Class: {class_label(predicted)}")
    print(f"Fraud Probability: {probability * 100:.2f}%")

    if actual == predicted:
        print("Prediction Result: ✓ Correct Prediction")
    else:
        print("Prediction Result: ✗ Incorrect Prediction")


def prompt_manual_input(model, scaler):
    """Collect manual feature values and run prediction."""
    print("\nEnter transaction details:")
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    values = {}

    try:
        for feature in feature_names:
            raw_value = input(f"{feature}: ").strip()
            values[feature] = float(raw_value)
    except ValueError:
        print("Invalid input. Please enter numeric values only.")
        return

    transaction = pd.DataFrame([values])
    predicted, probability = predict_transaction(model, scaler, transaction)

    print("\nPrediction Result")
    print("-" * 40)
    print(f"Prediction: {class_label(predicted)}")
    print(f"Probability: {probability * 100:.2f}%")


def display_menu():
    """Print the interactive prediction menu."""
    print("\n=================================")
    print(" CREDIT CARD FRAUD DETECTOR")
    print("=================================")
    print("1. Test random transaction from test set")
    print("2. Enter transaction manually")
    print("3. Exit")


def main():
    """Run the terminal prediction application."""
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    while True:
        display_menu()
        choice = input("\nSelect an option (1-3): ").strip()

        if choice == "1":
            test_random_transaction(model, scaler)
        elif choice == "2":
            prompt_manual_input(model, scaler)
        elif choice == "3":
            print("Exiting fraud detector. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
