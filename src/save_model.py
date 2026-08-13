"""Phase 9: Persist the best model and fitted scaler for inference."""

import joblib
import shutil

from preprocess import get_preprocessed_data
from utils import FRAUD_MODEL_PATH, MODELS_DIR, SCALER_PATH, ensure_directories


def save_artifacts():
    """
    Save the tuned Random Forest model and fitted StandardScaler.

    Ensures both fraud_model.pkl and scaler.pkl are available for prediction.
    """
    ensure_directories()

    rf_path = MODELS_DIR / "random_forest.pkl"
    if not rf_path.exists():
        raise FileNotFoundError(
            "Random Forest model not found. Run train_random_forest.py first."
        )

    # Persist best model under the production filename.
    shutil.copy(rf_path, FRAUD_MODEL_PATH)
    print(f"Fraud detection model saved to {FRAUD_MODEL_PATH}")

    # Fit and save scaler on the training split if not already present.
    if not SCALER_PATH.exists():
        get_preprocessed_data(save_scaler=True)

    if SCALER_PATH.exists():
        print(f"Scaler saved to {SCALER_PATH}")
    else:
        raise FileNotFoundError("Scaler could not be saved.")


if __name__ == "__main__":
    save_artifacts()
