"""Phase 6 & 7: Evaluate models on the test set and compare performance."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from preprocess import get_preprocessed_data
from utils import MODELS_DIR, RESULTS_DIR, ensure_directories


def load_trained_models():
    """Load all trained model artifacts from disk."""
    import joblib

    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
        "Decision Tree": MODELS_DIR / "decision_tree.pkl",
        "Random Forest": MODELS_DIR / "random_forest.pkl",
    }

    models = {}
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. Train all models before evaluation."
            )
        models[name] = joblib.load(path)
    return models


def compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics for a single model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, y_proba),
    }


def save_classification_report(X_test, y_test, models):
    """Generate and save detailed classification reports for all models."""
    report_lines = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        report_lines.append(f"{'=' * 60}")
        report_lines.append(name)
        report_lines.append(f"{'=' * 60}")
        report_lines.append(
            classification_report(
                y_test,
                y_pred,
                target_names=["Legitimate", "Fraud"],
                zero_division=0,
            )
        )

    report_path = RESULTS_DIR / "classification_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Classification report saved to {report_path}")


def plot_confusion_matrix(X_test, y_test, models):
    """Plot confusion matrices for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    output_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curve(X_test, y_test, models):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()

    output_path = RESULTS_DIR / "roc_curve.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {output_path}")


def create_model_comparison(X_test, y_test, models):
    """Build comparison table and identify the best model."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        rows.append(
            {
                "Model": name,
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1 Score"],
                "ROC AUC": metrics["ROC AUC"],
            }
        )

    comparison_df = pd.DataFrame(rows)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    best_model = comparison_df.sort_values("F1", ascending=False).iloc[0]
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest model by F1 score: {best_model['Model']}")
    print(f"Comparison table saved to {comparison_path}")

    return comparison_df


def plot_feature_importance(top_n: int = 15):
    """Plot and print top feature importances from the Random Forest model."""
    import joblib

    model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        print("Random Forest model not found. Skipping feature importance.")
        return

    model = joblib.load(model_path)
    importance_df = pd.DataFrame(
        {
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1],
        color="teal",
    )
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Random Forest Feature Importances")
    plt.tight_layout()

    output_path = RESULTS_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nTop {top_n} Feature Importances:")
    for rank, row in enumerate(importance_df.itertuples(index=False), start=1):
        print(f"{rank:2d}. {row.Feature}: {row.Importance:.6f}")

    print(f"Feature importance plot saved to {output_path}")


def evaluate_models():
    """Run full evaluation pipeline on the held-out test set."""
    ensure_directories()
    _, X_test, _, y_test, _ = get_preprocessed_data()
    models = load_trained_models()

    print("Evaluating models on the test set...")
    save_classification_report(X_test, y_test, models)
    plot_confusion_matrix(X_test, y_test, models)
    plot_roc_curve(X_test, y_test, models)
    create_model_comparison(X_test, y_test, models)
    plot_feature_importance()


if __name__ == "__main__":
    evaluate_models()
