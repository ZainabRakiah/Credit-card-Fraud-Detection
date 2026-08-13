"""Phase 2: Exploratory data analysis and visualization generation."""

import matplotlib.pyplot as plt
import seaborn as sns

from utils import RESULTS_DIR, TARGET_COLUMN, ensure_directories, load_dataset


def plot_class_distribution(df) -> None:
    """Plot and save legitimate vs fraud transaction counts."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]

    plt.figure(figsize=(8, 5))
    bars = [counts.get(0, 0), counts.get(1, 0)]
    sns.barplot(x=labels, y=bars, hue=labels, palette="Set2", legend=False)
    plt.title("Legitimate vs Fraud Transactions")
    plt.xlabel("Transaction Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150)
    plt.close()


def plot_amount_distribution(df) -> None:
    """Plot and save overall transaction amount distribution."""
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Amount"], bins=50, kde=True, color="steelblue")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "amount_distribution.png", dpi=150)
    plt.close()


def plot_amount_by_class(df) -> None:
    """Plot and save amount distribution comparison by class."""
    class_labels = df[TARGET_COLUMN].map({0: "Legitimate", 1: "Fraud"})
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=class_labels, y=df["Amount"], hue=class_labels, palette={"Legitimate": "green", "Fraud": "red"}, legend=False)
    plt.title("Transaction Amount by Class")
    plt.xlabel("Class")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "amount_by_class.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df) -> None:
    """Plot and save feature correlation heatmap."""
    plt.figure(figsize=(14, 12))
    correlation = df.corr()
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        linewidths=0.1,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()
    return correlation


def display_top_fraud_correlations(correlation) -> None:
    """Print top positive and negative correlations with fraud class."""
    fraud_corr = correlation[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()

    print("\nTop 10 Negative Correlations with Fraud:")
    print(fraud_corr.head(10))

    print("\nTop 10 Positive Correlations with Fraud:")
    print(fraud_corr.tail(10).sort_values(ascending=False))


def run_eda() -> None:
    """Execute full exploratory data analysis pipeline."""
    ensure_directories()

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("Generating EDA visualizations...")
    plot_class_distribution(df)
    plot_amount_distribution(df)
    plot_amount_by_class(df)
    correlation = plot_correlation_heatmap(df)
    display_top_fraud_correlations(correlation)

    print(f"\nEDA complete. Visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_eda()
