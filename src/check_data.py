"""Phase 1: Data understanding and basic dataset inspection."""

from utils import TARGET_COLUMN, ensure_directories, load_dataset


def display_dataset_summary() -> None:
    """Load dataset and print shape, types, missing values, and class stats."""
    ensure_directories()

    try:
        df = load_dataset()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    print("=" * 60)
    print("CREDIT CARD FRAUD DATASET SUMMARY")
    print("=" * 60)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns ({len(df.columns)}):")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        print(missing[missing > 0])

    print("\nClass Distribution:")
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    print(class_counts)

    fraud_count = class_counts.get(1, 0)
    total_count = len(df)
    fraud_percentage = (fraud_count / total_count) * 100
    print(f"\nFraud Percentage: {fraud_percentage:.4f}%")
    print(f"Legitimate Percentage: {100 - fraud_percentage:.4f}%")

    print("\nAmount Statistics:")
    print(df["Amount"].describe())


if __name__ == "__main__":
    display_dataset_summary()
