import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Build path to train.csv relative to this file:
    # project_root/train.csv (using relative path)
    csv_path = Path(__file__).parent.parent / "train.csv"

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Expect a column named 'label' where 1 = sarcastic, 0 = non-sarcastic
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in train.csv")

    total_samples = len(df)

    # Count sarcastic vs non-sarcastic
    label_counts = df["label"].value_counts().sort_index()  # 0 then 1
    non_sarcastic = int(label_counts.get(0, 0))
    sarcastic = int(label_counts.get(1, 0))

    non_sarcastic_pct = non_sarcastic / total_samples * 100 if total_samples else 0
    sarcastic_pct = sarcastic / total_samples * 100 if total_samples else 0

    # Decide if dataset is balanced (here: both classes within 10 percentage points)
    diff_pct = abs(non_sarcastic_pct - sarcastic_pct)
    balanced = diff_pct <= 10

    # Text summary
    print("=== Dataset Summary ===")
    print(f"Total text samples: {total_samples}")
    print(f"Sarcastic examples (label=1): {sarcastic} ({sarcastic_pct:.2f}%)")
    print(f"Non-sarcastic examples (label=0): {non_sarcastic} ({non_sarcastic_pct:.2f}%)")
    print()
    print("=== Balance Assessment ===")
    print(f"Class percentage difference: {diff_pct:.2f}%")
    print(f"Dataset is {'BALANCED' if balanced else 'IMBALANCED'} under a 10% threshold.")
    print()

    # Small table
    print("=== Class Distribution Table ===")
    print(f"{'Class':<20}{'Count':>10}{'Percent':>12}")
    print("-" * 42)
    print(f"{'Non-sarcastic (0)':<20}{non_sarcastic:>10}{non_sarcastic_pct:>11.2f}%")
    print(f"{'Sarcastic (1)':<20}{sarcastic:>10}{sarcastic_pct:>11.2f}%")

    # Bar chart
    plt.figure(figsize=(5, 4))
    classes = ["Non-sarcastic (0)", "Sarcastic (1)"]
    counts = [non_sarcastic, sarcastic]
    plt.bar(classes, counts, color=["steelblue", "salmon"])
    plt.ylabel("Count")
    plt.title("Class Distribution: Sarcastic vs Non-sarcastic")
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts) * 0.01, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

