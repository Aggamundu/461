import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    csv_path = Path(__file__).parent.parent / "train.csv"

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in train.csv")

    total_samples = len(df)

    label_counts = df["label"].value_counts().sort_index()
    non_sarcastic = int(label_counts.get(0, 0))
    sarcastic = int(label_counts.get(1, 0))

    non_sarcastic_pct = non_sarcastic / total_samples * 100 if total_samples else 0
    sarcastic_pct = sarcastic / total_samples * 100 if total_samples else 0

    diff_pct = abs(non_sarcastic_pct - sarcastic_pct)
    balanced = diff_pct <= 10

    print("=== Dataset Summary ===")
    print(f"Total text samples: {total_samples}")
    print(f"Sarcastic examples (label=1): {sarcastic} ({sarcastic_pct:.2f}%)")
    print(f"Non-sarcastic examples (label=0): {non_sarcastic} ({non_sarcastic_pct:.2f}%)")
    print()
    print("=== Balance Assessment ===")
    print(f"Class percentage difference: {diff_pct:.2f}%")
    print(f"Dataset is {'BALANCED' if balanced else 'IMBALANCED'} under a 10% threshold.")
    print()

    print("=== Class Distribution Table ===")
    print(f"{'Class':<20}{'Count':>10}{'Percent':>12}")
    print("-" * 42)
    print(f"{'Non-sarcastic (0)':<20}{non_sarcastic:>10}{non_sarcastic_pct:>11.2f}%")
    print(f"{'Sarcastic (1)':<20}{sarcastic:>10}{sarcastic_pct:>11.2f}%")
    
    print("\n" + "=" * 70)
    print("=== Text Length Statistics ===")
    print("=" * 70)
    
    df['char_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("\nOverall Statistics:")
    print(f"  Average character length: {df['char_length'].mean():.2f}")
    print(f"  Median character length: {df['char_length'].median():.2f}")
    print(f"  Min character length: {df['char_length'].min()}")
    print(f"  Max character length: {df['char_length'].max()}")
    print(f"  Std deviation (characters): {df['char_length'].std():.2f}")
    print()
    print(f"  Average word count: {df['word_count'].mean():.2f}")
    print(f"  Median word count: {df['word_count'].median():.2f}")
    print(f"  Min word count: {df['word_count'].min()}")
    print(f"  Max word count: {df['word_count'].max()}")
    print(f"  Std deviation (words): {df['word_count'].std():.2f}")
    
    print("\nStatistics by Class:")
    print(f"{'Metric':<25}{'Non-sarcastic (0)':>20}{'Sarcastic (1)':>20}")
    print("-" * 65)
    
    non_sarc_df = df[df['label'] == 0]
    sarc_df = df[df['label'] == 1]
    
    print(f"{'Avg char length':<25}{non_sarc_df['char_length'].mean():>19.2f}{sarc_df['char_length'].mean():>20.2f}")
    print(f"{'Median char length':<25}{non_sarc_df['char_length'].median():>19.2f}{sarc_df['char_length'].median():>20.2f}")
    print(f"{'Avg word count':<25}{non_sarc_df['word_count'].mean():>19.2f}{sarc_df['word_count'].mean():>20.2f}")
    print(f"{'Median word count':<25}{non_sarc_df['word_count'].median():>19.2f}{sarc_df['word_count'].median():>20.2f}")

    print("\n" + "=" * 70)
    print("Generating visualization tables...")
    print("=" * 70)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.axis('tight')
    ax1.axis('off')
    
    overall_data = [
        ['Average Word Count', f"{df['word_count'].mean():.2f}"],
        ['Median Word Count', f"{df['word_count'].median():.2f}"],
        ['95th Percentile Word Count', f"{df['word_count'].quantile(0.95):.2f}"]
    ]
    
    table1 = ax1.table(cellText=overall_data,
                      colLabels=['Metric', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.6, 0.4])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.8)
    
    for i in range(len(overall_data) + 1):
        for j in range(2):
            cell = table1[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
    
    ax1.set_title('Overall Text Length Statistics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.axis('tight')
    ax2.axis('off')
    
    class_data = [
        ['Average Character Length', 
         f"{non_sarc_df['char_length'].mean():.2f}", 
         f"{sarc_df['char_length'].mean():.2f}"],
        ['Median Character Length', 
         f"{non_sarc_df['char_length'].median():.2f}", 
         f"{sarc_df['char_length'].median():.2f}"],
        ['Average Word Count', 
         f"{non_sarc_df['word_count'].mean():.2f}", 
         f"{sarc_df['word_count'].mean():.2f}"],
        ['Median Word Count', 
         f"{non_sarc_df['word_count'].median():.2f}", 
         f"{sarc_df['word_count'].median():.2f}"]
    ]
    
    table2 = ax2.table(cellText=class_data,
                       colLabels=['Metric', 'Non-sarcastic (0)', 'Sarcastic (1)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.5, 0.25, 0.25])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 2.0)
    
    for i in range(len(class_data) + 1):
        for j in range(3):
            cell = table2[(i, j)]
            if i == 0:
                cell.set_facecolor('#2196F3')
                cell.set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('white')
                if j == 1:
                    cell.set_facecolor('#E3F2FD' if i % 2 == 0 else '#BBDEFB')
                elif j == 2:
                    cell.set_facecolor('#FFEBEE' if i % 2 == 0 else '#FFCDD2')
    
    ax2.set_title('Text Length Statistics by Class', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(5, 4))
    classes = ["Non-sarcastic (0)", "Sarcastic (1)"]
    counts = [non_sarcastic, sarcastic]
    plt.bar(classes, counts, color=["steelblue", "salmon"])
    plt.ylabel("Count")
    plt.title("Fig. 1. Class Distribution: Sarcastic vs Non-sarcastic")
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts) * 0.01, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
