"""
Predict sarcasm in text using trained CNN/BiRNN ensemble.

Run with --cnn --rnn to see the predictions from the base models as well.

Example usage:
python3 predict_sarcasm.py --input test.csv --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# src directory to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
from preprocessing import preprocess_text



DEFAULT_MAX_LEN = 200


def load_artifacts(models_dir: Path):
    models_dir = Path(models_dir)

    tokenizer_path = models_dir / "tokenizer.pkl"
    cnn_path = models_dir / "cnn_model.h5"
    rnn_path = models_dir / "birnn_model.h5"
    ensemble_path = models_dir / "ensemble_model.pkl"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not cnn_path.exists():
        raise FileNotFoundError(f"CNN model not found: {cnn_path}")
    if not rnn_path.exists():
        raise FileNotFoundError(f"BiRNN model not found: {rnn_path}")
    if not ensemble_path.exists():
        raise FileNotFoundError(f"Ensemble model not found: {ensemble_path}")

    # Load models
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    cnn_model = load_model(cnn_path)
    rnn_model = load_model(rnn_path)
    with open(ensemble_path, "rb") as f:
        ensemble_model = pickle.load(f)

    return tokenizer, cnn_model, rnn_model, ensemble_model


def build_stacking_features(cnn_probs: np.ndarray, rnn_probs: np.ndarray) -> np.ndarray:
    avg = 0.5 * (cnn_probs + rnn_probs)
    diff = cnn_probs - rnn_probs
    prod = cnn_probs * rnn_probs
    return np.column_stack([cnn_probs, rnn_probs, avg, diff, prod])

# Padding is handled by the models during training not by the preprocessing, so we need to pad inputs here
def pad_text(tokenizer, texts, max_len=DEFAULT_MAX_LEN):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')


# Print out metrics
def print_base_metrics(name, probs, true_labels):

    preds = (probs > 0.5).astype(int)
    
    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average='weighted')
    print(f"\n{name} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    
    
    

def predict_sarcasm(input_csv, output_csv, models_dir="models", show_cnn_metrics=False, show_rnn_metrics=False):
    """
    Main function of predict_sarcasm
    
    :param input_csv: input file
    :param output_csv: output file
    :param models_dir: the directory where the models are stored
    :param show_cnn_metrics: optional flag to show CNN metrics
    :param show_rnn_metrics: optional flag to show BiRNN metrics
    """

    # Load the models
    print("Loading models (may take a while)...")
    tokenizer, cnn_model, rnn_model, ensemble_model = load_artifacts(models_dir)
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError(f"Input CSV must contain a 'text' column. Found columns: {list(df.columns)}")
    
    print("Calculating Prediction...")
    
    df['text_preprocessed'] = df['text'].apply(preprocess_text)
    padded_sequences = pad_text(tokenizer, df['text_preprocessed'])
    cnn_probs = cnn_model.predict(padded_sequences, verbose=0).flatten()
    rnn_probs = rnn_model.predict(padded_sequences, verbose=0).flatten()
    avg = 0.5 * (cnn_probs + rnn_probs)
    diff = cnn_probs - rnn_probs
    prod = cnn_probs * rnn_probs
    stacked_features = np.column_stack([cnn_probs, rnn_probs, avg, diff, prod])
    ensemble_probs = ensemble_model.predict_proba(stacked_features)[:, 1]
    predictions = (ensemble_probs > 0.5).astype(int)

    # Create output DataFrame
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    output_df.to_csv(output_csv, index=False)
    
    print("Predictions Complete!")
    print(f"Saved to: {output_csv}")
    print(f"Total samples: {len(output_df)}")
    print(f"Predictions: Non-sarcastic (0): {(predictions == 0).sum()}, Sarcastic (1): {(predictions == 1).sum()}")
    
    # Evaluate metrics if labels are available
    if 'label' in df.columns:
        print("--------------------------------------------------\n")
        print("Results:")

        true_labels = df['label'].values.astype(int)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\nDetailed Classification Report:\n")
        print(classification_report(true_labels, predictions, target_names=['Non-sarcastic', 'Sarcastic'], zero_division=0))
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
        
        print(f"\nPer-Class Metrics:")
        print(f"  Non-sarcastic (0):")
        print(f"    Precision: {precision_per_class[0]:.4f}")
        print(f"    Recall:    {recall_per_class[0]:.4f}")
        print(f"    F1-Score:  {f1_per_class[0]:.4f}")
        print(f"  Sarcastic (1):")
        print(f"    Precision: {precision_per_class[1]:.4f}")
        print(f"    Recall:    {recall_per_class[1]:.4f}")
        print(f"    F1-Score:  {f1_per_class[1]:.4f}")

        if show_cnn_metrics:
            print_base_metrics("CNN", cnn_probs, true_labels)
        if show_rnn_metrics:
            print_base_metrics("BiRNN", rnn_probs, true_labels)
    else:
        print("\nNote: No 'label' column found in input CSV. Skipping evaluation metrics.")
        if show_cnn_metrics or show_rnn_metrics:
            print("Base model metrics requested, but 'label' column is missing; unable to compute.")


def main():
    parser = argparse.ArgumentParser(
        description='Predict sarcasm in text using trained CNN-SVM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--input',type=str,required=True,help='Path to input CSV file (must contain a "text" column)')
    parser.add_argument('--output',type=str,required=True,help='Path to output CSV file (will contain "text" and "prediction" columns)')
    parser.add_argument('--models-dir',type=str,default='models',help='Directory containing saved models (default: models)')
    parser.add_argument('-cnn', '--cnn',action='store_true',help='Also print evaluation metrics for the CNN model (requires label column)')
    parser.add_argument('-rnn', '--rnn',action='store_true',help='Also print evaluation metrics for the BiRNN model (requires label column)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    predict_sarcasm(
        input_csv=str(input_path),
        output_csv=args.output,
        models_dir=str(models_dir),
        show_cnn_metrics=args.cnn,
        show_rnn_metrics=args.rnn
    )


if __name__ == "__main__":
    main()
