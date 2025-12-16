#!/usr/bin/env python3
"""
Predict sarcasm in text using trained CNN-SVM model.

Usage:
    python predict_sarcasm.py --input test_data.csv --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Add src directory to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from preprocessing import preprocess_text


def load_models(models_dir):
    """Load all saved models and artifacts."""
    # Ensure models_dir is a relative path
    models_dir = Path(models_dir)
    
    # Load model parameters
    with open(models_dir / "model_params.json", 'r') as f:
        params = json.load(f)
    
    # Load tokenizer
    with open(models_dir / "tokenizer.pkl", 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load CNN feature extractor
    cnn_model = load_model(models_dir / "cnn_feature_extractor.h5")
    
    # Load SVM model
    with open(models_dir / "svm_model.pkl", 'rb') as f:
        svm_model = pickle.load(f)
    
    return cnn_model, svm_model, tokenizer, params


def predict_sarcasm(input_csv, output_csv, models_dir="models"):
    """
    Predict sarcasm for texts in input CSV and save to output CSV.
    
    Args:
        input_csv: Path to input CSV file with 'text' column
        output_csv: Path to output CSV file
        models_dir: Directory containing saved models
    """
    print("=" * 70)
    print("Loading models...")
    print("=" * 70)
    
    # Load models
    cnn_model, svm_model, tokenizer, params = load_models(models_dir)
    
    print(f"Loaded models from: {models_dir}")
    print(f"Model parameters: max_words={params['max_words']}, max_len={params['max_len']}")
    
    # Load input data
    print("\n" + "=" * 70)
    print("Loading input data...")
    print("=" * 70)
    
    df = pd.read_csv(input_csv)
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise ValueError(f"Input CSV must contain a 'text' column. Found columns: {list(df.columns)}")
    
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Preprocess text
    print("\n" + "=" * 70)
    print("Preprocessing text...")
    print("=" * 70)
    
    df['text_preprocessed'] = df['text'].apply(preprocess_text)
    
    # Tokenize and pad sequences
    print("\n" + "=" * 70)
    print("Tokenizing and creating sequences...")
    print("=" * 70)
    
    sequences = tokenizer.texts_to_sequences(df['text_preprocessed'])
    padded_sequences = pad_sequences(
        sequences, 
        maxlen=params['max_len'], 
        padding='post', 
        truncating='post'
    )
    
    print(f"Sequence shape: {padded_sequences.shape}")
    
    # Extract features using CNN
    print("\n" + "=" * 70)
    print("Extracting features using CNN...")
    print("=" * 70)
    
    features = cnn_model.predict(padded_sequences, verbose=0)
    print(f"Feature shape: {features.shape}")
    
    # Predict using SVM
    print("\n" + "=" * 70)
    print("Predicting with SVM...")
    print("=" * 70)
    
    predictions = svm_model.predict(features)
    predictions = predictions.astype(int)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    # Save predictions
    output_df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 70)
    print("Predictions complete!")
    print("=" * 70)
    print(f"Saved predictions to: {output_csv}")
    print(f"Total samples: {len(output_df)}")
    print(f"Predictions - Non-sarcastic (0): {(predictions == 0).sum()}, Sarcastic (1): {(predictions == 1).sum()}")
    
    # Evaluate metrics if labels are available
    if 'label' in df.columns:
        print("\n" + "=" * 70)
        print("Evaluation Metrics (Test Set)")
        print("=" * 70)
        
        true_labels = df['label'].values.astype(int)
        
        # Calculate metrics
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
        print(f"\nDetailed Classification Report:")
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
    else:
        print("\nNote: No 'label' column found in input CSV. Skipping evaluation metrics.")


def main():
    parser = argparse.ArgumentParser(
        description='Predict sarcasm in text using trained CNN-SVM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_sarcasm.py --input test_data.csv --output predictions.csv
  python predict_sarcasm.py --input data/test.csv --output results/predictions.csv --models-dir models
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (must contain a "text" column)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file (will contain "text" and "prediction" columns)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing saved models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Validate models directory exists
    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Run prediction
    predict_sarcasm(
        input_csv=str(input_path),
        output_csv=args.output,
        models_dir=str(models_dir)
    )


if __name__ == "__main__":
    main()

