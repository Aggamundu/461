import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def normalize_whitespace(text):
    """
    Normalize whitespace in text by:
    - Replacing multiple spaces with a single space
    - Removing leading/trailing whitespace
    - Normalizing tabs, newlines, and other whitespace characters
    """
    if not isinstance(text, str):
        return ""
    # Replace all whitespace characters (spaces, tabs, newlines, etc.) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing whitespace
    text = text.strip()
    return text


def preprocess_text(text):
    """
    Apply basic text preprocessing:
    1. Normalize whitespace
    2. Convert to lowercase
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = normalize_whitespace(text)
    return text


def apply_tfidf(texts, max_features=None, min_df=1, max_df=1.0, ngram_range=(1, 1), analyzer='word'):
    """
    Apply TF-IDF vectorization to a list of texts.
    
    Parameters:
    -----------
    texts : list or pandas Series
        List of preprocessed text strings
    max_features : int, optional
        Maximum number of features (vocabulary size). If None, use all features.
    min_df : float or int, default=1
        Minimum document frequency. If float, represents proportion of documents.
    max_df : float or int, default=1.0
        Maximum document frequency. If float, represents proportion of documents.
    ngram_range : tuple, default=(1, 1)
        Range of n-grams to extract (e.g., (1, 2) for unigrams and bigrams)
    analyzer : str, default='word'
        Whether to use 'word' or 'char' n-grams
    
    Returns:
    --------
    tfidf_matrix : scipy.sparse matrix
        TF-IDF feature matrix
    vectorizer : TfidfVectorizer
        Fitted vectorizer object (can be used for transforming new data)
    feature_names : list
        List of feature names (vocabulary)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        analyzer=analyzer,
        lowercase=False  # We already lowercased in preprocess_text
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, vectorizer, feature_names


def preprocess_dataset(csv_path=None, text_column='text', save_preprocessed=False, output_path=None):
    """
    Preprocess the entire dataset from CSV file.
    
    Parameters:
    -----------
    csv_path : str or Path, optional
        Path to the CSV file. If None, uses train.csv in project root.
    text_column : str, default='text'
        Name of the column containing text data
    save_preprocessed : bool, default=False
        Whether to save preprocessed text to a new CSV file
    output_path : str or Path, optional
        Path to save preprocessed CSV. If None and save_preprocessed=True,
        saves as 'train_preprocessed.csv' in project root.
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with preprocessed text in a new column 'text_preprocessed'
    """
    # Build path to train.csv if not provided (relative path)
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "train.csv"
    else:
        csv_path = Path(csv_path)
    
    # Load the CSV file
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Apply preprocessing
    print("Applying text preprocessing (lowercasing and whitespace normalization)...")
    df['text_preprocessed'] = df[text_column].apply(preprocess_text)
    
    print(f"Preprocessed {len(df)} samples.")
    print(f"Sample original text: {df[text_column].iloc[0][:100]}...")
    print(f"Sample preprocessed text: {df['text_preprocessed'].iloc[0][:100]}...")
    
    # Save if requested
    if save_preprocessed:
        if output_path is None:
            output_path = Path(__file__).parent.parent / "train_preprocessed.csv"
        else:
            output_path = Path(output_path)
        
        df.to_csv(output_path, index=False)
        print(f"\nPreprocessed data saved to {output_path}")
    
    return df


def preprocess_and_vectorize(csv_path=None, text_column='text', max_features=None, 
                             min_df=1, max_df=1.0, ngram_range=(1, 1)):
    """
    Preprocess text and apply TF-IDF vectorization in one step.
    
    Parameters:
    -----------
    csv_path : str or Path, optional
        Path to the CSV file. If None, uses train.csv in project root.
    text_column : str, default='text'
        Name of the column containing text data
    max_features : int, optional
        Maximum number of TF-IDF features
    min_df : float or int, default=1
        Minimum document frequency for TF-IDF
    max_df : float or int, default=1.0
        Maximum document frequency for TF-IDF
    ngram_range : tuple, default=(1, 1)
        N-gram range for TF-IDF
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with preprocessed text
    tfidf_matrix : scipy.sparse matrix
        TF-IDF feature matrix
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    feature_names : list
        List of feature names
    """
    # Preprocess the dataset
    df = preprocess_dataset(csv_path=csv_path, text_column=text_column, save_preprocessed=False)
    
    # Apply TF-IDF
    print("\nApplying TF-IDF vectorization...")
    tfidf_matrix, vectorizer, feature_names = apply_tfidf(
        df['text_preprocessed'],
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range
    )
    
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:10]}")
    
    return df, tfidf_matrix, vectorizer, feature_names


def main():
    """
    Example usage of the preprocessing functions.
    """
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    # Option 1: Just preprocess text (normalize whitespace and lowercase)
    print("\n[Option 1] Text Preprocessing Only")
    print("-" * 60)
    df_preprocessed = preprocess_dataset(save_preprocessed=False)
    
    # Option 2: Preprocess and apply TF-IDF
    print("\n\n[Option 2] Text Preprocessing + TF-IDF Vectorization")
    print("-" * 60)
    df, tfidf_matrix, vectorizer, feature_names = preprocess_and_vectorize(
        max_features=1000,  # Limit to top 1000 features for demonstration
        min_df=2,  # Ignore terms that appear in fewer than 2 documents
        max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
