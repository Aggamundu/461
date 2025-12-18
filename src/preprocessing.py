import pandas as pd
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def normalize_whitespace(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def apply_tfidf(texts, max_features=None, min_df=1, max_df=1.0, ngram_range=(1, 1), analyzer='word'):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        analyzer=analyzer,
        lowercase=False
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return tfidf_matrix, vectorizer, feature_names


def preprocess_dataset(csv_path=None, text_column='text', save_preprocessed=False, output_path=None):
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "train.csv"
    else:
        csv_path = Path(csv_path)
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    print("Applying text preprocessing (lowercasing and whitespace normalization)...")
    df['text_preprocessed'] = df[text_column].apply(preprocess_text)
    
    print(f"Preprocessed {len(df)} samples.")
    print(f"Sample original text: {df[text_column].iloc[0][:100]}...")
    print(f"Sample preprocessed text: {df['text_preprocessed'].iloc[0][:100]}...")
    
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
    df = preprocess_dataset(csv_path=csv_path, text_column=text_column, save_preprocessed=False)
    
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
    print("=" * 60)
    print("Data Preprocessing Pipeline")
    print("=" * 60)
    
    print("\n[Option 1] Text Preprocessing Only")
    print("-" * 60)
    df_preprocessed = preprocess_dataset(save_preprocessed=False)
    
    print("\n\n[Option 2] Text Preprocessing + TF-IDF Vectorization")
    print("-" * 60)
    df, tfidf_matrix, vectorizer, feature_names = preprocess_and_vectorize(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
