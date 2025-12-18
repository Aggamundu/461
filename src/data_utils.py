import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from preprocessing import preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


@dataclass
class PreparedData:
    project_root: Path
    tokenizer: Tokenizer
    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    max_words: int
    max_len: int

    def features(self, split: str) -> np.ndarray:
        return getattr(self, f"X_{split}")

    def labels(self, split: str) -> np.ndarray:
        return getattr(self, f"y_{split}")


def prepare_data(project_root: Optional[Path] = None, max_words: int = 10000, max_len: int = 200) -> PreparedData:
    if project_root is None:
        project_root = Path(__file__).parent.parent

    print("=" * 70)
    print("Loading datasets")
    print("=" * 70)

    train_df = pd.read_csv(project_root / "train.csv")
    valid_df = pd.read_csv(project_root / "valid.csv")
    test_df = pd.read_csv(project_root / "test.csv")

    for df in (train_df, valid_df, test_df):
        df["text_preprocessed"] = df["text"].apply(preprocess_text)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text_preprocessed"])

    def to_sequences(df: pd.DataFrame) -> np.ndarray:
        sequences = tokenizer.texts_to_sequences(df["text_preprocessed"])
        return pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    X_train = to_sequences(train_df)
    X_valid = to_sequences(valid_df)
    X_test = to_sequences(test_df)

    y_train = train_df["label"].values
    y_valid = valid_df["label"].values
    y_test = test_df["label"].values

    print("Data prepared:")
    print(f"  Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    return PreparedData(
        project_root=project_root,
        tokenizer=tokenizer,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        max_words=max_words,
        max_len=max_len,
    )


def evaluate_split(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    preds = (probs > 0.5).astype(int)
    return {"accuracy": accuracy_score(y_true, preds), "f1": f1_score(y_true, preds, average="weighted"), "predictions": preds}


def save_tokenizer_and_params(models_dir: Path, data: PreparedData, params: Dict) -> None:
    models_dir.mkdir(exist_ok=True)

    tokenizer_path = models_dir / "tokenizer.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(data.tokenizer, f)
    print(f"Saved tokenizer to: {tokenizer_path}")

    params_path = models_dir / "model_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Saved model parameters to: {params_path}")
