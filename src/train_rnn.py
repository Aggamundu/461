import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.optimizers import Adam

from data_utils import PreparedData, evaluate_split, prepare_data

DEFAULT_EMBEDDING_DIM = 32
DEFAULT_RNN_UNITS = 96

np.random.seed(42)
tf.random.set_seed(42)


def build_rnn(data: PreparedData,
              embedding_dim: int,
              rnn_units: int,
              dropout_rate: float = 0.3) -> Sequential:
    model = Sequential([
        Embedding(input_dim=data.max_words, output_dim=embedding_dim, input_length=data.max_len),
        Bidirectional(LSTM(rnn_units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(rnn_units // 2, return_sequences=False)),
        Dropout(dropout_rate / 2),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_rnn(data: Optional[PreparedData] = None,
              project_root: Optional[Path] = None,
              epochs: int = 6,
              batch_size: int = 64,
              embedding_dim: int = DEFAULT_EMBEDDING_DIM,
              rnn_units: int = DEFAULT_RNN_UNITS) -> Tuple[PreparedData, Sequential, dict]:
    if data is None:
        data = prepare_data(project_root)

    print("\n" + "=" * 70)
    print("Training Bidirectional RNN classifier")
    print("=" * 70)

    model = build_rnn(data, embedding_dim, rnn_units)
    model.summary()
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=2, mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)
    ]

    model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_valid, data.y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks
    )

    val_metrics = evaluate_split(data.y_valid, model.predict(data.X_valid, verbose=0).flatten())
    print(f"BiRNN Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"BiRNN Validation F1-score: {val_metrics['f1']:.4f}")

    config = {
        "embedding_dim": embedding_dim,
        "rnn_units": rnn_units
    }
    return data, model, config


def save_rnn(model: Sequential, data: PreparedData) -> None:
    models_dir = data.project_root / "models"
    models_dir.mkdir(exist_ok=True)
    rnn_path = models_dir / "birnn_model.h5"
    model.save(str(rnn_path))
    print(f"Saved BiRNN model to: {rnn_path}")

def describe_results(model: Sequential, data: PreparedData) -> None:
    probs = model.predict(data.X_test, verbose=0).flatten()
    metrics = evaluate_split(data.y_test, probs)
    print("\nBiRNN Test Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print("\nDetailed Classification Report (BiRNN):")
    print(classification_report(data.y_test, metrics["predictions"], target_names=["Non-sarcastic", "Sarcastic"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiRNN sarcasm classifier.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data, model, _ = train_rnn(epochs=args.epochs, batch_size=args.batch_size)
    describe_results(model, data)
    save_rnn(model, data)


if __name__ == "__main__":
    main()
