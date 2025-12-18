import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.optimizers import Adam

from data_utils import PreparedData, evaluate_split, prepare_data

DEFAULT_EMBEDDING_DIM = 16
DEFAULT_NUM_FILTERS = 36
DEFAULT_FILTER_SIZES = [3, 4, 20]

np.random.seed(42)
tf.random.set_seed(42)


def build_cnn(data: PreparedData,
              embedding_dim: int,
              num_filters: int,
              filter_sizes,
              dropout_rate: float = 0.1) -> Model:

    model_input = tf.keras.Input(shape=(data.max_len,), name="input")
    embedding = Embedding(data.max_words, embedding_dim, input_length=data.max_len)(model_input)

    conv_layers = []
    for size in filter_sizes:
        conv = Conv1D(num_filters, size, activation="relu")(embedding)
        pool = GlobalMaxPooling1D()(conv)
        conv_layers.append(pool)

    concatenated = Concatenate()(conv_layers) if len(conv_layers) > 1 else conv_layers[0]
    dropout = Dropout(dropout_rate)(concatenated)
    dense = Dense(24, activation="relu", name="dense")(dropout)
    output = Dense(1, activation="sigmoid", name="classification")(dense)

    model = Model(inputs=model_input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_cnn(data: Optional[PreparedData] = None,
              project_root: Optional[Path] = None,
              epochs: int = 2,
              batch_size: int = 64,
              embedding_dim: int = DEFAULT_EMBEDDING_DIM,
              num_filters: int = DEFAULT_NUM_FILTERS,
              filter_sizes=None) -> Tuple[PreparedData, Model, dict]:
    if data is None:
        data = prepare_data(project_root)
    filter_sizes = filter_sizes or DEFAULT_FILTER_SIZES

    print("\n" + "=" * 70)
    print("Training CNN classifier")
    print("=" * 70)

    model = build_cnn(data, embedding_dim, num_filters, filter_sizes)
    model.summary()
    model.fit(
        data.X_train, data.y_train,
        validation_data=(data.X_valid, data.y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    val_metrics = evaluate_split(data.y_valid, model.predict(data.X_valid, verbose=0).flatten())
    print(f"CNN Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"CNN Validation F1-score: {val_metrics['f1']:.4f}")

    config = {
        "embedding_dim": embedding_dim,
        "num_filters": num_filters,
        "filter_sizes": filter_sizes
    }
    return data, model, config


def save_cnn(model: Model, data: PreparedData) -> None:
    models_dir = data.project_root / "models"
    models_dir.mkdir(exist_ok=True)

    cnn_path = models_dir / "cnn_model.h5"
    model.save(str(cnn_path))
    print(f"Saved CNN model to: {cnn_path}")

    feature_extractor = Model(inputs=model.input, outputs=model.get_layer("dense").output)
    feature_extractor_path = models_dir / "cnn_feature_extractor.h5"
    feature_extractor.save(str(feature_extractor_path))
    print(f"Saved CNN feature extractor to: {feature_extractor_path}")

def describe_results(model: Model, data: PreparedData) -> None:
    probs = model.predict(data.X_test, verbose=0).flatten()
    metrics = evaluate_split(data.y_test, probs)
    print("\nCNN Test Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-score: {metrics['f1']:.4f}")
    print("\nDetailed Classification Report (CNN):")
    print(classification_report(data.y_test, metrics["predictions"], target_names=["Non-sarcastic", "Sarcastic"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN sarcasm classifier.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data, model, _ = train_cnn(epochs=args.epochs, batch_size=args.batch_size)
    describe_results(model, data)
    save_cnn(model, data)


if __name__ == "__main__":
    main()
