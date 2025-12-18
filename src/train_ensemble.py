import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from data_utils import PreparedData, evaluate_split, prepare_data, save_tokenizer_and_params


def _default_project_root() -> Path:
    return Path(__file__).parent.parent


def _load_model(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at {path}. Train and save it first.")
    print(f"Loading {label} from {path}")
    return load_model(path)


def _load_saved_params(models_dir: Path) -> dict:
    params_path = models_dir / "model_params.json"
    if params_path.exists():
        with open(params_path, "r") as f:
            return json.load(f)
    return {}


def _build_features(cnn_probs: np.ndarray, rnn_probs: np.ndarray) -> np.ndarray:
    avg = 0.5 * (cnn_probs + rnn_probs)
    diff = cnn_probs - rnn_probs
    prod = cnn_probs * rnn_probs
    return np.column_stack([cnn_probs, rnn_probs, avg, diff, prod])


def train_ensemble(data: Optional[PreparedData] = None, project_root: Optional[Path] = None,
                   cnn_model=None, cnn_config: Optional[dict] = None, rnn_model=None, rnn_config: Optional[dict] = None):
    if project_root is None:
        project_root = _default_project_root()

    if data is None:
        data = prepare_data(project_root)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    saved_params = _load_saved_params(models_dir)

    if cnn_model is None:
        cnn_model = _load_model(models_dir / "cnn_model.h5", "CNN model")
    if cnn_config is None:
        cnn_config = saved_params.get("cnn", {})

    if rnn_model is None:
        rnn_model = _load_model(models_dir / "birnn_model.h5", "BiRNN model")
    if rnn_config is None:
        rnn_config = saved_params.get("rnn", {})

    print("\nGenerating stacking features...")
    cnn_valid = cnn_model.predict(data.X_valid, verbose=0).flatten()
    rnn_valid = rnn_model.predict(data.X_valid, verbose=0).flatten()
    valid_features = _build_features(cnn_valid, rnn_valid)

    stacker = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            max_iter=2000,
            scoring="f1",
            solver="liblinear",
            class_weight="balanced"
        )),
    ])

    print("Training ensemble (logistic regression with CV)...")
    val_cv_probs = cross_val_predict(stacker, valid_features, data.y_valid,
                                     cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                     method="predict_proba")[:, 1]

    val_metrics = evaluate_split(data.y_valid, val_cv_probs)
    print(f"Ensemble Validation Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Ensemble Validation F1-score: {val_metrics['f1']:.4f}")

    stacker.fit(valid_features, data.y_valid)

    cnn_test = cnn_model.predict(data.X_test, verbose=0).flatten()
    rnn_test = rnn_model.predict(data.X_test, verbose=0).flatten()
    test_features = _build_features(cnn_test, rnn_test)
    test_probs = stacker.predict_proba(test_features)[:, 1]
    test_metrics = evaluate_split(data.y_test, test_probs)
    print(f"\nEnsemble Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Ensemble Test F1-score: {test_metrics['f1']:.4f}")
    print("\nDetailed Classification Report (Ensemble):")
    print(classification_report(data.y_test, test_metrics["predictions"], target_names=["Non-sarcastic", "Sarcastic"]))

    ensemble_path = models_dir / "ensemble_model.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(stacker, f)
    print(f"Saved ensemble model to: {ensemble_path}")

    save_tokenizer_and_params(models_dir, data, {
        "max_words": data.max_words,
        "max_len": data.max_len,
        "cnn": cnn_config,
        "rnn": rnn_config,
        "ensemble": {"type": "logistic_regression_cv", "features": ["cnn", "rnn", "avg", "diff", "prod"]},
    })

    return stacker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ensemble layer using saved CNN and BiRNN models.")
    parser.parse_args()
    train_ensemble()


if __name__ == "__main__":
    main()
