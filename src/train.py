import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from pathlib import Path
import pickle
import json
from preprocessing import preprocess_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# Step 1: Load all datasets
# ============================================================================
print("=" * 70)
print("Step 1: Loading datasets")
print("=" * 70)
# Use relative paths - go up from src/ to project root
project_root = Path(__file__).parent.parent

train_df = pd.read_csv(project_root / "train.csv")
valid_df = pd.read_csv(project_root / "valid.csv")
test_df = pd.read_csv(project_root / "test.csv")

print(f"Training set: {len(train_df)} samples")
print(f"Validation set: {len(valid_df)} samples")
print(f"Test set: {len(test_df)} samples")

# ============================================================================
# Step 2: Preprocess text for all datasets
# ============================================================================
print("\n" + "=" * 70)
print("Step 2: Preprocessing text (normalize whitespace, lowercase)")
print("=" * 70)

train_df['text_preprocessed'] = train_df['text'].apply(preprocess_text)
valid_df['text_preprocessed'] = valid_df['text'].apply(preprocess_text)
test_df['text_preprocessed'] = test_df['text'].apply(preprocess_text)

print("Text preprocessing complete for all datasets.")

# ============================================================================
# Step 3: Tokenize and create sequences
# ============================================================================
print("\n" + "=" * 70)
print("Step 3: Tokenizing text and creating sequences")
print("=" * 70)

# Tokenize based on training data only
max_words = 10000  # Vocabulary size
max_len = 200  # Maximum sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df['text_preprocessed'])

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(train_df['text_preprocessed'])
X_valid_seq = tokenizer.texts_to_sequences(valid_df['text_preprocessed'])
X_test_seq = tokenizer.texts_to_sequences(test_df['text_preprocessed'])

# Pad sequences to same length
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_valid_padded = pad_sequences(X_valid_seq, maxlen=max_len, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Extract labels
y_train = train_df['label'].values
y_valid = valid_df['label'].values
y_test = test_df['label'].values

print(f"Sequence shapes:")
print(f"  Training: {X_train_padded.shape}")
print(f"  Validation: {X_valid_padded.shape}")
print(f"  Test: {X_test_padded.shape}")
print(f"  Vocabulary size: {len(tokenizer.word_index)}")

# ============================================================================
# Stage A: Train CNN classifier
# ============================================================================
print("\n" + "=" * 70)
print("Stage A: Training CNN classifier")
print("=" * 70)

embedding_dim = 128
num_filters = 64
filter_sizes = [3, 4, 5]  # Different filter sizes to capture different n-gram patterns

# Build CNN model
model_input = tf.keras.Input(shape=(max_len,), name='input')
embedding = Embedding(max_words, embedding_dim, input_length=max_len)(model_input)

# Multiple Conv1D layers with different filter sizes
conv_layers = []
for filter_size in filter_sizes:
    conv = Conv1D(num_filters, filter_size, activation='relu')(embedding)
    pool = GlobalMaxPooling1D()(conv)
    conv_layers.append(pool)

# Concatenate all pooled features
concatenated = Concatenate()(conv_layers) if len(conv_layers) > 1 else conv_layers[0]
dropout = Dropout(0.5)(concatenated)
dense = Dense(64, activation='relu', name='dense')(dropout)
output = Dense(1, activation='sigmoid', name='classification')(dense)

cnn_model = Model(inputs=model_input, outputs=output)
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("\nCNN Model Architecture:")
cnn_model.summary()

# Train CNN
print("\nTraining CNN...")
history = cnn_model.fit(
    X_train_padded, y_train,
    validation_data=(X_valid_padded, y_valid),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate CNN on validation set
print("\nEvaluating CNN on validation set...")
cnn_val_pred = (cnn_model.predict(X_valid_padded) > 0.5).astype(int).flatten()
cnn_val_accuracy = accuracy_score(y_valid, cnn_val_pred)
cnn_val_f1 = f1_score(y_valid, cnn_val_pred, average='weighted')
print(f"CNN Validation Accuracy: {cnn_val_accuracy:.4f}")
print(f"CNN Validation F1-score: {cnn_val_f1:.4f}")

# ============================================================================
# Stage B: Extract features and train SVM
# ============================================================================
print("\n" + "=" * 70)
print("Stage B: Extracting CNN features and training SVM")
print("=" * 70)

# Create feature extraction model (remove final classification layer)
# Extract features from the dense layer before the final output
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('dense').output)

print("\nExtracting features from CNN...")
X_train_features = feature_extractor.predict(X_train_padded, verbose=0)
X_valid_features = feature_extractor.predict(X_valid_padded, verbose=0)
X_test_features = feature_extractor.predict(X_test_padded, verbose=0)

print(f"Extracted feature shapes:")
print(f"  Training: {X_train_features.shape}")
print(f"  Validation: {X_valid_features.shape}")
print(f"  Test: {X_test_features.shape}")

# Train SVM on extracted features
print("\nTraining SVM on CNN features...")
svm_model = LinearSVC(C=1.0, max_iter=2000, random_state=42, dual=False)
svm_model.fit(X_train_features, y_train)

# Evaluate SVM on validation set
print("\nEvaluating SVM on validation set...")
svm_val_pred = svm_model.predict(X_valid_features)
svm_val_accuracy = accuracy_score(y_valid, svm_val_pred)
svm_val_f1 = f1_score(y_valid, svm_val_pred, average='weighted')
print(f"SVM Validation Accuracy: {svm_val_accuracy:.4f}")
print(f"SVM Validation F1-score: {svm_val_f1:.4f}")

# ============================================================================
# Final evaluation on test set
# ============================================================================
print("\n" + "=" * 70)
print("Final Evaluation on Test Set")
print("=" * 70)

# Test CNN
cnn_test_pred = (cnn_model.predict(X_test_padded, verbose=0) > 0.5).astype(int).flatten()
cnn_test_accuracy = accuracy_score(y_test, cnn_test_pred)
cnn_test_f1 = f1_score(y_test, cnn_test_pred, average='weighted')

# Test SVM
svm_test_pred = svm_model.predict(X_test_features)
svm_test_accuracy = accuracy_score(y_test, svm_test_pred)
svm_test_f1 = f1_score(y_test, svm_test_pred, average='weighted')

print(f"\nCNN-SVM Model Results:")
print(f"  CNN Test Accuracy: {cnn_test_accuracy:.4f}")
print(f"  CNN Test F1-score: {cnn_test_f1:.4f}")
print(f"  SVM Test Accuracy: {svm_test_accuracy:.4f}")
print(f"  SVM Test F1-score: {svm_test_f1:.4f}")

print(f"\nDetailed Classification Report (SVM):")
print(classification_report(y_test, svm_test_pred, target_names=['Non-sarcastic', 'Sarcastic']))

# ============================================================================
# Save models and artifacts
# ============================================================================
print("\n" + "=" * 70)
print("Saving models and artifacts")
print("=" * 70)

# Create models directory (relative path from project root)
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

# Save CNN feature extractor model
feature_extractor_path = models_dir / "cnn_feature_extractor.h5"
feature_extractor.save(str(feature_extractor_path))
print(f"Saved CNN feature extractor to: {feature_extractor_path}")

# Save SVM model
svm_path = models_dir / "svm_model.pkl"
with open(svm_path, 'wb') as f:
    pickle.dump(svm_model, f)
print(f"Saved SVM model to: {svm_path}")

# Save tokenizer
tokenizer_path = models_dir / "tokenizer.pkl"
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"Saved tokenizer to: {tokenizer_path}")

# Save model parameters
model_params = {
    'max_words': max_words,
    'max_len': max_len,
    'embedding_dim': embedding_dim,
    'num_filters': num_filters,
    'filter_sizes': filter_sizes
}
params_path = models_dir / "model_params.json"
with open(params_path, 'w') as f:
    json.dump(model_params, f, indent=2)
print(f"Saved model parameters to: {params_path}")

print("\n" + "=" * 70)
print("Training pipeline complete!")
print("=" * 70)
