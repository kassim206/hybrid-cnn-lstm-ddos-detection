from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FIGURE_DIR = BASE_DIR / "results" / "figures"
METRICS_DIR = BASE_DIR / "results" / "metrics"
TABLE_DIR = BASE_DIR / "results" / "tables"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 10
RANDOM_STATE = 42


def load_dataset():
    print("Loading processed dataset...")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(DATA_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(X, y, sequence_length):
    print(f"Creating sequences with sequence length = {sequence_length}...")

    X_values = X.values.astype("float32")
    y_values = y.values.astype("int32")

    X_seq = []
    y_seq = []

    for i in range(len(X_values) - sequence_length + 1):
        X_seq.append(X_values[i:i + sequence_length])
        y_seq.append(y_values[i + sequence_length - 1])

    return np.array(X_seq), np.array(y_seq)


def build_cnn_lstm_model(input_shape):
    print(f"Building CNN-LSTM model with input shape: {input_shape}")

    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(64, return_sequences=False),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def calculate_metrics(model, X_test_seq, y_test_seq):
    print("Evaluating CNN-LSTM model...")

    start_time = time.time()
    y_pred_proba = model.predict(X_test_seq, batch_size=BATCH_SIZE).ravel()
    latency_seconds = time.time() - start_time
    latency_ms = (latency_seconds / len(X_test_seq)) * 1000

    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "model": "CNN-LSTM",
        "accuracy": accuracy_score(y_test_seq, y_pred),
        "precision": precision_score(y_test_seq, y_pred, zero_division=0),
        "recall": recall_score(y_test_seq, y_pred, zero_division=0),
        "f1_score": f1_score(y_test_seq, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test_seq, y_pred_proba),
        "latency_ms_per_sample": latency_ms,
    }

    return metrics


def plot_training_history(history):
    print("Saving CNN-LSTM training curve figure...")

    plt.figure(figsize=(10, 6))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("CNN-LSTM Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = FIGURE_DIR / "cnn_lstm_training_curves.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Training curve saved to: {output_path}")


def main():
    print("=" * 70)
    print("TRAINING HYBRID CNN-LSTM MODEL")
    print("=" * 70)

    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQUENCE_LENGTH)

    print(f"X_train_seq shape: {X_train_seq.shape}")
    print(f"X_val_seq shape:   {X_val_seq.shape}")
    print(f"X_test_seq shape:  {X_test_seq.shape}")

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    model = build_cnn_lstm_model(input_shape)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    print("Starting CNN-LSTM training...")

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    model_path = MODEL_DIR / "cnn_lstm_model.keras"
    model.save(model_path)
    print(f"CNN-LSTM model saved to: {model_path}")

    metrics = calculate_metrics(model, X_test_seq, y_test_seq)

    metrics_df = pd.DataFrame([metrics])

    metrics_path = METRICS_DIR / "cnn_lstm_metrics.csv"
    table_path = TABLE_DIR / "cnn_lstm_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(table_path, index=False)

    print("\nCNN-LSTM metrics:")
    print(metrics_df)

    print(f"\nMetrics saved to:")
    print(f"- {metrics_path}")
    print(f"- {table_path}")

    plot_training_history(history)

    print("\nCNN-LSTM training completed successfully.")


if __name__ == "__main__":
    main()