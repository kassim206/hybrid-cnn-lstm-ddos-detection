from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
TABLE_DIR = BASE_DIR / "results" / "tables"
METRICS_DIR = BASE_DIR / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


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


def calculate_metrics(model, X_test, y_test, model_name):
    print(f"Evaluating {model_name}...")

    start_time = time.time()
    y_pred = model.predict(X_test)
    latency_seconds = time.time() - start_time
    latency_ms = (latency_seconds / len(X_test)) * 1000

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "latency_ms_per_sample": latency_ms,
    }

    return metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    print("\nTraining Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    model_path = MODEL_DIR / "random_forest_model.joblib"
    joblib.dump(model, model_path)

    print(f"Random Forest model saved to: {model_path}")

    return calculate_metrics(model, X_test, y_test, "Random Forest")


def train_svm(X_train, y_train, X_test, y_test):
    print("\nTraining SVM model...")

    # SVM can be slow on large datasets, so use a sample for thesis testing.
    max_samples = 30000

    if len(X_train) > max_samples:
        print(f"SVM training dataset is large. Using first {max_samples} samples for practical testing.")
        X_train_svm = X_train.iloc[:max_samples]
        y_train_svm = y_train.iloc[:max_samples]
    else:
        X_train_svm = X_train
        y_train_svm = y_train

    model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_svm, y_train_svm)

    model_path = MODEL_DIR / "svm_model.joblib"
    joblib.dump(model, model_path)

    print(f"SVM model saved to: {model_path}")

    return calculate_metrics(model, X_test, y_test, "SVM")


def main():
    print("=" * 70)
    print("TRAINING BASELINE MODELS: RANDOM FOREST AND SVM")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset()

    results = []

    rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_metrics)

    svm_metrics = train_svm(X_train, y_train, X_test, y_test)
    results.append(svm_metrics)

    results_df = pd.DataFrame(results)

    table_path = TABLE_DIR / "baseline_model_results.csv"
    metrics_path = METRICS_DIR / "baseline_model_results.csv"

    results_df.to_csv(table_path, index=False)
    results_df.to_csv(metrics_path, index=False)

    print("\nBaseline model results:")
    print(results_df)

    print(f"\nResults saved to:")
    print(f"- {table_path}")
    print(f"- {metrics_path}")

    print("\nBaseline training completed successfully.")


if __name__ == "__main__":
    main()