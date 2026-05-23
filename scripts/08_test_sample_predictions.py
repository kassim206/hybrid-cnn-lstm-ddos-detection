from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("SAMPLE ATTACK DATA PREDICTION TEST")
    print("=" * 70)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    rf_model = joblib.load(MODEL_DIR / "random_forest_model.joblib")
    svm_model = joblib.load(MODEL_DIR / "svm_model.joblib")

    benign_indices = y_test[y_test == 0].head(100).index
    attack_indices = y_test[y_test == 1].head(100).index

    selected_indices = list(benign_indices) + list(attack_indices)

    X_sample = X_test.loc[selected_indices]
    y_sample = y_test.loc[selected_indices]

    print(f"Selected BENIGN samples: {len(benign_indices)}")
    print(f"Selected ATTACK samples: {len(attack_indices)}")
    print(f"Total sample size: {len(selected_indices)}")

    rf_predictions = rf_model.predict(X_sample)
    rf_probabilities = rf_model.predict_proba(X_sample)[:, 1]

    svm_predictions = svm_model.predict(X_sample)
    svm_probabilities = svm_model.predict_proba(X_sample)[:, 1]

    detailed_results = pd.DataFrame({
        "sample_index": selected_indices,
        "actual_label": y_sample.values,
        "actual_class": ["BENIGN" if label == 0 else "ATTACK" for label in y_sample.values],
        "rf_prediction": rf_predictions,
        "rf_predicted_class": ["BENIGN" if label == 0 else "ATTACK" for label in rf_predictions],
        "rf_attack_probability": rf_probabilities,
        "svm_prediction": svm_predictions,
        "svm_predicted_class": ["BENIGN" if label == 0 else "ATTACK" for label in svm_predictions],
        "svm_attack_probability": svm_probabilities,
    })

    summary_rows = []

    for model_name, predictions in [
        ("Random Forest", rf_predictions),
        ("SVM", svm_predictions),
    ]:
        tn, fp, fn, tp = confusion_matrix(y_sample, predictions).ravel()

        summary_rows.append({
            "model": model_name,
            "sample_size": len(y_sample),
            "benign_samples": len(benign_indices),
            "attack_samples": len(attack_indices),
            "accuracy": accuracy_score(y_sample, predictions),
            "precision": precision_score(y_sample, predictions, zero_division=0),
            "recall": recall_score(y_sample, predictions, zero_division=0),
            "f1_score": f1_score(y_sample, predictions, zero_division=0),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        })

    summary_results = pd.DataFrame(summary_rows)

    detailed_output = RESULTS_DIR / "sample_attack_prediction_test.csv"
    summary_output = RESULTS_DIR / "sample_attack_prediction_summary.csv"

    detailed_results.to_csv(detailed_output, index=False)
    summary_results.to_csv(summary_output, index=False)

    print("\nSample prediction summary:")
    print(summary_results)

    print("\nDetailed sample prediction output:")
    print(detailed_output)

    print("\nSummary output:")
    print(summary_output)

    print("\nSample attack data prediction test completed successfully.")


if __name__ == "__main__":
    main()