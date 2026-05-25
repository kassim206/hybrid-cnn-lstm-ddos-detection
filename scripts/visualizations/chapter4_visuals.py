import os
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
THESIS_FIGURES_DIR = BASE_DIR / "thesis_figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required real experiment file not found: {path}\n"
            "Run the corresponding experiment script first. "
            "This visualization script does not generate synthetic or fallback results."
        )


def save_figure(fig, filename: str):
    output_png = FIGURES_DIR / filename
    thesis_png = THESIS_FIGURES_DIR / filename

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(thesis_png, dpi=300, bbox_inches="tight")

    print(f"Saved: {output_png}")
    print(f"Saved: {thesis_png}")


def plot_training_curves():
    print("\nGenerating real CNN-LSTM training curves...")

    history_path = METRICS_DIR / "cnn_lstm_training_history.json"
    require_file(history_path)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    required_keys = ["loss", "val_loss", "accuracy", "val_accuracy"]
    for key in required_keys:
        if key not in history:
            raise KeyError(f"Missing key '{key}' in {history_path}")

    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(epochs, history["loss"], label="Training Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["accuracy"], label="Training Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        "CNN-LSTM Training Loss and Accuracy Curves from Real Training History",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_figure(fig, "figure_4.2_training_curves.png")
    plt.close(fig)


def plot_model_comparison():
    print("\nGenerating model comparison from real CSV...")

    comparison_path = TABLES_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        comparison_path = METRICS_DIR / "model_comparison.csv"

    require_file(comparison_path)

    df = pd.read_csv(comparison_path)

    required_columns = ["model", "accuracy", "precision", "recall", "f1_score"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {comparison_path}")

    plot_df = df.set_index("model")[["accuracy", "precision", "recall", "f1_score"]]
    plot_df = plot_df.rename(
        columns={
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df.plot(kind="bar", ax=ax, edgecolor="black")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Performance Comparison of Random Forest, SVM, and CNN-LSTM Models",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0.95, 1.01)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_figure(fig, "figure_4.3_performance_comparison.png")
    plt.close(fig)


def plot_latency_comparison():
    print("\nGenerating latency comparison from real CSV...")

    comparison_path = TABLES_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        comparison_path = METRICS_DIR / "model_comparison.csv"

    require_file(comparison_path)

    df = pd.read_csv(comparison_path)

    required_columns = ["model", "latency_ms_per_sample"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {comparison_path}")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.bar(
        df["model"],
        df["latency_ms_per_sample"],
        edgecolor="black",
    )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Latency (ms/sample)", fontsize=12)
    ax.set_title(
        "Inference Latency Comparison from Real Evaluation Results",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(df["latency_ms_per_sample"]):
        ax.text(i, value, f"{value:.6f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    save_figure(fig, "figure_4.4_latency_comparison.png")
    plt.close(fig)


def plot_sample_attack_prediction_results():
    print("\nGenerating sample attack prediction graph from real CSV...")

    summary_path = TABLES_DIR / "sample_attack_prediction_summary.csv"
    require_file(summary_path)

    df = pd.read_csv(summary_path)

    required_columns = ["model", "true_positives", "false_positives", "false_negatives"]
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in {summary_path}")

    plot_df = df.set_index("model")[["true_positives", "false_positives", "false_negatives"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df.plot(kind="bar", ax=ax, edgecolor="black")

    ax.set_title("Sample Attack Prediction Test Results", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()

    save_figure(fig, "sample_attack_prediction_results.png")
    plt.close(fig)


def main():
    print("=" * 70)
    print("CHAPTER 4 FIGURE GENERATION FROM REAL EXPERIMENT OUTPUTS")
    print("=" * 70)

    plot_training_curves()
    plot_model_comparison()
    plot_latency_comparison()
    plot_sample_attack_prediction_results()

    print("\nAll figures generated from real saved experiment outputs.")
    print("No synthetic fallback graphs or hardcoded metrics were used.")


if __name__ == "__main__":
    main()