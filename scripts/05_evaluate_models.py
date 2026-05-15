from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = BASE_DIR / "results" / "metrics"
TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    baseline_path = METRICS_DIR / "baseline_model_results.csv"
    cnn_lstm_path = METRICS_DIR / "cnn_lstm_metrics.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing file: {baseline_path}")

    if not cnn_lstm_path.exists():
        raise FileNotFoundError(f"Missing file: {cnn_lstm_path}")

    baseline_df = pd.read_csv(baseline_path)
    cnn_lstm_df = pd.read_csv(cnn_lstm_path)

    comparison_df = pd.concat([baseline_df, cnn_lstm_df], ignore_index=True)

    return comparison_df


def save_comparison_table(comparison_df):
    metrics_path = METRICS_DIR / "model_comparison.csv"
    table_path = TABLE_DIR / "model_comparison.csv"

    comparison_df.to_csv(metrics_path, index=False)
    comparison_df.to_csv(table_path, index=False)

    print("Model comparison table:")
    print(comparison_df)

    print(f"\nSaved comparison files:")
    print(f"- {metrics_path}")
    print(f"- {table_path}")


def plot_model_performance(comparison_df):
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    plot_df = comparison_df.set_index("model")[metric_columns]

    ax = plot_df.plot(kind="bar", figsize=(11, 6))

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURE_DIR / "model_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Performance comparison figure saved to: {output_path}")


def plot_latency_comparison(comparison_df):
    ax = comparison_df.plot(
        x="model",
        y="latency_ms_per_sample",
        kind="bar",
        figsize=(9, 5),
        legend=False
    )

    ax.set_title("Inference Latency Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency per Sample (ms)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = FIGURE_DIR / "latency_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Latency comparison figure saved to: {output_path}")


def main():
    print("=" * 70)
    print("EVALUATING MODEL COMPARISON")
    print("=" * 70)

    comparison_df = load_metrics()

    save_comparison_table(comparison_df)
    plot_model_performance(comparison_df)
    plot_latency_comparison(comparison_df)

    print("\nModel evaluation comparison completed successfully.")


if __name__ == "__main__":
    main()