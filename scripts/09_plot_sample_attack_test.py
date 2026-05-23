from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

TABLE_DIR = BASE_DIR / "results" / "tables"
FIGURE_DIR = BASE_DIR / "results" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

summary_path = TABLE_DIR / "sample_attack_prediction_summary.csv"

if not summary_path.exists():
    raise FileNotFoundError(f"Missing file: {summary_path}")

df = pd.read_csv(summary_path)

# Plot true positives, false positives, and false negatives
plot_df = df.set_index("model")[["true_positives", "false_positives", "false_negatives"]]

ax = plot_df.plot(kind="bar", figsize=(10, 6))

ax.set_title("Sample Attack Prediction Test Results")
ax.set_xlabel("Model")
ax.set_ylabel("Number of Samples")
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.xticks(rotation=0)
plt.tight_layout()

output_path = FIGURE_DIR / "sample_attack_prediction_results.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved figure to: {output_path}")