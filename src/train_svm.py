"""
SVM training wrapper.

The main baseline training implementation is available in:
scripts/03_train_baselines.py
"""

import subprocess
import sys
from pathlib import Path


def train_svm():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_train_baselines.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


if __name__ == "__main__":
    train_svm()