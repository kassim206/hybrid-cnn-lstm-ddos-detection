import kagglehub

# Download the dataset to see what files it contains
print("Downloading dataset to check file structure...")
path = kagglehub.dataset_download("rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files")
print(f"Dataset downloaded to: {path}")

import os
print("\nFiles in dataset:")
print("-" * 50)
for root, dirs, files in os.walk(path):
    for file in files:
        full_path = os.path.join(root, file)
        size = os.path.getsize(full_path) / (1024 * 1024)  # Size in MB
        print(f"  {file} ({size:.1f} MB)")
        print(f"    Path: {full_path}")
