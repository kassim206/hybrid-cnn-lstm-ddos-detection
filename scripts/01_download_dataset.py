from pathlib import Path

def main():
    print("Checking local dataset...")
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(raw_dir.glob("*.csv"))
    if csv_files:
        print("Dataset files found:")
        for file in csv_files:
            print(f"- {file.name}")
    else:
        print("No CSV files found in data/raw.")
        print("Please place CIC-DDoS2019 CSV files in data/raw.")

if __name__ == "__main__":
    main()