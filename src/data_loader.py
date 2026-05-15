"""
Complete data loader for the CIC-DDoS2019 dataset.
Handles loading from cache, downloading, preprocessing, and splitting.
"""

import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from src.config import DATA_DIR, RANDOM_STATE

warnings.filterwarnings("ignore")


def load_cic_ddos2019(nrows=None):
    """
    Load the CIC-DDoS2019 dataset from local cache or Kaggle.
    """
    local_path = os.path.join(DATA_DIR, "cic_ddos2019_sample.csv")
    
    # Check local cache first
    if os.path.exists(local_path):
        print(f"Loading from local cache: {local_path}")
        df = pd.read_csv(local_path)
        print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        return df
    
    print("=" * 60)
    print("Downloading CIC-DDoS2019 dataset from Kaggle")
    print("=" * 60)
    
    try:
        import kagglehub
        
        # Download the dataset
        download_path = kagglehub.dataset_download("rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files")
        print(f"Downloaded to: {download_path}")
        
        # Find all CSV files
        csv_files = []
        for root, dirs, files in os.walk(download_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Load a sample from each file
        all_dfs = []
        samples_per_file = 12000 if nrows is None else max(1, nrows // len(csv_files))
        
        for i, csv_file in enumerate(csv_files[:18]):  # Load first 18 files
            file_name = os.path.basename(csv_file)
            print(f"Loading {i+1}/{min(18, len(csv_files))}: {file_name}")
            try:
                df_part = pd.read_csv(csv_file, nrows=samples_per_file)
                all_dfs.append(df_part)
                print(f"  Loaded {len(df_part):,} rows")
            except Exception as e:
                print(f"  Error: {e}")
        
        if not all_dfs:
            raise ValueError("No CSV files could be loaded")
        
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total: {len(df):,} rows with {len(df.columns)} columns")
        
        # Save local copy
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"Saved local copy to: {local_path}")
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease ensure the dataset exists in:", local_path)
        raise


def identify_label_column(df):
    """
    Identify the label column in the dataset.
    """
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'label' in col_lower:
            print(f"Label column identified: '{col}'")
            return col, df[col].copy()
    
    # Check for columns with BENIGN values (attack type column)
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().unique()
            if 'BENIGN' in sample_vals:
                print(f"Attack type column identified: '{col}'")
                return col, df[col].copy()
    
    raise ValueError("Could not find label column")


def convert_labels_to_binary(y):
    """
    Convert labels to binary (0 = BENIGN, 1 = ATTACK).
    """
    print(f"\nOriginal label values: {y.unique()[:10]}")
    
    if y.dtype == 'object':
        y = y.apply(lambda x: 0 if str(x).upper().strip() == 'BENIGN' else 1)
    else:
        y = (y != 0).astype(int)
    
    benign_count = sum(y == 0)
    attack_count = sum(y == 1)
    print(f"Binary conversion: BENIGN={benign_count:,}, ATTACK={attack_count:,}")
    
    return y


def get_feature_columns(df, label_col):
    """
    Get feature columns (all columns except label and non-numeric).
    """
    exclude_patterns = ['flow', 'id', 'timestamp', 'time', 'ip', 'port']
    
    feature_cols = []
    for col in df.columns:
        if col == label_col:
            continue
        
        col_lower = col.lower()
        should_exclude = False
        for pattern in exclude_patterns:
            if pattern in col_lower:
                should_exclude = True
                break
        
        if not should_exclude:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
    
    print(f"Found {len(feature_cols)} numeric feature columns")
    return feature_cols


def load_and_split(df, test_size=0.15, val_size=0.15):
    """
    Split the dataset into train, validation, and test sets.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "=" * 60)
    print("Splitting Dataset")
    print("=" * 60)
    
    # Identify label column
    label_col, y = identify_label_column(df)
    
    # Convert labels to binary
    y = convert_labels_to_binary(y)
    
    # Get feature columns
    feature_cols = get_feature_columns(df, label_col)
    X = df[feature_cols].copy()
    
    print(f"Feature matrix: {X.shape}")
    
    # Split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    
    # Split: train vs val
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative,
        random_state=RANDOM_STATE, stratify=y_train_val
    )
    
    print(f"\nSplit sizes:")
    print(f"  Training:   {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_preprocessed_data():
    """
    Load preprocessed data from saved CSV files.
    """
    X_train_path = os.path.join(DATA_DIR, "X_train.csv")
    
    if os.path.exists(X_train_path):
        print("Loading preprocessed data from CSV files...")
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
        X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
        X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
        y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()
        
        print(f"Loaded training: {X_train.shape}")
        print(f"Loaded validation: {X_val.shape}")
        print(f"Loaded test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    return None