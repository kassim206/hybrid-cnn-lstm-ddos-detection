"""
Data preprocessing utilities with proper handling of non-numeric columns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.config import RANDOM_STATE


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    
    if missing_cols:
        print(f"Found missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
    
    return df


def remove_infinite_values(df):
    """Replace infinite values with NaN and fill with appropriate values."""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)
    
    return df


def identify_label_column(df):
    """
    Identify the label column before removing non-numeric columns.
    Returns the label column name and a copy of the label values.
    """
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'label' in col_lower or 'attack' in col_lower:
            print(f"Label column identified: '{col}'")
            return col, df[col].copy()
    
    # Check for columns with BENIGN values
    for col in df.columns:
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().unique()
            if 'BENIGN' in sample_vals or 'benign' in str(sample_vals).lower():
                print(f"Label column identified (by values): '{col}'")
                return col, df[col].copy()
    
    raise ValueError("Could not find label column")


def remove_non_numeric_columns(df, label_col):
    """
    Remove columns that contain non-numeric data (IP addresses, strings, etc.).
    Preserves the label column.
    """
    # Columns to explicitly drop (identifiers, IPs, timestamps)
    columns_to_drop = []
    
    for col in df.columns:
        if col == label_col:
            continue  # Preserve the label column
        
        col_lower = col.lower()
        # Drop identifier columns
        if any(keyword in col_lower for keyword in ['flow', 'id', 'timestamp', 'time', 'ip', 'port']):
            columns_to_drop.append(col)
            print(f"Dropping identifier column: {col}")
        # Check if column contains non-numeric data
        elif df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"Converted to numeric: {col}")
            except (ValueError, TypeError):
                # If conversion fails, drop the column
                columns_to_drop.append(col)
                print(f"Dropping non-numeric column: {col}")
    
    # Drop all identified columns
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Removed {len(columns_to_drop)} non-numeric columns")
    
    return df


def convert_labels_to_binary(y, label_col_name):
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
    print(f"Binary conversion complete:")
    print(f"  BENIGN (0): {benign_count:,}")
    print(f"  ATTACK (1): {attack_count:,}")
    
    return y


def normalize_features(X_train, X_val, X_test, method='standard'):
    """
    Normalize features using StandardScaler or MinMaxScaler.
    Only works with numeric DataFrames.
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    
    # Ensure all columns are numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame with original column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Features normalized using {method} scaling")
    print(f"Final feature count: {X_train.shape[1]}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def prepare_sequences_for_lstm(X, sequence_length):
    """Prepare data for LSTM by creating sequences."""
    n_samples, n_features = X.shape
    n_sequences = n_samples - sequence_length + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Sequence length {sequence_length} exceeds number of samples {n_samples}")
    
    sequences = np.zeros((n_sequences, sequence_length, n_features))
    
    for i in range(n_sequences):
        sequences[i] = X.iloc[i:i + sequence_length].values
    
    return sequences