"""
PROMISE JM1 (NASA Defect Prediction) Data Loader.

JM1 is a NASA spacecraft instrumentation software dataset with 21
McCabe and Halstead code metrics + binary defect label.

Source: PROMISE Software Engineering Repository
Size: ~10,878 modules (after cleaning), 21 features + defect label

LEAKAGE PREVENTION:
- All 21 features are code metrics computed before defect discovery
- No temporal dimension -- random split is standard
- No leakage risk (lowest of all datasets)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

RANDOM_SEED = 42

# Standard JM1 feature names (McCabe + Halstead metrics)
JM1_FEATURES = [
    'loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
    'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
    'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount',
]


def load_jm1_data(
    filepath: Optional[Path] = None,
    test_fraction: float = 0.20,
) -> Dict:
    """
    Load JM1 dataset and prepare for cascade.

    Tries multiple file locations and formats (CSV, ARFF).

    Returns dict with:
        - train_df, test_df: DataFrames
        - feature_cols: list of feature columns
        - stats: dataset statistics
    """
    if filepath is None:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent.parent / 'data' / 'external' / 'jm1.csv',
            Path(__file__).parent.parent.parent.parent / 'data' / 'external' / 'jm1.arff',
            Path(__file__).parent.parent.parent.parent / 'data' / 'jm1.csv',
        ]
        for p in candidates:
            if p.exists():
                filepath = p
                break

    if filepath is None or not filepath.exists():
        raise FileNotFoundError("JM1 dataset not found. Download from PROMISE repository.")

    filepath = Path(filepath)
    print(f"Loading JM1 data from {filepath}...")

    if filepath.suffix == '.arff':
        df = _load_arff(filepath)
    else:
        df = pd.read_csv(filepath)

    print(f"  Raw shape: {df.shape}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the defect/class column
    defect_col = None
    for candidate in ['defects', 'defect', 'class', 'bug', 'faulty']:
        if candidate in df.columns:
            defect_col = candidate
            break

    if defect_col is None:
        # Try last column
        defect_col = df.columns[-1]
        print(f"  Using last column as target: '{defect_col}'")

    # Normalize defect labels to 0/1
    unique_vals = df[defect_col].unique()
    print(f"  Defect column '{defect_col}' values: {unique_vals}")

    if set(unique_vals) <= {0, 1}:
        df['defective'] = df[defect_col].astype(int)
    elif set(unique_vals) <= {True, False, 'true', 'false', 'TRUE', 'FALSE'}:
        df['defective'] = df[defect_col].map(
            {True: 1, False: 0, 'true': 1, 'false': 0, 'TRUE': 1, 'FALSE': 0}
        ).fillna(0).astype(int)
    elif set(unique_vals) <= {-1, 1}:
        df['defective'] = (df[defect_col] == 1).astype(int)
    elif set(unique_vals) <= {'Y', 'N', 'y', 'n'}:
        df['defective'] = df[defect_col].map({'Y': 1, 'N': 0, 'y': 1, 'n': 0}).fillna(0).astype(int)
    else:
        df['defective'] = df[defect_col].astype(int)

    # Find feature columns (all numeric except target)
    feature_cols = [c for c in df.columns
                    if c not in [defect_col, 'defective']
                    and pd.api.types.is_numeric_dtype(df[c])]

    # Drop rows with missing values in features
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print(f"  Clean shape: {df.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Defective: {df['defective'].sum()} / {len(df)} "
          f"({df['defective'].mean():.1%})")

    # Random train/test split (no temporal dimension in JM1)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(df))
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[indices[:split_idx]].copy()
    test_df = df.iloc[indices[split_idx:]].copy()

    stats = {
        'n_total': len(df),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': len(feature_cols),
        'defect_rate_train': train_df['defective'].mean(),
        'defect_rate_test': test_df['defective'].mean(),
    }

    print(f"  Train: {len(train_df)} (defect rate: {stats['defect_rate_train']:.1%})")
    print(f"  Test:  {len(test_df)} (defect rate: {stats['defect_rate_test']:.1%})")

    return {
        'train_df': train_df,
        'test_df': test_df,
        'feature_cols': feature_cols,
        'stats': stats,
    }


def _load_arff(filepath: Path) -> pd.DataFrame:
    """Parse ARFF format into DataFrame."""
    columns = []
    data_started = False
    rows = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                col_name = parts[1].strip("'\"")
                columns.append(col_name)
            elif line.upper().startswith('@DATA'):
                data_started = True
            elif data_started:
                values = line.split(',')
                rows.append(values)

    df = pd.DataFrame(rows, columns=columns)
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
    return df


if __name__ == '__main__':
    data = load_jm1_data()
    print(f"\nJM1 Dataset ready:")
    print(f"  Features: {data['feature_cols'][:5]}...")
    print(f"  Train: {data['stats']['n_train']}")
    print(f"  Test: {data['stats']['n_test']}")
