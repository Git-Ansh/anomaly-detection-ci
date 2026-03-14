"""
Eclipse Bug Dataset Loader (MSR 2013).

Parses JSON files from the MSR 2013 Eclipse bug dataset into a clean
DataFrame suitable for cascade classification.

Data source: https://github.com/ansymo/msr2013-bug_dataset
165,547 Eclipse Bugzilla bug reports.

LEAKAGE PREVENTION:
- Resolution IS the target for S0 (noise gate) -- never used as feature
- For severity/component/priority: use ONLY the FIRST entry (pre-triage)
- All subsequent changes are POST-triage and constitute leakage
- Temporal 80/20 split on opening timestamp
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import Counter


ECLIPSE_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'external' / 'msr2013-bug_dataset' / 'data' / 'v02' / 'eclipse'

# Resolution categories for Stage 0 (noise gate)
# Only clearly wrong reports are noise; DUPLICATE/WONTFIX/WORKSFORME are valid
# bugs that still need triage -- they just have different outcomes.
NOISE_RESOLUTIONS = {'INVALID', 'NOT_ECLIPSE'}
REAL_RESOLUTIONS = {'FIXED', 'DUPLICATE', 'WONTFIX', 'WORKSFORME'}
# Empty resolution = unresolved (filter out for training)

# Top components for Stage 2 (anything else -> 'Other')
TOP_N_COMPONENTS = 30

RANDOM_SEED = 42


def load_eclipse_json(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse Eclipse MSR 2013 JSON files into a single DataFrame.

    Each JSON file has structure:
        {"field_name": {"bug_id": [{"when": ts, "what": value, "who": user}, ...]}}

    We extract INITIAL (first entry) and FINAL (last entry) values for
    each field. Only initial values are safe features (pre-triage).

    Returns:
        DataFrame with one row per bug, columns for all extracted features.
    """
    if data_dir is None:
        data_dir = ECLIPSE_DATA_DIR

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Eclipse data not found at {data_dir}")

    print(f"Loading Eclipse data from {data_dir}...")

    # Load reports.json (core bug info: opening timestamp, reporter, status, resolution)
    reports = _load_json(data_dir / 'reports.json', 'reports')
    print(f"  Reports: {len(reports)} bugs")

    # Load change-history JSON files
    severity = _load_json(data_dir / 'severity.json', 'severity')
    component = _load_json(data_dir / 'component.json', 'component')
    priority = _load_json(data_dir / 'priority.json', 'priority')
    short_desc = _load_json(data_dir / 'short_desc.json', 'short_desc')

    # Build DataFrame
    rows = []
    for bug_id, info in reports.items():
        row = {
            'bug_id': int(bug_id),
            'opening_ts': info.get('opening', 0),
            'reporter_id': info.get('reporter', -1),
            'current_status': info.get('current_status', ''),
            'current_resolution': info.get('current_resolution', ''),
        }

        # Severity: initial (pre-triage) and final
        sev_changes = severity.get(bug_id, [])
        if sev_changes:
            row['initial_severity'] = sev_changes[0].get('what', 'normal') if isinstance(sev_changes[0], dict) else str(sev_changes[0])
            row['final_severity'] = sev_changes[-1].get('what', 'normal') if isinstance(sev_changes[-1], dict) else str(sev_changes[-1])
            row['severity_changes'] = len(sev_changes)
        else:
            row['initial_severity'] = 'normal'
            row['final_severity'] = 'normal'
            row['severity_changes'] = 0

        # Component: initial (pre-triage) and final
        comp_changes = component.get(bug_id, [])
        if comp_changes:
            row['initial_component'] = comp_changes[0].get('what', 'Unknown') if isinstance(comp_changes[0], dict) else str(comp_changes[0])
            row['final_component'] = comp_changes[-1].get('what', 'Unknown') if isinstance(comp_changes[-1], dict) else str(comp_changes[-1])
            row['component_changes'] = len(comp_changes)
        else:
            row['initial_component'] = 'Unknown'
            row['final_component'] = 'Unknown'
            row['component_changes'] = 0

        # Priority: initial
        pri_changes = priority.get(bug_id, [])
        if pri_changes:
            row['initial_priority'] = pri_changes[0].get('what', 'None') if isinstance(pri_changes[0], dict) else str(pri_changes[0])
            row['priority_changes'] = len(pri_changes)
        else:
            row['initial_priority'] = 'None'
            row['priority_changes'] = 0

        # Short description (title text)
        desc_changes = short_desc.get(bug_id, [])
        if desc_changes:
            row['short_desc'] = desc_changes[0].get('what', '') if isinstance(desc_changes[0], dict) else str(desc_changes[0])
        else:
            row['short_desc'] = ''

        rows.append(row)

    df = pd.DataFrame(rows)

    # Convert opening timestamp to datetime
    df['opening_date'] = pd.to_datetime(df['opening_ts'], unit='s', errors='coerce')
    df = df.sort_values('opening_ts').reset_index(drop=True)

    print(f"  Built DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for the cascade stages.

    PRE-TRIAGE features only (safe):
    - Reporter history: number of bugs filed, frequency
    - Initial severity, priority (as reported, not modified by triager)
    - Component size (derived from training data)
    - Text features from short_desc
    - Temporal features (day of week, hour, month)

    POST-TRIAGE (LEAKAGE -- excluded):
    - Final severity/component/priority (modified by triager)
    - Resolution (this IS the target)
    - current_status
    """
    df = df.copy()

    # --- Temporal features ---
    if df['opening_date'].notna().any():
        df['open_hour'] = df['opening_date'].dt.hour
        df['open_dayofweek'] = df['opening_date'].dt.dayofweek
        df['open_month'] = df['opening_date'].dt.month
        df['open_year'] = df['opening_date'].dt.year

    # --- Reporter features (computed from training data) ---
    reporter_counts = df['reporter_id'].value_counts()
    df['reporter_bug_count'] = df['reporter_id'].map(reporter_counts).fillna(0).astype(int)

    # Reporter severity history: what fraction of this reporter's bugs are critical+
    sev_map = {'blocker': 5, 'critical': 4, 'major': 3, 'normal': 2, 'minor': 1, 'trivial': 0, 'enhancement': -1}
    df['severity_numeric'] = df['initial_severity'].map(sev_map).fillna(2)
    reporter_avg_sev = df.groupby('reporter_id')['severity_numeric'].mean()
    df['reporter_avg_severity'] = df['reporter_id'].map(reporter_avg_sev).fillna(2.0)

    # --- Component features ---
    comp_size = df['initial_component'].value_counts()
    df['component_size'] = df['initial_component'].map(comp_size).fillna(0).astype(int)

    # --- Noise label for Stage 0 ---
    # is_noise: INVALID, NOT_ECLIPSE vs FIXED/DUPLICATE/WONTFIX/WORKSFORME
    df['is_noise'] = df['current_resolution'].isin(NOISE_RESOLUTIONS).astype(int)
    # Filter to resolved bugs only (non-empty resolution)
    df['is_resolved'] = df['current_resolution'].str.len() > 0

    # --- Description length ---
    df['desc_length'] = df['short_desc'].str.len().fillna(0).astype(int)
    df['desc_word_count'] = df['short_desc'].str.split().str.len().fillna(0).astype(int)

    # --- Map initial_severity to broader categories ---
    df['is_enhancement'] = (df['initial_severity'] == 'enhancement').astype(int)
    df['is_high_severity'] = df['initial_severity'].isin(['blocker', 'critical']).astype(int)

    print(f"  Engineered features: {len(df.columns)} columns")
    return df


def prepare_eclipse_data(
    data_dir: Optional[Path] = None,
    test_fraction: float = 0.20,
    top_n_components: int = TOP_N_COMPONENTS,
) -> Dict:
    """
    Load Eclipse data, engineer features, and create temporal train/test split.

    Returns dict with:
        - train_df: Training DataFrame
        - test_df: Test DataFrame
        - feature_cols: List of feature column names
        - text_col: Name of text column
        - target_cols: Dict of target columns per stage
        - stats: Dataset statistics
    """
    df = load_eclipse_json(data_dir)
    df = engineer_features(df)

    # Filter to resolved bugs only
    resolved = df[df['is_resolved']].copy()
    print(f"\n  Resolved bugs: {len(resolved)} / {len(df)}")

    # Resolution distribution
    print(f"  Resolution distribution:")
    for res, count in resolved['current_resolution'].value_counts().items():
        print(f"    {res}: {count} ({count/len(resolved):.1%})")

    # --- Temporal train/test split ---
    split_idx = int(len(resolved) * (1 - test_fraction))
    train_df = resolved.iloc[:split_idx].copy()
    test_df = resolved.iloc[split_idx:].copy()

    print(f"\n  Train: {len(train_df)} ({len(train_df)/len(resolved):.1%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(resolved):.1%})")
    print(f"  Train date range: {train_df['opening_date'].min()} to {train_df['opening_date'].max()}")
    print(f"  Test date range:  {test_df['opening_date'].min()} to {test_df['opening_date'].max()}")

    # --- Compute training-set-derived features ---
    # Reporter bug count (training set only, applied to both)
    train_reporter_counts = train_df['reporter_id'].value_counts()
    train_df['reporter_bug_count'] = train_df['reporter_id'].map(train_reporter_counts).fillna(0).astype(int)
    test_df['reporter_bug_count'] = test_df['reporter_id'].map(train_reporter_counts).fillna(0).astype(int)

    # Component size (training set only)
    train_comp_size = train_df['initial_component'].value_counts()
    train_df['component_size'] = train_df['initial_component'].map(train_comp_size).fillna(0).astype(int)
    test_df['component_size'] = test_df['initial_component'].map(train_comp_size).fillna(0).astype(int)

    # Reporter avg severity (training set only)
    train_reporter_sev = train_df.groupby('reporter_id')['severity_numeric'].mean()
    train_df['reporter_avg_severity'] = train_df['reporter_id'].map(train_reporter_sev).fillna(2.0)
    test_df['reporter_avg_severity'] = test_df['reporter_id'].map(train_reporter_sev).fillna(2.0)

    # Top N components (anything else -> 'Other')
    top_components = train_df['initial_component'].value_counts().head(top_n_components).index.tolist()
    train_df['component_top'] = train_df['initial_component'].apply(
        lambda x: x if x in top_components else 'Other'
    )
    test_df['component_top'] = test_df['initial_component'].apply(
        lambda x: x if x in top_components else 'Other'
    )

    # Severity target for Stage 1
    train_df['severity_target'] = train_df['final_severity']
    test_df['severity_target'] = test_df['final_severity']

    # Component target for Stage 2
    train_df['component_target'] = train_df['final_component'].apply(
        lambda x: x if x in top_components else 'Other'
    )
    test_df['component_target'] = test_df['final_component'].apply(
        lambda x: x if x in top_components else 'Other'
    )

    # --- Feature columns (PRE-TRIAGE only) ---
    # REMOVED: severity_numeric (= initial_severity encoded, leaks into severity
    #   target ~95% of the time since it rarely changes).
    # REMOVED: severity_changes, component_changes, priority_changes (all
    #   post-triage counts -- only known after triager modifies the bug).
    numeric_features = [
        'reporter_bug_count', 'component_size', 'reporter_avg_severity',
        'desc_length', 'desc_word_count',
        'is_enhancement', 'is_high_severity',
        'open_hour', 'open_dayofweek', 'open_month',
    ]
    # Only include features that exist
    numeric_features = [f for f in numeric_features if f in train_df.columns]

    categorical_features = ['initial_severity', 'initial_priority', 'initial_component']

    stats = {
        'n_total': len(df),
        'n_resolved': len(resolved),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_components': len(top_components),
        'resolutions': resolved['current_resolution'].value_counts().to_dict(),
        'noise_ratio_train': train_df['is_noise'].mean(),
        'noise_ratio_test': test_df['is_noise'].mean(),
    }

    return {
        'train_df': train_df,
        'test_df': test_df,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'text_col': 'short_desc',
        'top_components': top_components,
        'stats': stats,
    }


def _load_json(filepath: Path, expected_key: str) -> dict:
    """Load a JSON file and extract the inner dict."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Structure is {"key": {data}} -- extract inner dict
    if expected_key in data:
        return data[expected_key]
    # Fallback: return as-is
    return data


if __name__ == '__main__':
    data = prepare_eclipse_data()
    print(f"\nDataset ready:")
    print(f"  Train: {data['stats']['n_train']}")
    print(f"  Test: {data['stats']['n_test']}")
    print(f"  Numeric features: {data['numeric_features']}")
    print(f"  Components (top {len(data['top_components'])}): {data['top_components'][:5]}...")
    print(f"  Noise ratio train: {data['stats']['noise_ratio_train']:.1%}")
    print(f"  Noise ratio test: {data['stats']['noise_ratio_test']:.1%}")
