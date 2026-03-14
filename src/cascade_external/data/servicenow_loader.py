"""
UCI ServiceNow ITSM Dataset Loader.

Parses the Incident Management Process Enriched Event Log (UCI dataset 498)
into per-incident records suitable for cascade classification.

Source: https://archive.ics.uci.edu/dataset/498
141,712 events across 24,918 incidents, 36 attributes.

The event log has multiple rows per incident (state transitions).
We aggregate to get one row per incident with initial and final states.

LEAKAGE PREVENTION:
- Features: Only initial incident attributes (available at creation time)
- Targets: Final priority, assignment_group (determined during triage)
- Temporal split on opened_at timestamp
- reassignment_count, reopen_count, sys_mod_count are POST-TRIAGE
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

SERVICENOW_DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'external' / 'servicenow'
RANDOM_SEED = 42


def load_servicenow_events(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw event log CSV.

    Returns DataFrame with all 141K events.
    """
    if data_dir is None:
        data_dir = SERVICENOW_DATA_DIR

    csv_path = data_dir / 'incident_event_log.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ServiceNow data not found at {csv_path}. "
            "Download from https://archive.ics.uci.edu/dataset/498"
        )

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events for {df['number'].nunique()} incidents")
    return df


def aggregate_to_incidents(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate event log to one row per incident.

    For each incident, extract:
    - Initial state attributes (first event = creation)
    - Final state attributes (last event = resolution)
    - Process metrics (for reporting only, NOT features)
    """
    # Sort by incident + sys_mod_count to ensure chronological order
    events = events_df.sort_values(['number', 'sys_mod_count']).copy()

    # Get first event (creation) per incident
    first = events.groupby('number').first().reset_index()
    # Get last event (final state) per incident
    last = events.groupby('number').last().reset_index()

    incidents = pd.DataFrame()
    incidents['number'] = first['number']

    # --- PRE-TRIAGE features (from first/initial event) ---
    incidents['opened_at'] = pd.to_datetime(first['opened_at'], format='mixed', dayfirst=True)
    incidents['caller_id'] = first['caller_id']
    incidents['opened_by'] = first['opened_by']
    incidents['contact_type'] = first['contact_type']
    incidents['location'] = first['location']
    incidents['category'] = first['category']
    incidents['subcategory'] = first['subcategory']
    incidents['u_symptom'] = first['u_symptom']
    incidents['cmdb_ci'] = first['cmdb_ci']
    incidents['impact'] = first['impact']
    incidents['urgency'] = first['urgency']
    incidents['made_sla'] = first['made_sla']

    # Initial priority (computed from impact x urgency at creation)
    incidents['initial_priority'] = first['priority']

    # --- POST-TRIAGE targets (from last event) ---
    incidents['final_priority'] = last['priority']
    incidents['final_state'] = last['incident_state']
    incidents['assignment_group'] = last['assignment_group']
    incidents['assigned_to'] = last['assigned_to']
    incidents['closed_code'] = last['closed_code']
    incidents['resolved_by'] = last['resolved_by']
    incidents['resolved_at'] = last['resolved_at']
    incidents['closed_at'] = last['closed_at']

    # --- Process metrics (for reporting, NOT features) ---
    incidents['reassignment_count'] = last['reassignment_count'].astype(float)
    incidents['reopen_count'] = last['reopen_count'].astype(float)
    incidents['n_events'] = events.groupby('number').size().values

    print(f"Aggregated to {len(incidents)} incidents")
    return incidents


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer pre-triage features for cascade classification.

    All features are available at incident creation time.
    """
    df = df.copy()

    # Parse priority levels (e.g. "1 - Critical" -> 1)
    df['priority_num'] = df['initial_priority'].str.extract(r'(\d+)').astype(float)
    df['impact_num'] = df['impact'].str.extract(r'(\d+)').astype(float)
    df['urgency_num'] = df['urgency'].str.extract(r'(\d+)').astype(float)

    # Temporal features
    if df['opened_at'].notna().any():
        df['open_hour'] = df['opened_at'].dt.hour
        df['open_dayofweek'] = df['opened_at'].dt.dayofweek
        df['open_month'] = df['opened_at'].dt.month
        df['is_weekend'] = (df['open_dayofweek'] >= 5).astype(int)
        df['is_business_hours'] = ((df['open_hour'] >= 8) & (df['open_hour'] <= 17)).astype(int)

    # SLA flag
    df['made_sla_num'] = (df['made_sla'] == 'true').astype(int) if df['made_sla'].dtype == object else df['made_sla'].astype(int)

    # Contact type encoding
    df['is_phone'] = (df['contact_type'] == 'Phone').astype(int)
    df['is_email'] = (df['contact_type'] == 'Email').astype(int)

    # Caller frequency (computed from training data, recomputed in prepare)
    caller_counts = df['caller_id'].value_counts()
    df['caller_incident_count'] = df['caller_id'].map(caller_counts).fillna(0).astype(int)

    # Category-level statistics
    cat_counts = df['category'].value_counts()
    df['category_size'] = df['category'].map(cat_counts).fillna(0).astype(int)

    # Final priority target (numeric)
    df['final_priority_num'] = df['final_priority'].str.extract(r'(\d+)').astype(float)

    print(f"  Engineered features: {len(df.columns)} columns")
    return df


def prepare_servicenow_data(
    data_dir: Optional[Path] = None,
    test_fraction: float = 0.20,
    top_n_groups: int = 20,
) -> Dict:
    """
    Load, aggregate, engineer features, and split ServiceNow data.

    Args:
        data_dir: Path to data directory
        test_fraction: Fraction for test set
        top_n_groups: Number of top assignment groups to keep

    Returns:
        Dict with train_df, test_df, feature columns, targets, stats
    """
    events = load_servicenow_events(data_dir)
    incidents = aggregate_to_incidents(events)
    incidents = engineer_features(incidents)

    # Filter to resolved incidents (have a closed_at or resolved_at)
    resolved = incidents[
        incidents['final_state'].isin(['Closed', 'Resolved'])
    ].copy()
    print(f"\n  Resolved incidents: {len(resolved)} / {len(incidents)}")

    # Filter out '?' placeholder values in key fields
    resolved = resolved[resolved['category'] != '?'].copy()
    resolved = resolved[resolved['assignment_group'] != '?'].copy()

    # --- Temporal train/test split ---
    resolved = resolved.sort_values('opened_at').reset_index(drop=True)
    split_idx = int(len(resolved) * (1 - test_fraction))
    train_df = resolved.iloc[:split_idx].copy()
    test_df = resolved.iloc[split_idx:].copy()

    print(f"\n  Train: {len(train_df)} ({len(train_df)/len(resolved):.1%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(resolved):.1%})")

    # --- Recompute training-set-derived features ---
    train_caller_counts = train_df['caller_id'].value_counts()
    train_df['caller_incident_count'] = train_df['caller_id'].map(train_caller_counts).fillna(0).astype(int)
    test_df['caller_incident_count'] = test_df['caller_id'].map(train_caller_counts).fillna(0).astype(int)

    train_cat_size = train_df['category'].value_counts()
    train_df['category_size'] = train_df['category'].map(train_cat_size).fillna(0).astype(int)
    test_df['category_size'] = test_df['category'].map(train_cat_size).fillna(0).astype(int)

    # --- Top assignment groups (anything else -> 'Other') ---
    top_groups = train_df['assignment_group'].value_counts().head(top_n_groups).index.tolist()
    train_df['group_target'] = train_df['assignment_group'].apply(
        lambda x: x if x in top_groups else 'Other'
    )
    test_df['group_target'] = test_df['assignment_group'].apply(
        lambda x: x if x in top_groups else 'Other'
    )

    # --- Feature columns (PRE-TRIAGE only) ---
    numeric_features = [
        'priority_num', 'impact_num', 'urgency_num',
        'caller_incident_count', 'category_size',
        'made_sla_num', 'is_phone', 'is_email',
        'open_hour', 'open_dayofweek', 'open_month',
        'is_weekend', 'is_business_hours',
    ]
    numeric_features = [f for f in numeric_features if f in train_df.columns]

    categorical_features = [
        'contact_type', 'category', 'subcategory', 'location',
        'u_symptom', 'cmdb_ci',
    ]

    # Priority distribution
    print(f"\n  Priority distribution (train):")
    for p, count in train_df['initial_priority'].value_counts().items():
        print(f"    {p}: {count} ({count/len(train_df):.1%})")

    # Assignment group distribution
    print(f"\n  Assignment groups: {train_df['assignment_group'].nunique()} total, "
          f"top {top_n_groups} cover "
          f"{(train_df['group_target'] != 'Other').mean():.1%}")

    stats = {
        'n_total': len(incidents),
        'n_resolved': len(resolved),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_groups': len(top_groups),
        'n_categories': train_df['category'].nunique(),
        'priority_dist': train_df['initial_priority'].value_counts().to_dict(),
    }

    return {
        'train_df': train_df,
        'test_df': test_df,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'top_groups': top_groups,
        'stats': stats,
    }


if __name__ == '__main__':
    data = prepare_servicenow_data()
    print(f"\nDataset ready:")
    print(f"  Train: {data['stats']['n_train']}")
    print(f"  Test: {data['stats']['n_test']}")
    print(f"  Numeric features: {data['numeric_features']}")
    print(f"  Categories: {data['stats']['n_categories']}")
    print(f"  Assignment groups: {data['stats']['n_groups']}")
