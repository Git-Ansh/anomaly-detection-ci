"""
Eclipse Zenodo 2024 Dataset Loader.

Parses the Eclipse Issue Report Dataset (Zenodo record 15348468) into a
DataFrame suitable for cascade classification.

Source: https://zenodo.org/records/15348468
~304K bugs across 9 Eclipse projects with full descriptions, comments,
and activity history.

Download: wget "https://zenodo.org/api/records/15348468/files/Eclipse_dataset.zip/content" -O Eclipse_dataset.zip

The ZIP contains per-project directories:
  P_CDT/CDT_dataset_issues.csv
  P_Platform/Platform_dataset_issues.csv
  P_JDT/JDT_dataset_issues.csv
  etc.

CSV Columns:
  Issue URL, ID, Alias, Classification, Component, Product, Version,
  Platform, Op sys, Status, Resolution, Depends on, Dupe of, Blocks,
  Groups, Flags, Severity, Priority, Deadline, Target Milestone,
  Creator, Creator Detail, Creation time, Assigned to, Assigned to detail,
  CC, CC detail, Is CC accessible, Is confirmed, Is open,
  Is creator accessible, Summary, Description, URL, Whiteboard, Keywords,
  See also, Last change time, QA contact, History/Activity Log, Comments,
  Attachments

LEAKAGE PREVENTION:
- Features: Summary, Description (original reporter text), Creator, Platform,
  Op sys, Severity (initial), Priority (initial), Creation time
- Targets: Component (final), Resolution, Severity (if changed)
- POST-TRIAGE: Resolution, Status, Is confirmed, Keywords (often triager-applied),
  History/Activity Log, Comments, Last change time
- Temporal 80/20 split on Creation time
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

ECLIPSE_ZENODO_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'external' / 'eclipse_zenodo'
RANDOM_SEED = 42

# Projects in the dataset
# ZIP structure: Eclipse/P_{Project} or Eclipse/C_{Project}
PROJECTS = [
    'Platform', 'JDT', 'CDT', 'BIRT', 'PDE',
    'Equinox', 'Mylyn', 'Papyrus', 'TPTP',
]
# Note: BIRT -> Eclipse/C_BIRT/BIRT_dataset_issues.csv
# Mylyn -> Eclipse/C_mylyn/MYLYN_dataset_issues.csv

# Resolution categories for noise gate
NOISE_RESOLUTIONS = {'INVALID', 'DUPLICATE', 'WONTFIX', 'WORKSFORME', 'NOT_ECLIPSE'}
VALID_RESOLUTIONS = {'FIXED', 'VERIFIED'}

# Columns we actually need (avoids loading huge Comments/History/Attachments)
NEEDED_COLUMNS = [
    'Summary', 'Description', 'Severity', 'Priority', 'Platform', 'Op sys',
    'Component', 'Product', 'Resolution', 'Status', 'Creator',
    'Creation time', 'ID',
]


def load_eclipse_zenodo(
    data_dir: Optional[Path] = None,
    projects: Optional[List[str]] = None,
    max_per_project: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Eclipse Zenodo 2024 dataset from ZIP or extracted CSVs.

    Args:
        data_dir: Directory containing Eclipse_dataset.zip or extracted CSVs
        projects: List of projects to load (default: all)
        max_per_project: Max bugs per project (for testing)

    Returns:
        DataFrame with all bugs
    """
    if data_dir is None:
        data_dir = ECLIPSE_ZENODO_DIR

    data_dir = Path(data_dir)
    if projects is None:
        projects = PROJECTS

    all_dfs = []

    # Prefer extracted CSVs (ZIP uses Deflate64 which Python zipfile doesn't support)
    def _try_extracted_csvs():
        """Try loading from extracted CSV files on disk."""
        loaded = []
        for project in projects:
            proj_upper = project.upper()
            candidates = [
                data_dir / 'Eclipse' / f'P_{project}' / f'{project}_dataset_issues.csv',
                data_dir / 'Eclipse' / f'C_{project}' / f'{proj_upper}_dataset_issues.csv',
                data_dir / 'Eclipse' / f'C_{project.lower()}' / f'{proj_upper}_dataset_issues.csv',
                data_dir / f'P_{project}' / f'{project}_dataset_issues.csv',
                data_dir / f'{project}_dataset_issues.csv',
            ]
            for csv_path in candidates:
                if csv_path.exists():
                    # Only read needed columns to save memory
                    # (Comments/History/Attachments are huge and unused)
                    try:
                        df = pd.read_csv(csv_path, nrows=max_per_project,
                                         usecols=NEEDED_COLUMNS,
                                         on_bad_lines='skip', low_memory=False)
                    except ValueError:
                        # Some CSVs may have slightly different column names
                        df = pd.read_csv(csv_path, nrows=max_per_project,
                                         on_bad_lines='skip', low_memory=False)
                        # Keep only needed columns that exist
                        keep = [c for c in NEEDED_COLUMNS if c in df.columns]
                        df = df[keep]
                    df['project'] = project
                    loaded.append(df)
                    print(f"  {project}: {len(df)} bugs")
                    break
        return loaded

    # Try extracted CSVs first
    print("Loading from extracted CSVs...")
    all_dfs = _try_extracted_csvs()

    # Fallback to ZIP if no CSVs found
    if not all_dfs:
        zip_path = data_dir / 'Eclipse_dataset.zip'
        if zip_path.exists():
            print(f"Trying ZIP: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for project in projects:
                        proj_upper = project.upper()
                        candidates = [
                            f'Eclipse/P_{project}/{project}_dataset_issues.csv',
                            f'Eclipse/C_{project}/{proj_upper}_dataset_issues.csv',
                            f'Eclipse/C_{project.lower()}/{proj_upper}_dataset_issues.csv',
                            f'P_{project}/{project}_dataset_issues.csv',
                            f'{project}_dataset_issues.csv',
                        ]
                        found = False
                        for name in candidates:
                            if name in z.namelist():
                                with z.open(name) as f:
                                    df = pd.read_csv(f, nrows=max_per_project,
                                                     on_bad_lines='skip',
                                                     low_memory=False)
                                    df['project'] = project
                                    all_dfs.append(df)
                                    print(f"  {project}: {len(df)} bugs")
                                found = True
                                break
                        if not found:
                            for name in z.namelist():
                                if name.endswith('.csv') and project.lower() in name.lower():
                                    with z.open(name) as f:
                                        df = pd.read_csv(f, nrows=max_per_project,
                                                         on_bad_lines='skip',
                                                         low_memory=False)
                                        df['project'] = project
                                        all_dfs.append(df)
                                        print(f"  {project}: {len(df)} bugs (from {name})")
                                    break
            except NotImplementedError:
                print("ZIP uses Deflate64 compression. Extract CSVs first:")
                print(f"  python3 -c 'import inflate64; ...' or use system unzip")

    if not all_dfs:
        raise FileNotFoundError(
            f"Eclipse Zenodo data not found. Download from:\n"
            f"  wget 'https://zenodo.org/api/records/15348468/files/Eclipse_dataset.zip/content' "
            f"-O {zip_path}"
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(combined)} bugs from {len(all_dfs)} projects")
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer pre-triage features from Eclipse Zenodo data.

    PRE-TRIAGE (safe):
    - Summary and Description text
    - Creator metadata
    - Initial Severity, Priority
    - Platform, OS
    - Creation time

    POST-TRIAGE (LEAKAGE):
    - Resolution, Status (these ARE targets)
    - Is confirmed (triager decision)
    - Keywords (often triager-applied)
    - History/Activity Log, Comments
    """
    df = df.copy()

    # Parse creation time
    df['creation_time'] = pd.to_datetime(df['Creation time'], errors='coerce')

    # Temporal features
    if df['creation_time'].notna().any():
        df['open_hour'] = df['creation_time'].dt.hour
        df['open_dayofweek'] = df['creation_time'].dt.dayofweek
        df['open_month'] = df['creation_time'].dt.month
        df['open_year'] = df['creation_time'].dt.year

    # Text features
    df['summary_length'] = df['Summary'].fillna('').str.len()
    df['summary_word_count'] = df['Summary'].fillna('').str.split().str.len().fillna(0).astype(int)
    df['desc_length'] = df['Description'].fillna('').str.len()
    df['desc_word_count'] = df['Description'].fillna('').str.split().str.len().fillna(0).astype(int)
    df['has_description'] = (df['desc_length'] > 10).astype(int)

    # Severity encoding (PRE-TRIAGE: reporter-selected)
    sev_map = {'blocker': 5, 'critical': 4, 'major': 3, 'normal': 2,
               'minor': 1, 'trivial': 0, 'enhancement': -1}
    df['severity_numeric'] = df['Severity'].str.lower().map(sev_map).fillna(2)
    df['is_enhancement'] = (df['Severity'].str.lower() == 'enhancement').astype(int)
    df['is_high_severity'] = df['Severity'].str.lower().isin(['blocker', 'critical']).astype(int)

    # Priority encoding
    pri_map = {'P1': 4, 'P2': 3, 'P3': 2, 'P4': 1, 'P5': 0}
    df['priority_numeric'] = df['Priority'].map(pri_map).fillna(2)

    # Creator frequency
    creator_counts = df['Creator'].value_counts()
    df['creator_bug_count'] = df['Creator'].map(creator_counts).fillna(0).astype(int)

    # Component size
    comp_counts = df['Component'].value_counts()
    df['component_size'] = df['Component'].map(comp_counts).fillna(0).astype(int)

    # Noise label for S0
    df['is_noise'] = df['Resolution'].fillna('').str.upper().isin(NOISE_RESOLUTIONS).astype(int)

    # Resolved filter
    df['is_resolved'] = df['Resolution'].fillna('').str.len() > 0

    return df


def prepare_eclipse_zenodo_data(
    data_dir: Optional[Path] = None,
    projects: Optional[List[str]] = None,
    test_fraction: float = 0.20,
    top_n_components: int = 30,
    max_per_project: Optional[int] = None,
) -> Dict:
    """
    Load, engineer features, and split Eclipse Zenodo data.

    Returns dict with train_df, test_df, feature columns, stats.
    """
    df = load_eclipse_zenodo(data_dir, projects, max_per_project)
    df = engineer_features(df)

    # Filter to resolved
    resolved = df[df['is_resolved']].copy()
    print(f"\nResolved: {len(resolved)} / {len(df)}")

    # Temporal split
    resolved = resolved.sort_values('creation_time').reset_index(drop=True)
    split_idx = int(len(resolved) * (1 - test_fraction))
    train_df = resolved.iloc[:split_idx].copy()
    test_df = resolved.iloc[split_idx:].copy()

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Recompute training-only features
    train_creator_counts = train_df['Creator'].value_counts()
    train_df['creator_bug_count'] = train_df['Creator'].map(train_creator_counts).fillna(0).astype(int)
    test_df['creator_bug_count'] = test_df['Creator'].map(train_creator_counts).fillna(0).astype(int)

    train_comp_size = train_df['Component'].value_counts()
    train_df['component_size'] = train_df['Component'].map(train_comp_size).fillna(0).astype(int)
    test_df['component_size'] = test_df['Component'].map(train_comp_size).fillna(0).astype(int)

    # Top components
    top_components = train_df['Component'].value_counts().head(top_n_components).index.tolist()
    train_df['component_target'] = train_df['Component'].apply(
        lambda x: x if x in top_components else 'Other')
    test_df['component_target'] = test_df['Component'].apply(
        lambda x: x if x in top_components else 'Other')

    # Feature columns (PRE-TRIAGE only)
    # NOTE: severity_numeric is the INITIAL (reporter-selected) severity,
    # which is pre-triage in Zenodo data (unlike MSR 2013 where it was ambiguous).
    # However, for honest S1 Severity prediction, we should still exclude it
    # if Severity IS the target.
    numeric_features = [
        'creator_bug_count', 'component_size',
        'summary_length', 'summary_word_count',
        'desc_length', 'desc_word_count', 'has_description',
        'severity_numeric', 'priority_numeric',
        'is_enhancement', 'is_high_severity',
        'open_hour', 'open_dayofweek', 'open_month',
    ]
    numeric_features = [f for f in numeric_features if f in train_df.columns]

    categorical_features = ['Severity', 'Priority', 'Platform', 'Op sys', 'project']
    categorical_features = [f for f in categorical_features if f in train_df.columns]

    stats = {
        'n_total': len(df),
        'n_resolved': len(resolved),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_projects': len(set(df['project'])),
        'n_components': len(top_components),
        'noise_ratio_train': train_df['is_noise'].mean(),
    }

    return {
        'train_df': train_df,
        'test_df': test_df,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'text_col': 'Summary',
        'description_col': 'Description',
        'top_components': top_components,
        'stats': stats,
    }


if __name__ == '__main__':
    data = prepare_eclipse_zenodo_data()
    print(f"\nDataset ready:")
    print(f"  Train: {data['stats']['n_train']}")
    print(f"  Test: {data['stats']['n_test']}")
    print(f"  Components: {data['stats']['n_components']}")
    print(f"  Noise ratio: {data['stats']['noise_ratio_train']:.1%}")
