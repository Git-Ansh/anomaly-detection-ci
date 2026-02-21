#!/usr/bin/env python3
"""
Generate all figures for ICSME 2025 paper.

This script creates publication-ready figures from result CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set publication-quality style
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
sns.set_style("whitegrid")
sns.set_palette("colorblind")

# Paths
BASE_DIR = Path(__file__).parent.parent / "Doc_ansh" / "Doc_ansh"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

print("="*60)
print("GENERATING ALL FIGURES FOR ICSME 2025 PAPER")
print("="*60)

# ============================================
# Figure 1: Tolerance Sensitivity Analysis
# ============================================
print("\n[1/5] Generating tolerance sensitivity figure...")

try:
    df_tol = pd.read_csv(BASE_DIR / "phase_4" / "outputs_real_world" / "reports" / "E2_tolerance_sensitivity.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: F1 vs Tolerance
    ax1.plot(df_tol['tolerance'], df_tol['f1'], 'o-', linewidth=2, markersize=8, label='F1 Score')
    ax1.axhline(y=0.423, color='red', linestyle='--', linewidth=1.5, label='ML Baseline (0.423)')
    ax1.set_xlabel('Tolerance Window (τ)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax1.set_title('(a) F1 Score vs. Tolerance Window', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_xticks(df_tol['tolerance'])
    ax1.set_ylim([0, 0.6])

    # Plot 2: Precision vs Recall Trade-off
    ax2.plot(df_tol['recall'], df_tol['precision'], 'o-', linewidth=2, markersize=8, color='green')
    for i, tau in enumerate(df_tol['tolerance']):
        ax2.annotate(f'τ={int(tau)}',
                    (df_tol['recall'].iloc[i], df_tol['precision'].iloc[i]),
                    textcoords="offset points", xytext=(5,5), fontsize=9)
    ax2.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1.05])
    ax2.set_ylim([0, 0.5])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tolerance_sensitivity.pdf")
    plt.savefig(FIGURES_DIR / "tolerance_sensitivity.png", dpi=300)
    plt.close()

    print("  [OK] Created: tolerance_sensitivity.pdf")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================
# Figure 2: ML Model Comparison
# ============================================
print("\n[2/5] Generating ML model comparison figure...")

try:
    df_ml = pd.read_csv(BASE_DIR / "phase_7" / "outputs_fixed" / "reports" / "E1_model_comparison.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    models = df_ml['model'].values
    f1_scores = df_ml['f1_score'].values
    mcc_scores = df_ml['mcc'].values

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='steelblue')
    bars2 = ax.bar(x + width/2, mcc_scores, width, label='MCC', color='coral')

    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Supervised ML Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 0.5])

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ml_comparison.pdf")
    plt.savefig(FIGURES_DIR / "ml_comparison.png", dpi=300)
    plt.close()

    print("  [OK] Created: ml_comparison.pdf")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================
# Figure 3: Feature Ablation Study
# ============================================
print("\n[3/5] Generating ablation study figure...")

try:
    df_ablation = pd.read_csv(BASE_DIR / "phase_7" / "outputs_fixed" / "reports" / "E3_ablation.csv")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by F1 score
    df_ablation = df_ablation.sort_values('f1_score', ascending=True)

    colors = ['red' if 'without' in fg or fg == 'all_features' else 'green'
              for fg in df_ablation['feature_group']]

    bars = ax.barh(df_ablation['feature_group'], df_ablation['f1_score'], color=colors, alpha=0.7)

    # Add baseline line
    baseline = df_ablation[df_ablation['feature_group'] == 'all_features']['f1_score'].values[0]
    ax.axvline(x=baseline, color='black', linestyle='--', linewidth=1.5,
               label=f'Baseline (all features): {baseline:.3f}')

    ax.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('Feature Configuration', fontsize=11, fontweight='bold')
    ax.set_title('Feature Ablation Study Results', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (fg, f1) in enumerate(zip(df_ablation['feature_group'], df_ablation['f1_score'])):
        ax.text(f1 + 0.01, i, f'{f1:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ablation_study.pdf")
    plt.savefig(FIGURES_DIR / "ablation_study.png", dpi=300)
    plt.close()

    print("  [OK] Created: ablation_study.pdf")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================
# Figure 4: CPD Algorithm Comparison
# ============================================
print("\n[4/5] Generating CPD comparison figure...")

try:
    # Create comparison data from paper values
    cpd_data = {
        'Algorithm': ['PELT-RBF', 'PELT-L2', 'Binary Seg', 'BOCPD', 'Sliding Window'],
        'Precision': [0.150, 0.070, 0.587, 0.295, 0.766],
        'Recall': [0.139, 0.070, 0.722, 0.940, 1.000],
        'F1': [0.084, 0.070, 0.582, 0.427, 0.858]
    }
    df_cpd = pd.DataFrame(cpd_data)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df_cpd))
    width = 0.25

    bars1 = ax.bar(x - width, df_cpd['Precision'], width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, df_cpd['Recall'], width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, df_cpd['F1'], width, label='F1 Score', color='green')

    ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('CPD Algorithm Validation (Synthetic Data)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_cpd['Algorithm'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cpd_comparison.pdf")
    plt.savefig(FIGURES_DIR / "cpd_comparison.png", dpi=300)
    plt.close()

    print("  [OK] Created: cpd_comparison.pdf")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================
# Figure 5: Paradigm Comparison
# ============================================
print("\n[5/5] Generating paradigm comparison figure...")

try:
    # Create paradigm comparison
    paradigm_data = {
        'Paradigm': ['Supervised ML\n(Stacking)', 'Change-Point\n(Binary Seg, τ=10)', 'Forecasting\n(Naive Mean)'],
        'F1 Score': [0.423, 0.544, 0.740],
        'Category': ['Supervised', 'Unsupervised', 'Time Series']
    }
    df_paradigm = pd.DataFrame(paradigm_data)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {'Supervised': 'steelblue', 'Unsupervised': 'green', 'Time Series': 'coral'}
    bar_colors = [colors[cat] for cat in df_paradigm['Category']]

    bars = ax.bar(df_paradigm['Paradigm'], df_paradigm['F1 Score'],
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax.set_xlabel('Detection Paradigm', fontsize=11, fontweight='bold')
    ax.set_title('Performance Comparison Across Paradigms', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 0.8])

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['Supervised'], label='Supervised ML'),
                      Patch(facecolor=colors['Unsupervised'], label='Unsupervised CPD'),
                      Patch(facecolor=colors['Time Series'], label='Forecasting')]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "paradigm_comparison.pdf")
    plt.savefig(FIGURES_DIR / "paradigm_comparison.png", dpi=300)
    plt.close()

    print("  [OK] Created: paradigm_comparison.pdf")
except Exception as e:
    print(f"  [ERROR] {e}")

# ============================================
# Summary
# ============================================
print("\n" + "="*60)
print("FIGURE GENERATION COMPLETE")
print("="*60)
print(f"\nAll figures saved to: {FIGURES_DIR.absolute()}")
print("\nGenerated files:")
for fig in sorted(FIGURES_DIR.glob("*.pdf")):
    print(f"  [OK] {fig.name}")
print("\nFigures are ready for paper inclusion!")
print("="*60)
