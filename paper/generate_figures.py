"""
Generate figures for ICPE 2026 Data Challenge Paper.
Figures are saved to paper/figures/ directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# ============================================================
# Figure 1: Paradigm Comparison (Bar Chart) - REAL-WORLD RESULTS
# ============================================================
def create_paradigm_comparison():
    """Compare F1 scores across three paradigms on real-world has_bug prediction."""
    paradigms = ['Supervised ML\n(Stacking)', 'Change-Point\n(Binary Seg, τ=10)', 'Forecasting\n(Naive Mean)']
    f1_scores = [0.42, 0.54, 0.35]  # Real-world F1 scores, not synthetic
    colors = ['#3498db', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(paradigms, f1_scores, color=colors, edgecolor='black', linewidth=0.8)

    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('F1 Score')
    ax.set_title('Real-World Bug Prediction: Paradigm Comparison')
    ax.set_ylim(0, 0.7)
    ax.axhline(y=0.42, color='gray', linestyle='--', alpha=0.5, label='ML Baseline')

    plt.tight_layout()
    plt.savefig('figures/paradigm_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/paradigm_comparison.png', bbox_inches='tight')
    plt.close()
    print("Created: paradigm_comparison.pdf/png")

# ============================================================
# Figure 2: Change-Point Detection Algorithm Comparison
# ============================================================
def create_cpd_comparison():
    """Compare change-point detection algorithms."""
    algorithms = ['CUSUM', 'PELT', 'BOCPD', 'BinSeg', 'Sliding\nWindow']
    precision = [0.15, 0.07, 0.295, 0.587, 0.766]
    recall = [0.139, 0.07, 0.940, 0.722, 1.0]
    f1 = [0.084, 0.07, 0.427, 0.582, 0.858]
    
    x = np.arange(len(algorithms))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='black')
    
    ax.set_ylabel('Score')
    ax.set_title('Change-Point Detection Algorithm Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('figures/cpd_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/cpd_comparison.png', bbox_inches='tight')
    plt.close()
    print("Created: cpd_comparison.pdf/png")

# ============================================================
# Figure 3: Feature Ablation Study
# ============================================================
def create_ablation_study():
    """Feature ablation study visualization."""
    feature_groups = ['All Features', 'Without\nMagnitude', 'Without\nContext', 'Without\nAnomaly', 
                      'Magnitude\nOnly', 'Context\nOnly']
    f1_scores = [0.417, 0.493, 0.398, 0.349, 0.399, 0.494]
    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(feature_groups, f1_scores, color=colors, edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar, score in zip(bars, f1_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('F1 Score')
    ax.set_title('Feature Ablation Study: Impact of Feature Groups')
    ax.axvline(x=0.417, color='gray', linestyle='--', alpha=0.7, label='Baseline (All Features)')
    ax.set_xlim(0, 0.6)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/ablation_study.pdf', bbox_inches='tight')
    plt.savefig('figures/ablation_study.png', bbox_inches='tight')
    plt.close()
    print("Created: ablation_study.pdf/png")

# ============================================================
# Figure 4: Supervised ML Model Comparison
# ============================================================
def create_ml_comparison():
    """Compare supervised ML models."""
    models = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'XGBoost', 'Stacking\nEnsemble']
    f1_scores = [0.281, 0.322, 0.079, 0.417, 0.423]
    mcc_scores = [0.042, 0.116, 0.033, 0.202, 0.179]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, mcc_scores, width, label='MCC', color='#9b59b6', edgecolor='black')
    
    ax.set_ylabel('Score')
    ax.set_title('Supervised Machine Learning Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig('figures/ml_comparison.pdf', bbox_inches='tight')
    plt.savefig('figures/ml_comparison.png', bbox_inches='tight')
    plt.close()
    print("Created: ml_comparison.pdf/png")

# ============================================================
# Figure 5: Tolerance Window Sensitivity Analysis
# ============================================================
def create_tolerance_sensitivity():
    """Tolerance window sensitivity analysis for CPD."""
    tau_values = [3, 5, 7, 10]
    precision = [0.26, 0.268, 0.272, 0.374]
    recall = [0.22, 0.373, 0.509, 1.000]
    f1 = [0.24, 0.312, 0.354, 0.544]
    ml_baseline = 0.42

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: F1 vs tau with ML baseline
    ax1.plot(tau_values, f1, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='Binary Seg F1')
    ax1.axhline(y=ml_baseline, color='#3498db', linestyle='--', linewidth=2, label=f'ML Baseline (F1={ml_baseline})')
    ax1.fill_between(tau_values, f1, ml_baseline, where=[f > ml_baseline for f in f1],
                     alpha=0.3, color='#2ecc71', label='CPD > ML')
    ax1.set_xlabel('Tolerance Window (τ)')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('(a) F1 Score vs Tolerance Window')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0, 0.7)
    ax1.set_xticks(tau_values)

    # Right: Precision-Recall trade-off
    ax2.plot(tau_values, precision, 's-', color='#e74c3c', linewidth=2, markersize=8, label='Precision')
    ax2.plot(tau_values, recall, '^-', color='#9b59b6', linewidth=2, markersize=8, label='Recall')
    ax2.set_xlabel('Tolerance Window (τ)')
    ax2.set_ylabel('Score')
    ax2.set_title('(b) Precision-Recall Trade-off')
    ax2.legend(loc='center right')
    ax2.set_ylim(0, 1.1)
    ax2.set_xticks(tau_values)

    plt.tight_layout()
    plt.savefig('figures/tolerance_sensitivity.pdf', bbox_inches='tight')
    plt.savefig('figures/tolerance_sensitivity.png', bbox_inches='tight')
    plt.close()
    print("Created: tolerance_sensitivity.pdf/png")

# ============================================================
# Main execution
# ============================================================
if __name__ == '__main__':
    print("Generating figures for ICPE 2026 paper...")
    create_paradigm_comparison()
    create_cpd_comparison()
    create_ablation_study()
    create_ml_comparison()
    create_tolerance_sensitivity()
    print("\nAll figures generated successfully!")

