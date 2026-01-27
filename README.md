# Mozilla Performance Alert Analysis

A machine learning pipeline for detecting and analyzing performance regressions in Mozilla's Perfherder CI system, developed for the **ICPE 2026 Data Challenge Track**.

## Overview

This project implements a 7-phase machine learning pipeline that processes 17,989 performance alerts to compare different anomaly detection paradigms for performance regression triage:

- **Supervised ML** (F1 = 0.42): Traditional classification treating alerts as independent instances
- **Change-Point Detection** (F1 = 0.54): Unsupervised temporal analysis detecting distributional shifts
- **Forecasting-Based Anomaly Detection** (F1 = 0.35): Statistical outlier detection

**Key Finding**: Time-series-based change-point detection outperforms supervised learning by 29% for performance regression detection.

## Project Structure

```
Doc_ansh/
├── src/                          # Source code
│   ├── main.py                   # Main orchestrator
│   ├── common/                   # Shared utilities
│   │   ├── data_paths.py         # Centralized path configuration
│   │   ├── evaluation_utils.py   # Evaluation metrics
│   │   ├── model_utils.py        # Model utilities
│   │   └── visualization_utils.py
│   ├── phase_1/                  # Binary Classification
│   ├── phase_2/                  # Multi-class Status Prediction
│   ├── phase_3/                  # Time-Series Feature Extraction
│   ├── phase_4/                  # Change-Point Detection
│   ├── phase_5/                  # Forecasting-Based Anomaly Detection
│   ├── phase_6/                  # Root Cause Analysis
│   └── phase_7/                  # Stacking Ensemble
├── paper/                        # ICPE 2026 submission
│   ├── main.tex                  # LaTeX paper (4 pages + references)
│   ├── figures/                  # Publication-ready figures
│   ├── QUICK_START.md            # Paper compilation guide
│   └── SUBMISSION_README.md      # Submission instructions
├── data/                         # Dataset (not in version control)
│   ├── alerts_data.csv           # 17,989 performance alerts
│   ├── bugs_data.csv             # Bug metadata
│   └── timeseries_data/          # Time-series data
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.9+
- LaTeX distribution (for paper compilation)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Doc_ansh

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn ruptures joblib
```

## Usage

### Running the Full Pipeline

```bash
# Run all 7 phases sequentially
python src/main.py

# Run specific phases
python src/main.py --phase 1 2 3

# Run with error tolerance
python src/main.py --skip-errors
```

### Running Individual Phases

```bash
# Run Phase 1 (Binary Classification) with data leakage fixes
python src/phase_1/run_fixed.py

# Run Phase 4 real-world CPD evaluation
python src/phase_4/run_real_world_cpd.py
```

### Compiling the Paper

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Dataset

The Mozilla Perfherder dataset contains:
- **17,989** performance alerts (May 2023 - May 2024)
- **5,655** unique time series
- **4,471** alerts linked to bugs (24.9%)
- Multiple repositories: Autoland, Mozilla Beta, Firefox Android
- Multiple platforms: Windows, macOS, Linux, Android

**Note**: Dataset files are not included in version control. Place `alerts_data.csv` and `bugs_data.csv` in the `data/` directory.

## Key Technical Contributions

1. **Data Leakage Detection**: Identified that `is_regression` is deterministic (`is_regression = sign(change)`), not a valid ML target
2. **Direction-Agnostic Features**: Used `|amount|`, `|amount_pct|`, `|t_value|` to prevent information leakage
3. **Meaningful Prediction Target**: Predict `has_bug` (whether alert leads to bug report)
4. **Temporal Train-Test Splits**: 80%/20% chronological split preventing future information leakage

## Results Summary

| Paradigm | F1 Score | Precision | Recall |
|----------|----------|-----------|--------|
| Supervised ML (Stacking) | 0.42 | 0.48 | 0.37 |
| Change-Point Detection (τ=10) | 0.54 | 0.37 | 1.00 |
| Forecasting (Naive Mean) | 0.35 | 0.28 | 0.47 |

## Citation

```bibtex
@inproceedings{shah2026mozilla,
  title={Comparative Analysis of Anomaly Detection Paradigms for Performance Regression Triage in Mozilla's Perfherder},
  author={Shah, Ansh},
  booktitle={Proceedings of the 17th ACM/SPEC International Conference on Performance Engineering (ICPE)},
  year={2026}
}
```

## License

This project is developed for academic research purposes as part of the ICPE 2026 Data Challenge.

