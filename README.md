# MACCP: Model-Agreement Conditioned Conformal Prediction for Multi-Class Bug Triage

QRS 2026 submission. Conformal prediction sets with cross-model agreement conditioning for automated bug component assignment.

## Paper

`paper/qrs2026_maccp.tex` — 10-page IEEE conference paper with 12 tables, 6 figures, 33 references.

Compile in Overleaf with `paper/references.bib` and `paper/figures/*.pdf`.

## Key Results (Eclipse, 30 classes, alpha=0.10)

| Method | Coverage | Mean Set | Singletons | Sing. Acc |
|--------|----------|----------|------------|-----------|
| DeBERTa RAPS | 87.3% | 5.30 | 23.5% | 84.7% |
| XGBoost RAPS | 93.5% | 3.12 | 20.1% | 93.6% |
| **MACCP agree (47%)** | **91.5%** | **2.30** | **61.2%** | **90.0%** |
| MACCP disagree (53%) | 88.4% | 8.52 | 0.1% | — |

Cross-model agreement (DeBERTa + XGBoost) provides 43-69pp accuracy gaps within identical confidence bins.

## Datasets

| Dataset | Classes | Train | Cal | Test | Text |
|---------|---------|-------|-----|------|------|
| Eclipse (Zenodo 2024) | 30 | 102,725 | 30,017 | 25,499 | Summary + Description |
| Mozilla Core (BugsRepo) | 20 | 28,964 | 8,289 | 8,480 | Summary only |
| ServiceNow (UCI 498) | 11 | 13,635 | 4,545 | 4,546 | None (structured) |

## Repository Structure

```
paper/                          # QRS 2026 paper
  qrs2026_maccp.tex             # Main paper
  references.bib                # Bibliography
  figures/                      # PDF figures
  generate_qrs_figures.py       # Figure generation script

src/                            # Source code
  cascade_legacy/               # Legacy cascade framework (not used in paper)
  common/                       # Shared utilities
  conformal/                    # MACCP conformal prediction framework
    data/                       # Dataset loaders (Eclipse, ServiceNow, Mozilla)
    stages/                     # Per-dataset stage configs
    pipeline/                   # Entry points + MACCP scripts
      prepare_data.py           # 60/20/20 temporal split + label mapping
      finetune_deberta.py       # DeBERTa-v3-base fine-tuning (DDP)
      run_conformal.py          # RAPS conformal prediction + AUGRC
      run_maccp.py              # MACCP implementation
      run_xgb_agreement_analysis.py  # Cross-model agreement analysis
    explanation_generator.py    # Rule-based explanation module

model_ablation/                 # Side study: model ablation experiments
  scripts/                      # Experiment scripts (1-6)
  data/                         # Copied prediction files
  results/                      # Experiment results
    hybrid/                     # Hybrid MACCP configs
    llm_zeroshot/               # GLM-5 zero-shot (failed)
    deberta_large/              # DeBERTa-v3-large results
  cp_baselines/                 # Mondrian CP + SACP baselines
  mozilla_fix/                  # Mozilla XGBoost feature engineering

jobs/                           # TamIA SLURM job scripts
scripts/                        # Local helper scripts (monitoring, deployment)
data/                           # Dataset files (gitignored)
outputs/                        # Experiment outputs (gitignored)
```

## Running

### Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/WSL
pip install pandas numpy scikit-learn xgboost scipy matplotlib torch transformers
```

### Data Preparation
```bash
python src/conformal/pipeline/prepare_data.py \
  --data_dir data/external/eclipse_zenodo_lite \
  --output_dir data/eclipse/ \
  --top_k 30 --dataset_name eclipse --no_other \
  --date_col creation_time --component_col component \
  --summary_col summary --description_col description
```

### DeBERTa Fine-tuning (4x H100)
```bash
torchrun --nproc_per_node=4 src/conformal/pipeline/finetune_deberta.py \
  --data_dir data/eclipse/ --output_dir models/deberta_eclipse/ \
  --max_length 512 --batch_size 16 --epochs 5 --lr 2e-5 --distributed
```

### Conformal Prediction
```bash
python src/conformal/pipeline/run_conformal.py \
  --model_dir models/deberta_eclipse/ --data_dir data/eclipse/ \
  --output_dir results/eclipse/ --method raps --alpha_levels 0.01 0.05 0.10 0.20
```

### MACCP
```bash
python src/conformal/pipeline/run_maccp.py
```

## TamIA HPC

SLURM jobs in `jobs/`. Requires Alliance Canada account with GPU allocation.
- 4x NVIDIA H100 80GB per node
- `module load python/3.11.5 cuda/12.6 scipy-stack/2026a arrow/17.0.0`
- `HF_HUB_OFFLINE=1` on compute nodes (no internet)

## Citation

Paper under review at QRS 2026.
