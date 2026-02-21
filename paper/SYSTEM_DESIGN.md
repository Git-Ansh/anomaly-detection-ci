# Cascading Confidence-Gated Triage System — Complete Design Document

## Paper Narrative (4 Parts)

### Part 1: Detection ("If It Ain't Broke...")
- Perfherder's T-test detects >90% of true regressions
- Benchmark alternatives: PELT, BinSeg, CUSUM, BOCD (Phase 4), ARIMA/Naive/ES (Phase 5)
- Finding: marginal accuracy gains (BinSeg F1=0.54 vs T-test baseline)
- Conclusion: cost of switching detection systems NOT justified by marginal gain
- **Transition**: the real bottleneck is triage, not detection

### Part 2: Triage Cascade ("Where the Real Problem Is")
- Explain sheriff workflow and how our system fits in
- Full cascade: Stage 0 → Stage 1 → Stage 2 → Stage 3
- Performance gains and workload reduction results

### Part 3: Bug Description Prediction ("After has_bug")
- After has_bug=2 is predicted, predict bug details (component, product, summary)
- Only 482 bugs available — use LLM few-shot + retrieval baseline
- Phase 6 topic model (5 topics) as baseline comparison

### Part 4: Generalization
- Cross-repository transfer within Mozilla (train autoland → test mozilla-beta)
- External validation on 1-2 other datasets (Azure TUNA, TSB-AD or ApacheJIT)
- Framework description for applying to any CI system

---

## Data Overview

### Primary Dataset: Mozilla Perfherder
- **17,989 alerts** in **3,912 alert summaries** (groups)
- **482 unique bugs** linked to alert summaries
- **~11,310 time-series CSV files** across 6 repositories
- Time span: May 2023 to May 2024

### Alert Summary Status Distribution (Group Level)
| Code | Label         | Summaries | Alerts | has_bug Rate | Mean alerts/group |
|------|---------------|-----------|--------|:------------:|:-----------------:|
| 0    | Untriaged     | 656       | 2,203  | 0.0%         | 3.36              |
| 1    | Downstream    | 30        | 241    | 6.7%         | 8.03              |
| 2    | Reassigned    | 1,224     | 5,215  | 0.7%         | 4.26              |
| 3    | Invalid       | 488       | 854    | 0.4%         | 1.75              |
| 4    | Acknowledged  | 692       | 3,424  | 37.0%        | 4.95              |
| 5    | Investigating | 565       | 4,564  | 45.7%        | 8.08              |
| 6    | Wontfix       | 158       | 888    | 39.9%        | 5.62              |
| 7    | Fixed         | 84        | 564    | 33.3%        | 6.71              |
| 8    | Backedout     | 15        | 36     | 100%         | 2.40              |

### Individual Alert Status Distribution
| Code | Label        | Alerts |
|------|--------------|--------|
| 0    | Untriaged    | 2,171  |
| 1    | Downstream   | 270    |
| 2    | Reassigned   | 5,627  |
| 3    | Invalid      | 1,136  |
| 4    | Acknowledged | 8,785  |

### Training Data (Resolved Only)
- **2,691 summaries** (68.8%) — excluding Untriaged (656) and Investigating (565)
- **11,222 alerts** (62.4%)

---

## Key Definitions

### Alert
A single performance test that exceeded the regression detection threshold. Belongs to one test signature. Carries metadata: magnitude, T-value, suite, platform, noise profile.

### Alert Summary (Group)
Collection of alerts triggered by the same code push. 3,912 summaries, avg 4.6 alerts each. Receives a single group-level status and optionally a linked bug report.

### Summary Status vs Individual Alert Status
- **Summary status** (9 codes, 0-8): Group-level disposition including Investigating, Wontfix, Fixed, Backedout
- **Individual alert status** (5 codes, 0-4): Per-alert role within the group. Codes 5-8 do NOT exist at alert level.
- Summary status is identical for all alerts in a group (verified: 0 inconsistencies)
- Individual alert status can vary within a group (162 of 3,912 groups have mixed statuses)

### Status Semantics
ALL non-Invalid, non-Untriaged labels imply the alert is REAL. The difference is disposition:
- **Acknowledged**: Confirmed real, being tracked
- **Reassigned**: Real, transferred to another team
- **Downstream**: Real, but a side-effect of another regression
- **Wontfix**: Real, but cost of fixing not justified
- **Fixed**: Real, and a patch was shipped
- **Backedout**: Real, and the offending code was reverted

### has_bug
Binary group-level property: `alert_summary_bug_number.notna()`. Overall rate: 16.2% of summaries.

### "Investigating" Semantic Mismatch
- **Training data** (sheriff): "I'm still looking at this" — temporary, in-progress state
- **Model output** (inference): "I'm not confident enough" — final deferral to human
- **Solution**: Never train on Investigating labels. They emerge from the confidence gate.

---

## System Architecture

### Overview
Cascading pipeline with 4 stages. Each stage uses calibrated confidence thresholds (selective prediction / learning to defer). Only confident predictions are auto-labeled; uncertain cases are deferred.

### Label Scheme (Unified)
- Code **0** always means "not evaluated / uncertain / deferred to sheriff"
- This applies to summary status, individual alert status, AND has_bug

| Output           | 0                  | Other codes            |
|------------------|--------------------|------------------------|
| Summary status   | Investigating      | 1-4, 6-7 (dispositions)|
| Alert status     | Investigating      | 1-4 (roles)            |
| has_bug          | Uncertain          | 1 = no bug, 2 = has bug|

---

### Stage 0: Group Invalid Filter

**Level**: Summary (group)
**Task**: Binary — Invalid vs Valid
**Training data**: 488 Invalid vs 2,203 Valid summaries

**Key features**: Group size, dominant suite, aggregated TS noise metrics (CV, variance ratio, direction change rate)

**Discriminative signals**:
- Invalid summaries are smaller (median 1 alert, 73.2% single-alert vs 49.9% for valid)
- Certain suites near-deterministic: sccache (98.9% invalid), decision (100%), installer-size
- But 76.6% of invalids scattered across normal suites

**Confidence gate**:
- Confident Invalid → auto-label group + all alerts as Invalid; has_bug = 1 (no bug)
- Confident Valid → pass to Stage 1
- Uncertain → group = Investigating → sheriff queue (goes to Stage 2b for noise flagging)

**Expected performance**:
- Invalid precision: ~85-90% (with confidence gate)
- Invalid recall: ~40-50%
- ~200-245 summaries auto-invalidated

---

### Stage 1: Group Disposition

**Level**: Summary (group)
**Task**: 5-class — Acknowledged(+Backedout), Reassigned, Wontfix, Fixed, Downstream
**Training data**: 2,203 resolved non-Invalid summaries

**Class distribution**:
| Class                      | Count | %     |
|----------------------------|-------|-------|
| Reassigned                 | 1,224 | 55.6% |
| Acknowledged (+Backedout)  | 707   | 32.1% |
| Wontfix                    | 158   | 7.2%  |
| Fixed                      | 84    | 3.8%  |
| Downstream                 | 30    | 1.4%  |

**Design decisions**:
- Backedout (15 samples) merged into Acknowledged for training
- Investigating and Untriaged excluded from training
- "Investigating" is NEVER a predicted class — emerges from low confidence

**Confidence gate**:
- Confident → auto-label group; pass to Stage 2a
- Uncertain → group = Investigating; pass to Stage 2b; route to sheriff

**Challenge**: Fixed/Wontfix are post-triage decisions hard to predict from alert features. System compensates by deferring when uncertain.

**Expected performance**:
- ~50-60% of groups get confident auto-labels
- Overall accuracy on confident subset: ~72-80%

---

### Stage 2: Individual Alert Roles

#### Stage 2a: Full Classification (for confident groups from Stage 1)

**Level**: Individual alert
**Task**: 4-class — Acknowledged, Downstream, Reassigned, Invalid
**Confidence gate**: Confident → auto-label; Uncertain → alert = Investigating → sheriff

**Key signals**:
- Downstream: related_summary_id present, smaller magnitude than primary alert
- Invalid: high TS noise, small magnitude relative to variance
- Reassigned: different suite/platform than majority in group

#### Stage 2b: Simplified Noise Filter (for Investigating groups)

**Level**: Individual alert
**Task**: Binary — Invalid (confident noise) vs Investigating (everything else)
**Rationale**: Group is uncertain → don't claim alerts are "Acknowledged." Only flag obvious noise. All others inherit "Investigating."

**Expected performance (2a)**: ~70-78% on confident predictions

---

### Stage 3: Bug Linkage Prediction (has_bug)

**Level**: Summary (group)
**Task**: Binary with abstention — 0 (uncertain), 1 (no bug), 2 (has bug)

**Shortcut rules (no model)**:
- Invalid groups → has_bug = 1 (0.4% bug rate, negligible)
- Reassigned groups → has_bug = 1 (0.7% bug rate)

**Two prediction modes**:
- Mode A (confident status): Uses predicted disposition as feature. Higher accuracy.
- Mode B (Investigating groups): No status context. Prediction is a hint for sheriff.

**Training**: Uses cross-validated Stage 1 predictions (not true labels) to handle train-inference mismatch.

**Expected performance**: F1 0.55-0.65 (Mode A), F1 0.40-0.50 (Mode B)

---

## Estimated System Impact

### Workload Reduction
| Metric                          | Current | With System    | Savings     |
|---------------------------------|---------|----------------|-------------|
| Groups needing full manual review | 2,691   | ~1,300         | ~48-55%     |
| Alerts needing any human review | 11,222  | ~7,200-9,300   | ~17-36%     |
| Groups fully automated          | 0       | ~900-1,100     | 33-41%      |
| Invalid groups auto-filtered    | 0       | ~200-245       | 41-50%      |

### Per-Stage Accuracy (on auto-labeled subset)
| Stage                        | Accuracy  |
|------------------------------|-----------|
| Stage 0 (Invalid filter)    | 88-92%    |
| Stage 1 (Group disposition)  | 72-80%    |
| Stage 2 (Individual roles)   | 75-82%    |
| Stage 3 (Bug linkage)        | 70-78%    |
| **End-to-end**               | **65-75%**|

### Why Alert Savings < Group Savings
Invalid groups (caught best) are small (1.75 alerts avg). Large complex groups (8+ alerts) are hardest to auto-resolve. System efficiently handles many small clear-cut cases, defers complex ones to humans.

---

## Safety Properties

1. **No silent failures**: Every uncertain prediction is code 0. System never guesses when unsure.
2. **Self-correcting cascade**: Invalid groups that slip through Stage 0 will likely be uncertain in Stage 1 → routed to sheriff.
3. **Conservative errors**: Wrong predictions tend to be low-harm (Acknowledged ↔ Wontfix, both "real regression").
4. **Consistent uncertainty**: Code 0 = "needs human" across all outputs.
5. **Human-in-the-loop by design**: Assists sheriffs, doesn't replace them.

---

## Mapping Existing Phases to Paper

| Existing Phase | Paper Section | Role |
|---------------|---------------|------|
| Phase 1 (Binary classification) | Part 2: Stage 3 | has_bug prediction |
| Phase 2 (Multi-class status) | Part 2: Stage 1 | Group disposition |
| Phase 3 (TS feature extraction) | Part 2: All stages | Feature engineering |
| Phase 4 (CPD benchmarking) | Part 1: Detection | PELT/BinSeg/CUSUM/BOCD comparison |
| Phase 5 (Forecasting) | Part 1: Detection | ARIMA/Naive/ES comparison |
| Phase 6 (Root cause analysis) | Part 3: Bug prediction | Baseline topic model + clustering |
| Phase 7 (Stacking ensemble) | Part 2: Architecture | Meta-confidence / cascade design |
| Phase 8 (Deep learning) | Part 2: Alternative models | CNN/LSTM/Transformer for stages |

---

## Bug Description Prediction (Part 3)

### Data: bugs_data.csv
- 482 rows, 30 columns
- **summary**: Short text (avg 85 chars), all populated
- **component**: 101 unique values
- **product**: Core (273), Testing (61), DevTools (49), Firefox (38)...
- **type**: defect (337), task (89), enhancement (56)
- **resolution**: FIXED (292), WONTFIX (111), DUPLICATE (20), INVALID (11)
- **No full description** — only summary line

### Approach A: LLM Few-Shot (Primary)
- Feed alert features + 5 similar past bug examples → predict component, product, type
- Generate draft bug summary
- Evaluate: leave-one-out on 482 bugs
- Metrics: accuracy for component/product, ROUGE/BERTScore for summary

### Approach B: Retrieval + Template (Baseline)
- Embed 482 summaries with sentence-transformers
- For new alert, find 5 nearest bugs by alert features
- Predict component/product from majority vote of neighbors
- Fill template for summary text

### Approach C: Fine-tuned small model (Optional)
- DistilBERT on collapsed top-10 components
- Data augmentation for 482 samples
- Cross-validated

---

## Generalization (Part 4)

### Internal: Cross-Repository Transfer (Within Mozilla)
- Train on autoland repos → test on mozilla-beta/mozilla-release
- Tests whether learned patterns transfer across Mozilla sub-projects

### External Dataset 1: Azure TUNA (ICSME 2025)
- 483 days of VM benchmark noise on Azure
- Apply Phase 4/5 CPD and forecasting methods
- Validates detection algorithms beyond Mozilla

### External Dataset 2: TSB-AD (NeurIPS 2024) or ApacheJIT
- **TSB-AD**: 1,070 labeled anomaly time series. Benchmark our CPD algorithms.
- **ApacheJIT**: 106K labeled commits. Apply binary classification (like Phase 1).
- Choose based on time constraints.

### Generalization Framework
Abstract requirements for applying cascade to any CI:
1. Performance time-series data
2. Alert/notification system
3. Historical triage labels
4. Bug linkage (issue tracker integration)
