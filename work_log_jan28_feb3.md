# Work Log: January 28 - February 3, 2026

**Total Hours:** 10 hours  
**Focus Areas:** Research & Literature Review (5h), Deep Learning Implementation (5h)

---

## Wednesday, January 28, 2026 - 2 hours

### Research & Literature Review (2h)
- **08:30 - 10:30** - Literature review on anomaly detection in time series
  - Read "Deep Learning for Anomaly Detection: A Survey" (Chalapathy & Chawla, 2019)
  - Reviewed recent advances in CNN-based approaches for time series classification
  - Identified potential improvements:
    - Consider 1D CNN with dilated convolutions for better temporal receptive field
    - Explore attention mechanisms for LSTM/GRU to focus on critical time windows
    - Investigate ensemble methods combining statistical and deep learning approaches
  - Documented key findings on transfer learning applicability for CI/CD performance data

---

## Thursday, January 29, 2026 - 2 hours

### Deep Learning Implementation (2h)
- **14:00 - 16:00** - Initial CNN architecture design and setup
  - Set up PyTorch environment for phase_8_deep_learning experiments
  - Designed 1D CNN architecture with multiple convolutional blocks:
    - Conv layers: [64, 128, 256] filters
    - Kernel sizes: [3, 5, 7] for multi-scale feature extraction
    - Batch normalization and dropout (0.3) for regularization
  - Implemented data preprocessing pipeline for time series input
  - Created initial training loop with Adam optimizer (lr=0.001)

---

## Friday, January 30, 2026 - 1.5 hours

### Research & Literature Review (1.5h)
- **09:00 - 10:30** - Performance regression detection methods
  - Reviewed "Automated Performance Regression Detection in CI/CD" papers
  - Analyzed Mozilla's existing Perfherder system architecture
  - Brainstormed improvements for our paper:
    - Add comparative analysis section on computational cost vs. accuracy trade-offs
    - Include discussion on false positive reduction strategies
    - Consider hybrid approach: statistical methods for quick detection + DL for validation
  - Made notes on related work section enhancements

---

## Saturday, January 31, 2026 - 1.5 hours

### Deep Learning Implementation (1.5h)
- **15:30 - 17:00** - LSTM/GRU model development
  - Implemented bidirectional LSTM architecture:
    - 2 LSTM layers with 128 hidden units each
    - Dropout layers (0.4) between LSTM layers
    - Dense layer with sigmoid activation for binary classification
  - Created GRU variant with similar architecture for comparison
  - Added early stopping callback to prevent overfitting
  - Ran preliminary training on autoland1 dataset subset (50 epochs)

---

## Sunday, February 1, 2026 - 1 hour

### Research & Literature Review (1h)
- **11:00 - 12:00** - Feature engineering techniques review
  - Studied "Time Series Feature Extraction using tsfresh" methodology
  - Reviewed our phase_3 feature engineering approach
  - Identified potential enhancements:
    - Add frequency domain features (FFT components)
    - Include change point detection features as inputs to DL models
    - Consider wavelet-based features for multi-resolution analysis
  - Outlined experimental design for feature ablation study

---

## Monday, February 2, 2026 - 2 hours

### Deep Learning Implementation & Testing (2h)
- **10:00 - 12:00** - Model training and evaluation
  - Trained CNN model on full autoland dataset:
    - Training accuracy: 87.3%, Validation accuracy: 82.1%
    - Noted overfitting issues - adjusted regularization
  - Trained LSTM model with same data split:
    - Training accuracy: 84.6%, Validation accuracy: 81.8%
    - Better generalization than CNN, but slower training
  - Implemented evaluation metrics: Precision, Recall, F1, AUC-ROC
  - Generated confusion matrices and learning curves
  - Observed that LSTM performs better on sequential patterns but CNN captures local anomalies more effectively

---

## Tuesday, February 3, 2026 - 0 hours (today)

*(No work logged yet for today)*

---

## Summary of Accomplishments

### Research & Literature (5 hours total)
- Reviewed 3+ key papers on deep learning for anomaly detection
- Identified 5+ concrete improvements for project and paper
- Documented enhancement strategies for feature engineering and model architecture
- Prepared outline for paper revisions (related work, methodology sections)

### Deep Learning Implementation (5 hours total)
- Implemented CNN architecture with multi-scale convolution
- Developed LSTM and GRU models with proper regularization
- Created training pipeline with data preprocessing and evaluation
- Completed initial experiments on autoland dataset
- Generated performance metrics and visualizations

### Key Insights
1. **CNN strengths**: Better at detecting local anomalies and faster training
2. **LSTM strengths**: Superior sequential pattern recognition, better generalization
3. **Next steps**: 
   - Implement hybrid CNN-LSTM architecture
   - Conduct comprehensive cross-dataset evaluation (mozilla-central, firefox-android)
   - Fine-tune hyperparameters using grid search
   - Compare DL models against baseline statistical methods from earlier phases

### Files Modified/Created
- `phase_8_deep_learning/run_dl_experiments.py`
- `phase_8_deep_learning/src/cnn_model.py`
- `phase_8_deep_learning/src/lstm_model.py`
- `phase_8_deep_learning/src/data_preprocessing.py`
- `phase_8_deep_learning/outputs/training_logs/`

---

**Total Weekly Hours: 10 hours**  
**Average per day: ~1.67 hours** (distributed across 6 days)
