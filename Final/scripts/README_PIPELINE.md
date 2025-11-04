# Main Training Pipeline Usage Guide

## Overview

`main_training_pipeline.py` is a comprehensive, production-ready training pipeline for Sepsis-3 mortality prediction. It combines baseline modeling, hyperparameter optimization, calibration, and comprehensive evaluation in a single script with progress bars and detailed logging.

## Features

- **Two-Stage Pipeline**: Baseline models + Optimization
- **Progress Bars**: tqdm integration for all major operations
- **Comprehensive Logging**: Both file and console logging
- **Multiple Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, Stacking
- **Hyperparameter Optimization**: Optuna with Bayesian optimization
- **Threshold Optimization**: Multiple strategies (Youden's J, F2, Cost-based)
- **Model Calibration**: CV-level calibration with Isotonic Regression
- **Bootstrap Confidence Intervals**: 1000 iterations for robust estimates
- **Subgroup Analysis**: Performance across different patient subgroups
- **SHAP Interpretability**: Feature importance and explanations
- **Decision Curve Analysis**: Clinical utility assessment

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap optuna tqdm joblib matplotlib seaborn
```

## Usage

### Basic Usage

```bash
cd scripts
python main_training_pipeline.py \
  --data ../data/processed/sepsis3_cleaned.csv \
  --out_dir ../outputs \
  --seed 42 \
  --optuna_trials 60 \
  --cv_folds 5
```

### Arguments

- `--data`: (Required) Path to cleaned data CSV file
- `--out_dir`: (Optional) Output directory (default: `outputs`)
- `--seed`: (Optional) Random seed for reproducibility (default: 42)
- `--optuna_trials`: (Optional) Number of Optuna trials (default: 60)
- `--cv_folds`: (Optional) Number of CV folds for calibration (default: 5)

### Output Structure

```
outputs/
├── logs/
│   ├── train.log                              # Detailed training log
│   ├── model_performance_summary.csv          # All models' performance
│   ├── shap_feature_importance.csv            # Top 20 SHAP features
│   ├── threshold_optimization_report.csv      # Optimal thresholds
│   ├── bootstrap_confidence_intervals.csv     # 95% CI for metrics
│   └── subgroup_performance_table.csv         # Subgroup analysis
├── artifacts/
│   ├── best_sepsis_model.pkl                 # Best model (pickled)
│   └── optuna_best_params.json               # Optimized hyperparameters
└── plots/
    ├── roc_pr_curves.png                     # ROC and PR curves
    ├── calibration_plot.png                  # Calibration curve
    ├── shap_summary.png                      # SHAP summary plot
    └── dca_plot.png                          # Decision curve analysis
```

## Pipeline Stages

### Stage A: Baseline Modeling

1. **Data Loading & Splitting**
   - Loads cleaned data
   - Stratified train/val/test split (70/15/15)
   - Builds preprocessing pipeline

2. **Model Training**
   - Logistic Regression (L2 regularization, balanced weights)
   - Random Forest (balanced class weights)
   - XGBoost (scale_pos_weight, early stopping)
   - LightGBM (scale_pos_weight, early stopping)
   - Stacking Ensemble (LR + XGB + RF meta-learner)

3. **Evaluation**
   - AUC, AUPRC, Accuracy, Recall, F1, Brier Score
   - ROC and PR curves
   - Calibration curves

4. **Interpretability**
   - SHAP values computation
   - Feature importance ranking

### Stage B: Optimization

1. **Hyperparameter Optimization**
   - Optuna Bayesian optimization
   - 5-fold stratified CV
   - Optimizes best baseline model (XGBoost or LightGBM)

2. **Threshold Optimization**
   - Youden's J index (maximize TPR - FPR)
   - F2 score (favor recall)
   - Cost minimization (FN cost >> FP cost)

3. **Model Calibration**
   - CV-level Isotonic Regression calibration
   - Brier score improvement

4. **Robustness Assessment**
   - Bootstrap confidence intervals (1000 iterations)
   - Subgroup analysis (age, infection source, pressor use, RRT)
   - Decision curve analysis

## Progress Bars

The pipeline uses tqdm to show progress:

- `[Main] Pipeline`: Overall pipeline progress
- `[Data] Loading`: Data loading steps
- `[Model] XGBoost training`: Model-specific training
- `[SHAP] Computing values`: SHAP computation
- `[Optuna] Trials`: Hyperparameter optimization
- `[Threshold] F2 optimization`: Threshold search
- `[Bootstrap] AUC CI`: Bootstrap iterations

## Logging

All operations are logged to `outputs/logs/train.log` with timestamps. The log includes:

- Data statistics
- Model training progress
- Performance metrics
- Optimization results
- Error messages (if any)

## Model Selection

The pipeline automatically:
1. Trains all baseline models
2. Selects best model by AUC
3. Optimizes hyperparameters if best model is XGBoost/LightGBM
4. Calibrates probabilities
5. Compares optimized vs baseline performance

## Example Output

```
[Main] Pipeline: 100%|████████████| 4/4 [15:23<00:00, 230.75s/stage]

[COMPLETE] Pipeline finished in 923.45 seconds
Results saved to: outputs

Best Model: XGBoost_Optimized
  Test AUC: 0.8456
  Test AUPRC: 0.5234
  Test Brier Score: 0.1654
```

## Notes

- All random processes use the specified seed for reproducibility
- Early stopping prevents overfitting (50 rounds)
- Bootstrap CI provides robust uncertainty estimates
- Subgroup analysis ensures model fairness
- DCA helps identify clinically useful thresholds

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **Memory issues**: Reduce Optuna trials or use fewer models
3. **Slow SHAP**: Reduce max_samples parameter in compute_shap_values()
4. **Optuna timeout**: Reduce --optuna_trials if needed

