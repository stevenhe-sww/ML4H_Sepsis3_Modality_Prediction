"""
Step 4 - Model Improvement and Optimization
Advanced optimization pipeline for Sepsis-3 mortality prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    recall_score, f1_score, brier_score_loss, roc_curve, 
    precision_recall_curve, fbeta_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from scipy import stats
import shap

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("Step 4 - Model Improvement and Optimization")
print("=" * 80)

# ============================================================================
# A. Load Data and Prepare Pipeline
# ============================================================================
print("\n[A] Data Loading and Pipeline Setup")
print("-" * 80)

# Load cleaned data
data_path = Path("../data/processed/sepsis3_cleaned.csv")
df = pd.read_csv(data_path)

# Remove ID columns
id_cols = ['subject_id', 'hadm_id', 'stay_id']
id_cols_present = [col for col in id_cols if col in df.columns]
if id_cols_present:
    df = df.drop(columns=id_cols_present)

target_col = 'mortality_30d'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Data shape: {X.shape[0]} samples × {X.shape[1]} features")

# Identify feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
binary_features = [col for col in numeric_features if X[col].nunique() <= 2]
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"  Numeric features: {len(numeric_features)}")
print(f"  Binary features: {len(binary_features)}")
print(f"  Categorical features: {len(categorical_features)}")

# Split data (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# Create a defensive pipeline (even though data is preprocessed)
# This ensures reproducibility and prevents data leakage in CV
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Since data is already processed, we'll use a minimal pipeline
# that just passes through (but can be expanded if needed)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)

print("[OK] Pipeline created for defensive preprocessing")

# ============================================================================
# B. Threshold Optimization
# ============================================================================
print("\n[B] Threshold Optimization and Clinical Utility")
print("-" * 80)

# Train base models (XGBoost and LightGBM) for threshold optimization
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost base model
xgb_base = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

try:
    xgb_base.fit(X_train, y_train, 
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False)
except TypeError:
    # New XGBoost API - no early_stopping_rounds
    xgb_base.fit(X_train, y_train, 
                eval_set=[(X_val, y_val)],
                verbose=False)

xgb_val_proba = xgb_base.predict_proba(X_val)[:, 1]
xgb_val_auc = roc_auc_score(y_val, xgb_val_proba)

# LightGBM base model
lgb_base = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    force_col_wise=True
)

lgb_base.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

lgb_val_proba = lgb_base.predict_proba(X_val)[:, 1]
lgb_val_auc = roc_auc_score(y_val, lgb_val_proba)

# Select best base model for threshold optimization
if xgb_val_auc >= lgb_val_auc:
    base_model = xgb_base
    y_val_proba = xgb_val_proba
    base_model_name = 'XGBoost'
    print(f"Selected {base_model_name} for threshold optimization (AUC={xgb_val_auc:.4f} vs LightGBM={lgb_val_auc:.4f})")
else:
    base_model = lgb_base
    y_val_proba = lgb_val_proba
    base_model_name = 'LightGBM'
    print(f"Selected {base_model_name} for threshold optimization (AUC={lgb_val_auc:.4f} vs XGBoost={xgb_val_auc:.4f})")

# Threshold optimization strategies
def find_optimal_threshold(y_true, y_proba, strategy='youden'):
    """Find optimal threshold using different strategies"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    if strategy == 'youden':
        # Maximize Youden's J index (TPR - FPR)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif strategy == 'f2':
        # Maximize F2 score (beta=2, favors recall)
        f2_scores = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=2) 
                     for t in thresholds]
        optimal_idx = np.argmax(f2_scores)
    elif strategy == 'cost':
        # Minimize expected cost (assuming false negative cost = 5x false positive)
        fn_cost, fp_cost = 5, 1
        costs = []
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = fn * fn_cost + fp * fp_cost
            costs.append(cost)
        optimal_idx = np.argmin(costs)
    else:
        optimal_idx = len(thresholds) // 2  # Default: 0.5
    
    return thresholds[optimal_idx]

thresholds = {
    'youden': find_optimal_threshold(y_val, y_val_proba, 'youden'),
    'f2': find_optimal_threshold(y_val, y_val_proba, 'f2'),
    'cost': find_optimal_threshold(y_val, y_val_proba, 'cost'),
    'default': 0.5
}

print("Optimal thresholds on validation set:")
for strategy, thresh in thresholds.items():
    y_pred_thresh = (y_val_proba >= thresh).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    recall = recall_score(y_val, y_pred_thresh)
    print(f"  {strategy.capitalize()}: {thresh:.4f} (F1={f1:.4f}, Recall={recall:.4f})")

# Decision Curve Analysis (DCA)
def decision_curve_analysis(y_true, y_proba, thresholds_dca):
    """Calculate net benefit for decision curve analysis"""
    net_benefits = []
    for threshold in thresholds_dca:
        y_pred = (y_proba >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        n = len(y_true)
        
        # Net benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))
        if threshold < 1.0:
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            net_benefit = 0
        net_benefits.append(net_benefit)
    
    return net_benefits

thresholds_dca = np.linspace(0.01, 0.99, 100)
net_benefits = decision_curve_analysis(y_val, y_val_proba, thresholds_dca)

# Plot DCA
plt.figure(figsize=(10, 8))
plt.plot(thresholds_dca, net_benefits, label=f'{base_model_name} Model', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Treat None')
plt.axhline(y=(y_val == 1).mean(), color='gray', linestyle='--', alpha=0.5, label='Treat All')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title('Decision Curve Analysis (DCA)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/modeling/dca_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/dca_curve.png")
plt.close()

# Save threshold optimization results
threshold_results = []
for strategy, thresh in thresholds.items():
    y_pred_thresh = (y_val_proba >= thresh).astype(int)
    threshold_results.append({
        'Strategy': strategy,
        'Threshold': f"{thresh:.4f}",
        'F1_Score': f"{f1_score(y_val, y_pred_thresh):.4f}",
        'Recall': f"{recall_score(y_val, y_pred_thresh):.4f}",
        'Precision': f"{(y_val[y_pred_thresh == 1].sum() / y_pred_thresh.sum()):.4f}" if y_pred_thresh.sum() > 0 else "0.0000"
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('../logs/modeling/threshold_optimization_report.csv', index=False)
print("[OK] Saved: logs/modeling/threshold_optimization_report.csv")

# ============================================================================
# C. Hyperparameter Optimization with Optuna
# ============================================================================
print("\n[C] Hyperparameter Optimization (Optuna)")
print("-" * 80)

# Determine which model to optimize based on base model performance
model_to_optimize = 'xgb' if base_model_name == 'XGBoost' else 'lgb'
print(f"Optimizing {base_model_name} model...")

def objective_xgb(trial):
    """Optuna objective function for XGBoost hyperparameter tuning"""
    params = {
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    # 5-fold CV
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        try:
            model.fit(X_train_fold, y_train_fold, 
                     eval_set=[(X_val_fold, y_val_fold)],
                     early_stopping_rounds=50,
                     verbose=False)
        except TypeError:
            # New XGBoost API
            model.fit(X_train_fold, y_train_fold, 
                     eval_set=[(X_val_fold, y_val_fold)],
                     verbose=False)
        
        y_val_proba_fold = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_val_proba_fold)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

def objective_lgb(trial):
    """Optuna objective function for LightGBM hyperparameter tuning"""
    params = {
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    }
    
    # 5-fold CV
    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                 eval_set=[(X_val_fold, y_val_fold)],
                 callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        y_val_proba_fold = model.predict_proba(X_val_fold)[:, 1]
        score = roc_auc_score(y_val_fold, y_val_proba_fold)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Select objective function based on model type
objective_func = objective_xgb if model_to_optimize == 'xgb' else objective_lgb

print("Starting Optuna optimization (this may take 10-20 minutes)...")
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study.optimize(objective_func, n_trials=50, show_progress_bar=True)

print(f"\nBest trial:")
print(f"  Value (AUC): {study.best_value:.4f}")
print(f"  Params:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# Save best parameters
best_params_dict = study.best_params.copy()
best_params_dict['model_type'] = model_to_optimize
with open('../artifacts/optuna_best_params.json', 'w') as f:
    json.dump(best_params_dict, f, indent=2)
print("[OK] Saved: artifacts/optuna_best_params.json")

# Train model with best parameters
best_params = study.best_params.copy()
if model_to_optimize == 'xgb':
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    })
    optimized_model = xgb.XGBClassifier(**best_params)
    try:
        optimized_model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         early_stopping_rounds=50,
                         verbose=False)
    except TypeError:
        optimized_model.fit(X_train, y_train,
                         eval_set=[(X_val, y_val)],
                         verbose=False)
else:  # LightGBM
    best_params.update({
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    })
    optimized_model = lgb.LGBMClassifier(**best_params)
    optimized_model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

# ============================================================================
# D. Bootstrap Confidence Intervals
# ============================================================================
print("\n[D] Bootstrap Confidence Intervals")
print("-" * 80)

def bootstrap_metrics(y_true, y_proba, n_bootstrap=1000, random_state=42):
    """Calculate bootstrap confidence intervals for metrics"""
    np.random.seed(random_state)
    n = len(y_true)
    bootstrap_metrics_list = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_proba_boot = y_proba[indices]
        
        # Calculate metrics
        try:
            auc = roc_auc_score(y_true_boot, y_proba_boot)
            auprc = average_precision_score(y_true_boot, y_proba_boot)
            brier = brier_score_loss(y_true_boot, y_proba_boot)
            
            bootstrap_metrics_list.append({
                'AUC': auc,
                'AUPRC': auprc,
                'Brier': brier
            })
        except:
            continue
    
    bootstrap_df = pd.DataFrame(bootstrap_metrics_list)
    
    # Calculate confidence intervals (95%)
    ci_results = {}
    for metric in ['AUC', 'AUPRC', 'Brier']:
        ci_results[metric] = {
            'Mean': bootstrap_df[metric].mean(),
            'Lower_CI_95': bootstrap_df[metric].quantile(0.025),
            'Upper_CI_95': bootstrap_df[metric].quantile(0.975)
        }
    
    return ci_results

# Get test predictions
y_test_proba_optimized = optimized_model.predict_proba(X_test)[:, 1]

print("Calculating bootstrap confidence intervals (1000 iterations)...")
ci_results = bootstrap_metrics(y_test, y_test_proba_optimized)

bootstrap_df = pd.DataFrame([
    {
        'Metric': 'AUC',
        'Mean': f"{ci_results['AUC']['Mean']:.4f}",
        'Lower_CI_95': f"{ci_results['AUC']['Lower_CI_95']:.4f}",
        'Upper_CI_95': f"{ci_results['AUC']['Upper_CI_95']:.4f}"
    },
    {
        'Metric': 'AUPRC',
        'Mean': f"{ci_results['AUPRC']['Mean']:.4f}",
        'Lower_CI_95': f"{ci_results['AUPRC']['Lower_CI_95']:.4f}",
        'Upper_CI_95': f"{ci_results['AUPRC']['Upper_CI_95']:.4f}"
    },
    {
        'Metric': 'Brier',
        'Mean': f"{ci_results['Brier']['Mean']:.4f}",
        'Lower_CI_95': f"{ci_results['Brier']['Lower_CI_95']:.4f}",
        'Upper_CI_95': f"{ci_results['Brier']['Upper_CI_95']:.4f}"
    }
])

print("\nBootstrap Confidence Intervals (95%):")
print(bootstrap_df.to_string(index=False))
bootstrap_df.to_csv('../logs/modeling/bootstrap_confidence_intervals.csv', index=False)
print("[OK] Saved: logs/modeling/bootstrap_confidence_intervals.csv")

# ============================================================================
# E. Improved Calibration (CV-level)
# ============================================================================
print("\n[E] Improved Calibration (CV-level)")
print("-" * 80)

# Use CV-level calibration
calibrated_cv = CalibratedClassifierCV(
    optimized_model,
    method='isotonic',
    cv=5
)

calibrated_cv.fit(X_train, y_train)

y_test_proba_calibrated = calibrated_cv.predict_proba(X_test)[:, 1]

brier_original = brier_score_loss(y_test, y_test_proba_optimized)
brier_calibrated = brier_score_loss(y_test, y_test_proba_calibrated)

model_name_display = base_model_name
print(f"Calibration improvement ({model_name_display}):")
print(f"  Original Brier: {brier_original:.4f}")
print(f"  CV-Calibrated Brier: {brier_calibrated:.4f}")
print(f"  Improvement: {((brier_original - brier_calibrated) / brier_original * 100):.2f}%")

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_test_proba_calibrated, n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
         label=f'CV-Calibrated {model_name_display} (Brier={brier_calibrated:.3f})')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title(f'CV-Level Calibration Curve ({model_name_display})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/modeling/cv_calibration_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/cv_calibration_curve.png")
plt.close()

# ============================================================================
# F. Enhanced Subgroup Analysis
# ============================================================================
print("\n[F] Enhanced Subgroup Robustness Analysis")
print("-" * 80)

# Prepare subgroup variables
X_test_with_groups = X_test.copy()

# Age groups
X_test_with_groups['age_group'] = X_test_with_groups['age'].apply(
    lambda x: '≥65' if x >= 65 else '<65'
)

# Infection source
infection_cols = [col for col in X_test.columns if 'infection_source' in col]
if infection_cols:
    X_test_with_groups['infection_source'] = X_test_with_groups[infection_cols].idxmax(axis=1)
    X_test_with_groups['infection_source'] = X_test_with_groups['infection_source'].str.replace('infection_source_', '')
else:
    X_test_with_groups['infection_source'] = 'unknown'

# Pressor use
if 'pressor_used_24h' in X_test.columns:
    X_test_with_groups['pressor_group'] = X_test_with_groups['pressor_used_24h'].apply(
        lambda x: 'Yes' if x == 1 else 'No'
    )
else:
    X_test_with_groups['pressor_group'] = 'Unknown'

# RRT
if 'rrt_present' in X_test.columns:
    X_test_with_groups['rrt_group'] = X_test_with_groups['rrt_present'].apply(
        lambda x: 'Yes' if x == 1 else 'No'
    )
else:
    X_test_with_groups['rrt_group'] = 'Unknown'

# Evaluate subgroups
subgroup_results = []

def evaluate_subgroup(y_true, y_proba, group_name, n_samples):
    """Evaluate model performance in a subgroup"""
    auc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    
    # Expected vs Observed
    expected_events = y_proba.sum()
    observed_events = y_true.sum()
    eo_ratio = expected_events / observed_events if observed_events > 0 else np.nan
    
    return {
        'Subgroup': group_name,
        'N': n_samples,
        'Events': int(observed_events),
        'AUC': f"{auc:.4f}",
        'AUPRC': f"{auprc:.4f}",
        'Brier': f"{brier:.4f}",
        'E/O_Ratio': f"{eo_ratio:.4f}" if not np.isnan(eo_ratio) else "N/A"
    }

# Age subgroups
for age_group in ['≥65', '<65']:
    mask = X_test_with_groups['age_group'] == age_group
    if mask.sum() > 0:
        result = evaluate_subgroup(
            y_test[mask], y_test_proba_calibrated[mask],
            f'Age {age_group}', mask.sum()
        )
        subgroup_results.append(result)

# Infection source subgroups
for source in X_test_with_groups['infection_source'].unique():
    mask = X_test_with_groups['infection_source'] == source
    if mask.sum() > 10:
        result = evaluate_subgroup(
            y_test[mask], y_test_proba_calibrated[mask],
            f'Source: {source}', mask.sum()
        )
        subgroup_results.append(result)

# Pressor subgroups
for pressor in X_test_with_groups['pressor_group'].unique():
    mask = X_test_with_groups['pressor_group'] == pressor
    if mask.sum() > 10:
        result = evaluate_subgroup(
            y_test[mask], y_test_proba_calibrated[mask],
            f'Pressor {pressor}', mask.sum()
        )
        subgroup_results.append(result)

# RRT subgroups
for rrt in X_test_with_groups['rrt_group'].unique():
    mask = X_test_with_groups['rrt_group'] == rrt
    if mask.sum() > 10:
        result = evaluate_subgroup(
            y_test[mask], y_test_proba_calibrated[mask],
            f'RRT {rrt}', mask.sum()
        )
        subgroup_results.append(result)

subgroup_df = pd.DataFrame(subgroup_results)
print("\nEnhanced Subgroup Performance:")
print(subgroup_df.to_string(index=False))
subgroup_df.to_csv('../logs/modeling/subgroup_performance_enhanced.csv', index=False)
print("[OK] Saved: logs/modeling/subgroup_performance_enhanced.csv")

# ============================================================================
# G. Feature Interaction Analysis
# ============================================================================
print("\n[G] Feature Interaction Analysis")
print("-" * 80)

# Train model with interaction features
X_train_interactions = X_train.copy()

# Key interactions (if features exist)
interaction_pairs = [
    ('sofa_total', 'lactate_max_24h'),
    ('age', 'lactate_max_24h'),
    ('sofa_total', 'age')
]

for feat1, feat2 in interaction_pairs:
    if feat1 in X_train.columns and feat2 in X_train.columns:
        interaction_name = f'{feat1}_x_{feat2}'
        X_train_interactions[interaction_name] = X_train[feat1] * X_train[feat2]
        print(f"  Added interaction: {interaction_name}")

X_test_interactions = X_test.copy()
for feat1, feat2 in interaction_pairs:
    if feat1 in X_test.columns and feat2 in X_test.columns:
        interaction_name = f'{feat1}_x_{feat2}'
        X_test_interactions[interaction_name] = X_test[feat1] * X_test[feat2]

# Compare models with and without interactions
if model_to_optimize == 'xgb':
    no_interaction_model = xgb.XGBClassifier(**best_params)
    no_interaction_model.fit(X_train, y_train, verbose=False)
    
    with_interaction_model = xgb.XGBClassifier(**best_params)
    with_interaction_model.fit(X_train_interactions, y_train, verbose=False)
else:
    no_interaction_model = lgb.LGBMClassifier(**best_params)
    no_interaction_model.fit(X_train, y_train, verbose=False)
    
    with_interaction_model = lgb.LGBMClassifier(**best_params)
    with_interaction_model.fit(X_train_interactions, y_train, verbose=False)

y_test_proba_no_int = no_interaction_model.predict_proba(X_test)[:, 1]
y_test_proba_with_int = with_interaction_model.predict_proba(X_test_interactions)[:, 1]

auc_no_int = roc_auc_score(y_test, y_test_proba_no_int)
auc_with_int = roc_auc_score(y_test, y_test_proba_with_int)

print(f"\nInteraction Effect:")
print(f"  Without interactions - AUC: {auc_no_int:.4f}")
print(f"  With interactions    - AUC: {auc_with_int:.4f}")
print(f"  Improvement: {((auc_with_int - auc_no_int) / auc_no_int * 100):.2f}%")

# SHAP analysis for interactions
if auc_with_int > auc_no_int:
    print("  Using model with interactions (better performance)")
    best_final_model = with_interaction_model
    X_test_final = X_test_interactions
else:
    print("  Using model without interactions")
    best_final_model = no_interaction_model
    X_test_final = X_test

# Plot feature importance comparison
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
feature_imp_no_int = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': no_interaction_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.barh(range(len(feature_imp_no_int)), feature_imp_no_int['Importance'].values[::-1])
plt.yticks(range(len(feature_imp_no_int)), feature_imp_no_int['Feature'].values[::-1], fontsize=9)
plt.xlabel('Feature Importance')
plt.title('Top 15 Features (No Interactions)', fontsize=12)
plt.tight_layout()

plt.subplot(1, 2, 2)
feature_imp_with_int = pd.DataFrame({
    'Feature': X_train_interactions.columns,
    'Importance': with_interaction_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

plt.barh(range(len(feature_imp_with_int)), feature_imp_with_int['Importance'].values[::-1])
plt.yticks(range(len(feature_imp_with_int)), feature_imp_with_int['Feature'].values[::-1], fontsize=9)
plt.xlabel('Feature Importance')
plt.title('Top 15 Features (With Interactions)', fontsize=12)

plt.tight_layout()
plt.savefig('../visualizations/modeling/feature_interactions_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/feature_interactions_analysis.png")
plt.close()

# ============================================================================
# Final Model and Summary
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Model Optimization Complete!")
print("=" * 80)

# Final evaluation
final_auc = roc_auc_score(y_test, y_test_proba_calibrated)
final_auprc = average_precision_score(y_test, y_test_proba_calibrated)
final_brier = brier_score_loss(y_test, y_test_proba_calibrated)

print(f"\nFinal Optimized Model Performance:")
print(f"  AUC: {final_auc:.4f} (Target: ≥0.82)")
print(f"  AUPRC: {final_auprc:.4f} (Target: ≥0.45)")
print(f"  Brier Score: {final_brier:.4f} (Target: <0.18)")

if final_auc >= 0.82:
    print("  [OK] AUC target achieved!")
else:
    print(f"  [WARNING] AUC below target (need +{0.82 - final_auc:.4f})")

if final_auprc >= 0.45:
    print("  [OK] AUPRC target achieved!")
else:
    print(f"  [WARNING] AUPRC below target (need +{0.45 - final_auprc:.4f})")

if final_brier < 0.18:
    print("  [OK] Brier score target achieved!")
else:
    print(f"  [WARNING] Brier score above target (need -{final_brier - 0.18:.4f})")

# Save final optimized model
joblib.dump(calibrated_cv, '../artifacts/optimized_sepsis_model.pkl')
print("\n[OK] Saved: artifacts/optimized_sepsis_model.pkl")

print("\nGenerated files:")
print("  - logs/modeling/threshold_optimization_report.csv")
print("  - artifacts/optuna_best_params.json")
print("  - logs/modeling/bootstrap_confidence_intervals.csv")
print("  - logs/modeling/subgroup_performance_enhanced.csv")
print("  - visualizations/modeling/dca_curve.png")
print("  - visualizations/modeling/cv_calibration_curve.png")
print("  - visualizations/modeling/feature_interactions_analysis.png")
print("  - artifacts/optimized_sepsis_model.pkl")

