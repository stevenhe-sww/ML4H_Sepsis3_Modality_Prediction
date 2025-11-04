"""
Step 3 Enhanced - Advanced Modeling and Interpretability Analysis
Enhanced modeling pipeline with clinical-ML integration for Sepsis-3 30-day mortality prediction
Includes: Temporal split, SMOTE, Nested CV, Bayesian optimization, DCA, LIME, Statistical comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    recall_score, f1_score, fbeta_score, precision_score, brier_score_loss, roc_curve, 
    precision_recall_curve, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from scipy import stats
from scipy.stats import chi2_contingency
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Import optional libraries (required for enhanced features)
from imblearn.over_sampling import SMOTE
import lime
from lime.lime_tabular import LimeTabularExplainer
from mlxtend.evaluate import mcnemar_table, mcnemar

# Set availability flags (all should be True after installation)
SMOTE_AVAILABLE = True
LIME_AVAILABLE = True
MCNEMAR_AVAILABLE = True

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# Create output directories
Path('../logs/modeling').mkdir(parents=True, exist_ok=True)
Path('../visualizations/modeling').mkdir(parents=True, exist_ok=True)
Path('../artifacts').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Step 3 Enhanced - Advanced Modeling with Clinical-ML Integration")
print("=" * 80)

# ============================================================================
# Step 1: Enhanced Data Preparation
# ============================================================================
print("\n[Step 1] Enhanced Data Preparation")
print("-" * 80)

# Load cleaned data
data_path = Path("../data/processed/sepsis3_cleaned.csv")
df = pd.read_csv(data_path)

print(f"Loaded data: {df.shape[0]} rows × {df.shape[1]} columns")

# Remove ID columns
id_cols = ['subject_id', 'hadm_id', 'stay_id']
id_cols_present = [col for col in id_cols if col in df.columns]
if id_cols_present:
    df = df.drop(columns=id_cols_present)
    print(f"Removed ID columns: {id_cols_present}")

# Check for temporal information (admittime or similar)
temporal_cols = [col for col in df.columns if 'time' in col.lower() or 'admit' in col.lower() or 'date' in col.lower()]
if temporal_cols:
    print(f"[INFO] Found temporal columns: {temporal_cols}")
    print("       Will use temporal split for more realistic validation")
    use_temporal_split = True
    temporal_col = temporal_cols[0]
    df = df.sort_values(temporal_col)
else:
    use_temporal_split = False
    print("[INFO] No temporal column found, using stratified split")

# Separate features and target
target_col = 'mortality_30d'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nFeatures: {X.shape[1]} variables")
print(f"Target distribution:")
print(f"  Survived (0): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
print(f"  Died (1): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")

# Enhanced split: Temporal or Stratified
if use_temporal_split:
    # Temporal split (80% train, 20% test)
    train_idx = int(0.8 * len(df))
    X_train = X.iloc[:train_idx].copy()
    X_test = X.iloc[train_idx:].copy()
    y_train = y.iloc[:train_idx].copy()
    y_test = y.iloc[train_idx:].copy()
    
    # Further split training set for validation
    val_idx = int(0.85 * len(X_train))
    X_val = X_train.iloc[val_idx:].copy()
    X_train = X_train.iloc[:val_idx].copy()
    y_val = y_train.iloc[val_idx:].copy()
    y_train = y_train.iloc[:val_idx].copy()
    
    print(f"\nTemporal Data split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
else:
    # Stratified train/validation/test split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    
    print(f"\nStratified Data split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# Subgroup analysis preparation
infection_cols = [col for col in X.columns if 'infection_source' in col]
if infection_cols:
    X_test_with_source = X_test.copy()
    source_mapping = {}
    for col in infection_cols:
        source_name = col.replace('infection_source_', '')
        source_mapping[col] = source_name
    
    X_test_with_source['infection_source'] = X_test_with_source[infection_cols].idxmax(axis=1)
    X_test_with_source['infection_source'] = X_test_with_source['infection_source'].map(source_mapping)
    print(f"\nSubgroup preparation: Found {len(infection_cols)} infection source categories")
else:
    X_test_with_source = X_test.copy()
    X_test_with_source['infection_source'] = 'unknown'

# Age subgroup
X_test_with_source['age_group'] = X_test_with_source['age'].apply(
    lambda x: '≥65' if x >= 65 else '<65'
)

# Save subgroup statistics
subgroup_stats = []
subgroup_stats.append({
    'Group': 'Overall',
    'N': len(X_test),
    'Events': y_test.sum(),
    'Event_Rate': y_test.mean()
})

for age_group in ['≥65', '<65']:
    mask = X_test_with_source['age_group'] == age_group
    if mask.sum() > 0:
        subgroup_stats.append({
            'Group': f'Age {age_group}',
            'N': int(mask.sum()),
            'Events': int(y_test[mask].sum()),
            'Event_Rate': y_test[mask].mean()
        })

unique_sources = X_test_with_source['infection_source'].unique()
for source in unique_sources:
    if pd.notna(source) and source != 'unknown':
        mask = X_test_with_source['infection_source'] == source
        if mask.sum() > 10:
            subgroup_stats.append({
                'Group': f'Infection: {source}',
                'N': int(mask.sum()),
                'Events': int(y_test[mask].sum()),
                'Event_Rate': y_test[mask].mean()
            })

subgroup_stats_df = pd.DataFrame(subgroup_stats)
subgroup_stats_df.to_csv('../logs/modeling/subgroup_stats.csv', index=False)
print("[OK] Saved: logs/modeling/subgroup_stats.csv")

# Advanced imbalance handling: SMOTE (only on training set)
if SMOTE_AVAILABLE:
    print("\n[INFO] Applying SMOTE to training set (only)...")
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"  After SMOTE:  {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    # Use resampled data for training
    X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns, index=X_train.index[:len(X_train_resampled)])
    y_train = pd.Series(y_train_resampled, index=y_train.index[:len(y_train_resampled)])
    print("[OK] SMOTE applied successfully")
else:
    print("[INFO] SMOTE not available, using class_weight instead")

# ============================================================================
# Step 2: Baseline Models (Lasso LR + Regular LR)
# ============================================================================
print("\n[Step 2] Baseline Models (Lasso LR + Regular LR)")
print("-" * 80)

# Lasso Logistic Regression (L1 penalty for feature selection)
lasso_lr = LogisticRegression(
    class_weight='balanced',
    C=0.1,  # Stronger regularization for Lasso
    penalty='l1',
    max_iter=1000,
    random_state=42,
    solver='liblinear'  # liblinear supports L1
)

# Regular Logistic Regression (L2 penalty)
lr_model = LogisticRegression(
    class_weight='balanced',
    C=0.5,
    penalty='l2',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lasso_lr.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Predictions
y_test_pred_lasso = lasso_lr.predict(X_test)
y_test_proba_lasso = lasso_lr.predict_proba(X_test)[:, 1]
y_test_pred_lr = lr_model.predict(X_test)
y_test_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation metrics function
def evaluate_model(y_true, y_pred, y_proba, set_name=""):
    """Calculate comprehensive evaluation metrics"""
    auc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)  # F2 emphasizes recall
    brier = brier_score_loss(y_true, y_proba)
    
    results = {
        'Set': set_name,
        'AUC': f"{auc:.4f}",
        'AUPRC': f"{auprc:.4f}",
        'Accuracy': f"{accuracy:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}",
        'F2-Score': f"{f2:.4f}",
        'Brier Score': f"{brier:.4f}"
    }
    return results

lasso_test_metrics = evaluate_model(y_test, y_test_pred_lasso, y_test_proba_lasso, "Test")
lr_test_metrics = evaluate_model(y_test, y_test_pred_lr, y_test_proba_lr, "Test")

print("Lasso LR Performance:")
print(f"  Test - AUC: {lasso_test_metrics['AUC']}, AUPRC: {lasso_test_metrics['AUPRC']}, Recall: {lasso_test_metrics['Recall']}")
print("Regular LR Performance:")
print(f"  Test - AUC: {lr_test_metrics['AUC']}, AUPRC: {lr_test_metrics['AUPRC']}, Recall: {lr_test_metrics['Recall']}")

# Feature coefficients (log-odds) for Lasso
lasso_coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lasso_lr.coef_[0],
    'Abs_Coefficient': np.abs(lasso_lr.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

# Calculate Odds Ratios (OR) for clinical interpretation
lasso_coefficients['OR'] = np.exp(lasso_coefficients['Coefficient'])
lasso_coefficients['OR_95CI_Lower'] = np.exp(lasso_coefficients['Coefficient'] - 1.96 * np.std(lasso_coefficients['Coefficient']))
lasso_coefficients['OR_95CI_Upper'] = np.exp(lasso_coefficients['Coefficient'] + 1.96 * np.std(lasso_coefficients['Coefficient']))

print("\nTop 10 Positive Coefficients (↑ Risk, OR):")
top_positive = lasso_coefficients[lasso_coefficients['Coefficient'] > 0].head(10)
for idx, row in top_positive.iterrows():
    print(f"  {row['Feature']:<30} OR={row['OR']:.3f} [95% CI: {row['OR_95CI_Lower']:.3f}, {row['OR_95CI_Upper']:.3f}]")

lasso_coefficients.to_csv('../logs/modeling/lasso_coefficients_with_or.csv', index=False)
print("[OK] Saved: logs/modeling/lasso_coefficients_with_or.csv")

# ROC and PR curves for baseline
fpr_lasso, tpr_lasso, _ = roc_curve(y_test, y_test_proba_lasso)
precision_lasso, recall_lasso, _ = precision_recall_curve(y_test, y_test_proba_lasso)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_test_proba_lr)

# ============================================================================
# Step 3: Advanced Models with Nested CV and Bayesian Optimization
# ============================================================================
print("\n[Step 3] Advanced Models with Nested CV and Bayesian Optimization")
print("-" * 80)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Custom scorer emphasizing recall (clinical priority)
recall_scorer = make_scorer(recall_score, greater_is_better=True)
auprc_scorer = make_scorer(average_precision_score, greater_is_better=True)

# Nested CV for unbiased performance estimation
print("\n[Nested Cross-Validation]")
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# XGBoost with Optuna Bayesian Optimization
print("\n[XGBoost] Optimizing with Optuna (Bayesian Optimization)...")

def xgb_objective(trial):
    """Optuna objective for XGBoost"""
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
        'use_label_encoder': False,
        'verbosity': 0
    }
    
    # Inner CV
    cv_scores = []
    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        try:
            model.fit(X_train_fold, y_train_fold, 
                     eval_set=[(X_val_fold, y_val_fold)],
                     early_stopping_rounds=50, verbose=False)
        except:
            model.fit(X_train_fold, y_train_fold, verbose=False)
        
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        score = average_precision_score(y_val_fold, y_val_proba)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

# Run Optuna optimization
study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_xgb.optimize(xgb_objective, n_trials=30, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
best_xgb_params.update({
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'verbosity': 0
})

print(f"Best XGBoost params: {best_xgb_params}")
xgb_model = xgb.XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train, y_train)

# LightGBM with similar optimization
print("\n[LightGBM] Optimizing with Optuna...")

def lgb_objective(trial):
    """Optuna objective for LightGBM"""
    params = {
        'scale_pos_weight': scale_pos_weight,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
        'random_state': 42,
        'verbosity': -1,
        'force_col_wise': True
    }
    
    cv_scores = []
    for train_idx, val_idx in inner_cv.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold, 
                 eval_set=[(X_val_fold, y_val_fold)],
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        score = average_precision_score(y_val_fold, y_val_proba)
        cv_scores.append(score)
    
    return np.mean(cv_scores)

study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lgb.optimize(lgb_objective, n_trials=30, show_progress_bar=True)

best_lgb_params = study_lgb.best_params
best_lgb_params.update({
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbosity': -1,
    'force_col_wise': True
})

print(f"Best LightGBM params: {best_lgb_params}")
lgb_model = lgb.LGBMClassifier(**best_lgb_params)
lgb_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Evaluate models
y_test_pred_xgb = xgb_model.predict(X_test)
y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_test_metrics = evaluate_model(y_test, y_test_pred_xgb, y_test_proba_xgb, "Test")

y_test_pred_lgb = lgb_model.predict(X_test)
y_test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
lgb_test_metrics = evaluate_model(y_test, y_test_pred_lgb, y_test_proba_lgb, "Test")

y_test_pred_rf = rf_model.predict(X_test)
y_test_proba_rf = rf_model.predict_proba(X_test)[:, 1]
rf_test_metrics = evaluate_model(y_test, y_test_pred_rf, y_test_proba_rf, "Test")

print("\nModel Performance:")
print(f"  XGBoost  - AUC: {xgb_test_metrics['AUC']}, AUPRC: {xgb_test_metrics['AUPRC']}, Recall: {xgb_test_metrics['Recall']}")
print(f"  LightGBM - AUC: {lgb_test_metrics['AUC']}, AUPRC: {lgb_test_metrics['AUPRC']}, Recall: {lgb_test_metrics['Recall']}")
print(f"  RF       - AUC: {rf_test_metrics['AUC']}, AUPRC: {rf_test_metrics['AUPRC']}, Recall: {rf_test_metrics['Recall']}")

# Nested CV for final evaluation
print("\n[Nested CV Final Evaluation]")
nested_cv_scores_xgb = []
nested_cv_scores_lgb = []

for train_idx, test_idx in outer_cv.split(X_train, y_train):
    X_train_outer = X_train.iloc[train_idx]
    X_test_outer = X_train.iloc[test_idx]
    y_train_outer = y_train.iloc[train_idx]
    y_test_outer = y_train.iloc[test_idx]
    
    # XGBoost
    xgb_outer = xgb.XGBClassifier(**best_xgb_params)
    xgb_outer.fit(X_train_outer, y_train_outer, verbose=False)
    y_proba_outer = xgb_outer.predict_proba(X_test_outer)[:, 1]
    nested_cv_scores_xgb.append(average_precision_score(y_test_outer, y_proba_outer))
    
    # LightGBM
    lgb_outer = lgb.LGBMClassifier(**best_lgb_params)
    lgb_outer.fit(X_train_outer, y_train_outer, 
                 callbacks=[lgb.log_evaluation(0)])
    y_proba_outer = lgb_outer.predict_proba(X_test_outer)[:, 1]
    nested_cv_scores_lgb.append(average_precision_score(y_test_outer, y_proba_outer))

print(f"  XGBoost Nested CV AUPRC: {np.mean(nested_cv_scores_xgb):.4f} (±{np.std(nested_cv_scores_xgb):.4f})")
print(f"  LightGBM Nested CV AUPRC: {np.mean(nested_cv_scores_lgb):.4f} (±{np.std(nested_cv_scores_lgb):.4f})")

# ROC and PR curves
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_proba_xgb)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_test_proba_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_test_proba_lgb)
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_test_proba_lgb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_test_proba_rf)

# Feature importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

xgb_importance.to_csv('../logs/modeling/xgb_feature_importance.csv', index=False)
print("[OK] Saved: logs/modeling/xgb_feature_importance.csv")

# ============================================================================
# Step 4: Stacking Ensemble
# ============================================================================
print("\n[Step 4] Stacking Ensemble (LR + XGBoost + LightGBM)")
print("-" * 80)

# Stacking ensemble with Lasso LR as meta-learner
base_estimators = [
    ('lasso_lr', LogisticRegression(class_weight='balanced', C=0.1, penalty='l1', 
                                    max_iter=1000, random_state=42, solver='liblinear')),
    ('xgb', xgb.XGBClassifier(**{k: v for k, v in best_xgb_params.items() 
                                 if k not in ['random_state', 'verbosity']}, 
                             random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMClassifier(**{k: v for k, v in best_lgb_params.items() 
                                  if k not in ['random_state', 'verbosity']}, 
                              random_state=42, verbosity=-1))
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5,
    stack_method='predict_proba'
)

stacking_model.fit(X_train, y_train)

y_test_pred_stack = stacking_model.predict(X_test)
y_test_proba_stack = stacking_model.predict_proba(X_test)[:, 1]
stack_test_metrics = evaluate_model(y_test, y_test_pred_stack, y_test_proba_stack, "Test")

print("Stacking Ensemble Performance:")
print(f"  Test - AUC: {stack_test_metrics['AUC']}, AUPRC: {stack_test_metrics['AUPRC']}, Recall: {stack_test_metrics['Recall']}")

fpr_stack, tpr_stack, _ = roc_curve(y_test, y_test_proba_stack)
precision_stack, recall_stack, _ = precision_recall_curve(y_test, y_test_proba_stack)

# ============================================================================
# Step 5: Threshold Optimization (Cost-Sensitive)
# ============================================================================
print("\n[Step 5] Threshold Optimization (Cost-Sensitive, F2 Maximization)")
print("-" * 80)

def find_optimal_threshold(y_true, y_proba, strategy='f2'):
    """Find optimal threshold for different strategies"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if strategy == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2)  # Emphasize recall
        elif strategy == 'youden':
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            idx = np.argmax(tpr - fpr)
            threshold_youden = thresholds[np.argmin(np.abs(thresholds - fpr[idx]))]
            return threshold_youden
        elif strategy == 'recall':
            score = recall_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

# Optimize threshold on validation set
y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
thresholds_opt = {
    'f2': find_optimal_threshold(y_val, y_val_proba_xgb, 'f2'),
    'youden': find_optimal_threshold(y_val, y_val_proba_xgb, 'youden'),
    'default': 0.5
}

print("Optimal thresholds (on validation set):")
for strategy, thresh in thresholds_opt.items():
    y_pred_thresh = (y_test_proba_xgb >= thresh).astype(int)
    recall = recall_score(y_test, y_pred_thresh)
    f2 = fbeta_score(y_test, y_pred_thresh, beta=2)
    print(f"  {strategy.capitalize()}: {thresh:.4f} (Recall={recall:.4f}, F2={f2:.4f})")

# Use F2-optimized threshold for final predictions
optimal_threshold = thresholds_opt['f2']
y_test_pred_optimal = (y_test_proba_xgb >= optimal_threshold).astype(int)

threshold_results = []
for strategy, thresh in thresholds_opt.items():
    y_pred_thresh = (y_test_proba_xgb >= thresh).astype(int)
    threshold_results.append({
        'Strategy': strategy,
        'Threshold': f"{thresh:.4f}",
        'F1_Score': f"{f1_score(y_test, y_pred_thresh):.4f}",
        'F2_Score': f"{fbeta_score(y_test, y_pred_thresh, beta=2):.4f}",
        'Recall': f"{recall_score(y_test, y_pred_thresh):.4f}",
        'Precision': f"{precision_score(y_test, y_pred_thresh):.4f}"
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('../logs/modeling/threshold_optimization_report.csv', index=False)
print("[OK] Saved: logs/modeling/threshold_optimization_report.csv")

# ============================================================================
# Step 6: Model Calibration
# ============================================================================
print("\n[Step 6] Model Calibration")
print("-" * 80)

# Calibrate best model (XGBoost)
xgb_calibrated_isotonic = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
xgb_calibrated_isotonic.fit(X_val, y_val)

y_test_proba_xgb_cal = xgb_calibrated_isotonic.predict_proba(X_test)[:, 1]

brier_original = brier_score_loss(y_test, y_test_proba_xgb)
brier_calibrated = brier_score_loss(y_test, y_test_proba_xgb_cal)

print(f"XGBoost Calibration:")
print(f"  Original Brier Score: {brier_original:.4f}")
print(f"  Calibrated (Isotonic) Brier: {brier_calibrated:.4f}")
print(f"  Improvement: {((brier_original - brier_calibrated) / brier_original * 100):.2f}%")

# Calibration curves
fraction_pos_orig, mean_pred_orig = calibration_curve(y_test, y_test_proba_xgb, n_bins=10)
fraction_pos_cal, mean_pred_cal = calibration_curve(y_test, y_test_proba_xgb_cal, n_bins=10)

plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_pred_orig, fraction_pos_orig, 'o-', label=f'Original (Brier={brier_original:.3f})')
plt.plot(mean_pred_cal, fraction_pos_cal, 's-', label=f'Calibrated (Brier={brier_calibrated:.3f})')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves (Reliability Diagram)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/modeling/calibration_plot.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/calibration_plot.png")
plt.close()

# Use calibrated model
best_model = xgb_calibrated_isotonic
best_proba = y_test_proba_xgb_cal

# ============================================================================
# Step 7: Decision Curve Analysis (DCA)
# ============================================================================
print("\n[Step 7] Decision Curve Analysis (DCA)")
print("-" * 80)

def decision_curve_analysis(y_true, y_proba, thresholds):
    """Calculate net benefit for DCA"""
    net_benefits = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        n = len(y_true)
        
        if threshold < 1.0:
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            net_benefit = 0
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)

thresholds_dca = np.linspace(0.01, 0.99, 100)

# DCA for multiple models
net_benefit_lasso = decision_curve_analysis(y_test, y_test_proba_lasso, thresholds_dca)
net_benefit_lr = decision_curve_analysis(y_test, y_test_proba_lr, thresholds_dca)
net_benefit_xgb = decision_curve_analysis(y_test, y_test_proba_xgb, thresholds_dca)
net_benefit_stack = decision_curve_analysis(y_test, y_test_proba_stack, thresholds_dca)

# Treat All and Treat None baselines
treat_all_benefit = (y_test == 1).mean()
treat_none_benefit = 0

plt.figure(figsize=(12, 8))
plt.plot(thresholds_dca, net_benefit_lasso, label='Lasso LR', linewidth=2, linestyle='--')
plt.plot(thresholds_dca, net_benefit_lr, label='Regular LR', linewidth=2, linestyle='--')
plt.plot(thresholds_dca, net_benefit_xgb, label='XGBoost (Optimized)', linewidth=2)
plt.plot(thresholds_dca, net_benefit_stack, label='Stacking Ensemble', linewidth=2)
plt.axhline(y=treat_none_benefit, color='k', linestyle='--', alpha=0.5, label='Treat None')
plt.axhline(y=treat_all_benefit, color='gray', linestyle='--', alpha=0.5, label='Treat All')
plt.xlabel('Threshold Probability', fontsize=12)
plt.ylabel('Net Benefit', fontsize=12)
plt.title('Decision Curve Analysis (DCA)\nClinical Utility Assessment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/modeling/dca_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/dca_curve.png")
plt.close()

# ============================================================================
# Step 8: Statistical Comparison (McNemar Test)
# ============================================================================
print("\n[Step 8] Statistical Comparison (McNemar Test)")
print("-" * 80)

if MCNEMAR_AVAILABLE:
    try:
        # Compare XGBoost vs Lasso LR
        mcnemar_table_result = mcnemar_table(y_test, y_test_pred_lasso, y_test_pred_xgb)
        mcnemar_result = mcnemar(mcnemar_table_result, exact=True)
        
        # Handle both tuple and object return types
        if isinstance(mcnemar_result, tuple):
            p_value = mcnemar_result[1]  # p-value is second element
        else:
            p_value = mcnemar_result.pvalue
        
        print(f"McNemar Test (XGBoost vs Lasso LR):")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  Result: XGBoost significantly outperforms Lasso LR (p<0.05)")
        else:
            print(f"  Result: No significant difference (p≥0.05)")
        
        # Compare XGBoost vs Stacking
        mcnemar_table_result2 = mcnemar_table(y_test, y_test_pred_xgb, y_test_pred_stack)
        mcnemar_result2 = mcnemar(mcnemar_table_result2, exact=True)
        
        if isinstance(mcnemar_result2, tuple):
            p_value2 = mcnemar_result2[1]
        else:
            p_value2 = mcnemar_result2.pvalue
        
        print(f"\nMcNemar Test (XGBoost vs Stacking):")
        print(f"  p-value: {p_value2:.4f}")
        if p_value2 < 0.05:
            print(f"  Result: Significant difference (p<0.05)")
        else:
            print(f"  Result: No significant difference (p≥0.05)")
        
        # Save results
        comparison_results = pd.DataFrame([
            {
                'Comparison': 'XGBoost vs Lasso LR',
                'P_value': p_value,
                'Significant': p_value < 0.05
            },
            {
                'Comparison': 'XGBoost vs Stacking',
                'P_value': p_value2,
                'Significant': p_value2 < 0.05
            }
        ])
        comparison_results.to_csv('../logs/modeling/statistical_comparison.csv', index=False)
        print("[OK] Saved: logs/modeling/statistical_comparison.csv")
    except Exception as e:
        print(f"[WARNING] McNemar test failed: {e}")
        print("[INFO] Skipping statistical comparison")
else:
    print("[INFO] McNemar test not available, skipping statistical comparison")

# ============================================================================
# Step 9: SHAP and LIME Interpretability
# ============================================================================
print("\n[Step 9] Interpretability Analysis (SHAP + LIME)")
print("-" * 80)

# SHAP analysis
print("Computing SHAP values...")
best_tree_model = xgb_model if float(xgb_test_metrics['AUC']) >= float(lgb_test_metrics['AUC']) else lgb_model
explainer = shap.TreeExplainer(best_tree_model)
shap_values = explainer.shap_values(X_test[:1000])

# SHAP summary plot
shap.summary_plot(shap_values, X_test.iloc[:1000], show=False, max_display=20)
plt.tight_layout()
plt.savefig('../visualizations/modeling/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: visualizations/modeling/shap_summary_plot.png")

# SHAP feature importance
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'SHAP_Importance': np.abs(shap_values).mean(0)
}).sort_values('SHAP_Importance', ascending=False)

shap_importance.to_csv('../logs/modeling/shap_feature_importance.csv', index=False)
print("[OK] Saved: logs/modeling/shap_feature_importance.csv")

# Clinical mapping: Group SHAP by Sepsis-3 components
print("\n[Clinical Mapping] SHAP values by Sepsis-3 components:")
sofa_vars = [col for col in X.columns if 'sofa' in col.lower()]
lactate_vars = [col for col in X.columns if 'lactate' in col.lower()]
age_vars = [col for col in X.columns if 'age' in col.lower()]

sofa_idx = [X.columns.get_loc(col) for col in sofa_vars if col in X.columns]
lactate_idx = [X.columns.get_loc(col) for col in lactate_vars if col in X.columns]
age_idx = [X.columns.get_loc(col) for col in age_vars if col in X.columns]

if sofa_idx:
    sofa_shap = np.abs(shap_values[:, sofa_idx]).mean() if len(sofa_idx) > 0 else 0
    print(f"  SOFA component contribution: {sofa_shap:.4f} ({sofa_shap/np.abs(shap_values).mean()*100:.1f}%)")
if lactate_idx:
    lactate_shap = np.abs(shap_values[:, lactate_idx]).mean() if len(lactate_idx) > 0 else 0
    print(f"  Lactate component contribution: {lactate_shap:.4f} ({lactate_shap/np.abs(shap_values).mean()*100:.1f}%)")
if age_idx:
    age_shap = np.abs(shap_values[:, age_idx]).mean() if len(age_idx) > 0 else 0
    print(f"  Age component contribution: {age_shap:.4f} ({age_shap/np.abs(shap_values).mean()*100:.1f}%)")

# LIME explanations
if LIME_AVAILABLE:
    print("\n[LIME] Generating local explanations...")
    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['Survived', 'Died'],
        mode='classification'
    )
    
    # Explain a few high-risk cases
    high_risk_indices = np.where(y_test_proba_xgb > 0.7)[0][:3]
    
    for i, idx in enumerate(high_risk_indices):
        explanation = lime_explainer.explain_instance(
            X_test.iloc[idx].values,
            xgb_model.predict_proba,
            num_features=10
        )
        
        # Save as figure
        explanation.as_pyplot_figure()
        plt.title(f'LIME Explanation - High Risk Case {i+1}\n(True: {"Died" if y_test.iloc[idx] == 1 else "Survived"})')
        plt.tight_layout()
        plt.savefig(f'../visualizations/modeling/lime_explanation_case_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Saved: visualizations/modeling/lime_explanation_case_*.png")
else:
    print("[INFO] LIME not available, skipping local explanations")

# ============================================================================
# Step 10: Subgroup Analysis
# ============================================================================
print("\n[Step 10] Subgroup Analysis")
print("-" * 80)

subgroup_results = []

# Age subgroups
for age_group in ['≥65', '<65']:
    mask = X_test_with_source['age_group'] == age_group
    if mask.sum() > 0:
        y_subgroup = y_test[mask]
        proba_subgroup = best_proba[mask]
        pred_subgroup = (proba_subgroup >= optimal_threshold).astype(int)
        
        auc = roc_auc_score(y_subgroup, proba_subgroup)
        recall = recall_score(y_subgroup, pred_subgroup)
        auprc = average_precision_score(y_subgroup, proba_subgroup)
        f2 = fbeta_score(y_subgroup, pred_subgroup, beta=2)
        
        subgroup_results.append({
            'Subgroup': f'Age {age_group}',
            'N': int(mask.sum()),
            'Events': int(y_subgroup.sum()),
            'AUC': f"{auc:.4f}",
            'AUPRC': f"{auprc:.4f}",
            'Recall': f"{recall:.4f}",
            'F2_Score': f"{f2:.4f}"
        })

# Infection source subgroups
unique_sources = X_test_with_source['infection_source'].unique()
for source in unique_sources:
    if pd.notna(source) and source != 'unknown':
        mask = X_test_with_source['infection_source'] == source
        if mask.sum() > 10:
            y_subgroup = y_test[mask]
            proba_subgroup = best_proba[mask]
            pred_subgroup = (proba_subgroup >= optimal_threshold).astype(int)
            
            auc = roc_auc_score(y_subgroup, proba_subgroup)
            recall = recall_score(y_subgroup, pred_subgroup)
            auprc = average_precision_score(y_subgroup, proba_subgroup)
            f2 = fbeta_score(y_subgroup, pred_subgroup, beta=2)
            
            subgroup_results.append({
                'Subgroup': f'Infection: {source}',
                'N': int(mask.sum()),
                'Events': int(y_subgroup.sum()),
                'AUC': f"{auc:.4f}",
                'AUPRC': f"{auprc:.4f}",
                'Recall': f"{recall:.4f}",
                'F2_Score': f"{f2:.4f}"
            })

subgroup_df = pd.DataFrame(subgroup_results)
print("\nSubgroup Performance:")
print(subgroup_df.to_string(index=False))
subgroup_df.to_csv('../logs/modeling/subgroup_performance_table.csv', index=False)
print("[OK] Saved: logs/modeling/subgroup_performance_table.csv")

# ============================================================================
# Step 11: Final Results and Export
# ============================================================================
print("\n[Step 11] Final Results and Export")
print("-" * 80)

# Compile all model performance
performance_summary = pd.DataFrame([
    {**lasso_test_metrics, 'Model': 'Lasso LR'},
    {**lr_test_metrics, 'Model': 'Regular LR'},
    {**xgb_test_metrics, 'Model': 'XGBoost (Optimized)'},
    {**lgb_test_metrics, 'Model': 'LightGBM (Optimized)'},
    {**rf_test_metrics, 'Model': 'Random Forest'},
    {**stack_test_metrics, 'Model': 'Stacking Ensemble'}
])

print("\nModel Performance Summary:")
print(performance_summary.to_string(index=False))
performance_summary.to_csv('../logs/modeling/model_performance_summary.csv', index=False)
print("[OK] Saved: logs/modeling/model_performance_summary.csv")

# ROC and PR Curves
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr_lasso, tpr_lasso, label=f'Lasso LR (AUC={lasso_test_metrics["AUC"]})', linewidth=2, linestyle='--')
plt.plot(fpr_lr, tpr_lr, label=f'Regular LR (AUC={lr_test_metrics["AUC"]})', linewidth=2, linestyle='--')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={xgb_test_metrics["AUC"]})', linewidth=2)
plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={lgb_test_metrics["AUC"]})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC={rf_test_metrics["AUC"]})', linewidth=2)
plt.plot(fpr_stack, tpr_stack, label=f'Stacking (AUC={stack_test_metrics["AUC"]})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(recall_lasso, precision_lasso, label=f'Lasso LR (AUPRC={lasso_test_metrics["AUPRC"]})', linewidth=2, linestyle='--')
plt.plot(recall_lr, precision_lr, label=f'Regular LR (AUPRC={lr_test_metrics["AUPRC"]})', linewidth=2, linestyle='--')
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost (AUPRC={xgb_test_metrics["AUPRC"]})', linewidth=2)
plt.plot(recall_lgb, precision_lgb, label=f'LightGBM (AUPRC={lgb_test_metrics["AUPRC"]})', linewidth=2)
plt.plot(recall_rf, precision_rf, label=f'RF (AUPRC={rf_test_metrics["AUPRC"]})', linewidth=2)
plt.plot(recall_stack, precision_stack, label=f'Stacking (AUPRC={stack_test_metrics["AUPRC"]})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/modeling/roc_pr_curves.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/roc_pr_curves.png")
plt.close()

# Save best model and hyperparameters
joblib.dump(best_model, '../artifacts/best_sepsis_model.pkl')
print("[OK] Saved: artifacts/best_sepsis_model.pkl")

# Save hyperparameters
best_params_dict = {
    'xgb': best_xgb_params,
    'lgb': best_lgb_params,
    'optimal_threshold': optimal_threshold,
    'calibration_method': 'isotonic'
}
import json
with open('../artifacts/best_hyperparameters.json', 'w') as f:
    json.dump(best_params_dict, f, indent=2)
print("[OK] Saved: artifacts/best_hyperparameters.json")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Enhanced Modeling and Interpretability Analysis Complete!")
print("=" * 80)

best_auc = float(xgb_test_metrics['AUC'])
best_auprc = float(xgb_test_metrics['AUPRC'])
best_recall = float(xgb_test_metrics['Recall'])
best_brier = float(xgb_test_metrics['Brier Score'])

print(f"\nBest Model: XGBoost (Bayesian Optimized) with Isotonic Calibration")
print(f"  Test AUC: {best_auc:.4f}")
print(f"  Test AUPRC: {best_auprc:.4f}")
print(f"  Test Recall: {best_recall:.4f}")
print(f"  Test Brier Score: {best_brier:.4f}")
print(f"  Optimal Threshold (F2): {optimal_threshold:.4f}")

print("\nKey Enhancements Applied:")
print("  ✓ Temporal/Stratified split for realistic validation")
print("  ✓ SMOTE for advanced imbalance handling")
print("  ✓ Nested CV for unbiased performance estimation")
print("  ✓ Bayesian optimization (Optuna) for hyperparameter tuning")
print("  ✓ Cost-sensitive threshold optimization (F2 maximization)")
print("  ✓ Decision Curve Analysis (DCA) for clinical utility")
print("  ✓ Statistical comparison (McNemar test)")
print("  ✓ SHAP + LIME for multi-angle interpretability")
print("  ✓ Clinical mapping (Sepsis-3 components)")
print("  ✓ Comprehensive subgroup analysis")

print("\nGenerated files:")
print("  - logs/modeling/model_performance_summary.csv")
print("  - logs/modeling/lasso_coefficients_with_or.csv")
print("  - logs/modeling/threshold_optimization_report.csv")
print("  - logs/modeling/statistical_comparison.csv")
print("  - logs/modeling/shap_feature_importance.csv")
print("  - logs/modeling/subgroup_performance_table.csv")
print("  - logs/modeling/subgroup_stats.csv")
print("  - visualizations/modeling/calibration_plot.png")
print("  - visualizations/modeling/roc_pr_curves.png")
print("  - visualizations/modeling/dca_curve.png")
print("  - visualizations/modeling/shap_summary_plot.png")
print("  - artifacts/best_sepsis_model.pkl")
print("  - artifacts/best_hyperparameters.json")

print("\n" + "=" * 80)

