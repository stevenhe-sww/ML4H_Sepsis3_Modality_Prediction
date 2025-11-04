"""
Step 3 - Modeling and Interpretability Analysis
Complete modeling pipeline for Sepsis-3 30-day mortality prediction
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
    recall_score, f1_score, brier_score_loss, roc_curve, 
    precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("Step 3 - Modeling and Interpretability Analysis")
print("=" * 80)

# ============================================================================
# Step 1: Data Preparation
# ============================================================================
print("\n[Step 1] Data Preparation")
print("-" * 80)

# Load cleaned data
data_path = Path("../data/processed/sepsis3_cleaned.csv")
df = pd.read_csv(data_path)

print(f"Loaded data: {df.shape[0]} rows * {df.shape[1]} columns")

# Remove ID columns
id_cols = ['subject_id', 'hadm_id', 'stay_id']
id_cols_present = [col for col in id_cols if col in df.columns]
if id_cols_present:
    df = df.drop(columns=id_cols_present)
    print(f"Removed ID columns: {id_cols_present}")

# Separate features and target
target_col = 'mortality_30d'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Features: {X.shape[1]} variables")
print(f"Target distribution:")
print(f"  Survived (0): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
print(f"  Died (1): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")

# Stratified train/validation/test split (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp  # 0.1765 ≈ 15/85
)

print(f"\nData split:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"  Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# ============================================================================
# Step 2: Baseline Model (Logistic Regression)
# ============================================================================
print("\n[Step 2] Baseline Model (Logistic Regression)")
print("-" * 80)

# Logistic Regression with L2 regularization
lr_model = LogisticRegression(
    class_weight='balanced',
    C=0.5,
    penalty='l2',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

lr_model.fit(X_train, y_train)

# Predictions
y_train_pred_lr = lr_model.predict(X_train)
y_val_pred_lr = lr_model.predict(X_val)
y_test_pred_lr = lr_model.predict(X_test)
y_train_proba_lr = lr_model.predict_proba(X_train)[:, 1]
y_val_proba_lr = lr_model.predict_proba(X_val)[:, 1]
y_test_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
def evaluate_model(y_true, y_pred, y_proba, set_name=""):
    """Calculate comprehensive evaluation metrics"""
    auc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)
    
    results = {
        'Set': set_name,
        'AUC': f"{auc:.4f}",
        'AUPRC': f"{auprc:.4f}",
        'Accuracy': f"{accuracy:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}",
        'Brier Score': f"{brier:.4f}"
    }
    return results

lr_train_metrics = evaluate_model(y_train, y_train_pred_lr, y_train_proba_lr, "Train")
lr_val_metrics = evaluate_model(y_val, y_val_pred_lr, y_val_proba_lr, "Validation")
lr_test_metrics = evaluate_model(y_test, y_test_pred_lr, y_test_proba_lr, "Test")

print("Logistic Regression Performance:")
print(f"  Train - AUC: {lr_train_metrics['AUC']}, AUPRC: {lr_train_metrics['AUPRC']}, Brier: {lr_train_metrics['Brier Score']}")
print(f"  Val   - AUC: {lr_val_metrics['AUC']}, AUPRC: {lr_val_metrics['AUPRC']}, Brier: {lr_val_metrics['Brier Score']}")
print(f"  Test  - AUC: {lr_test_metrics['AUC']}, AUPRC: {lr_test_metrics['AUPRC']}, Brier: {lr_test_metrics['Brier Score']}")

# Feature coefficients (log-odds)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Positive Coefficients (↑ Risk):")
print(coefficients.head(10).to_string(index=False))
print("\nTop 10 Negative Coefficients (↓ Risk):")
print(coefficients.tail(10).sort_values('Coefficient').to_string(index=False))

coefficients.to_csv('../logs/modeling/lr_coefficients.csv', index=False)

# ROC and PR curves for baseline
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_test_proba_lr)

# ============================================================================
# Step 3: Advanced Models (XGBoost, LightGBM, RandomForest)
# ============================================================================
print("\n[Step 3] Advanced Models (XGBoost, LightGBM, RandomForest)")
print("-" * 80)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
    force_col_wise=True
)

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

# Evaluate XGBoost
y_test_pred_xgb = xgb_model.predict(X_test)
y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
xgb_test_metrics = evaluate_model(y_test, y_test_pred_xgb, y_test_proba_xgb, "Test")

# Evaluate LightGBM
y_test_pred_lgb = lgb_model.predict(X_test)
y_test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
lgb_test_metrics = evaluate_model(y_test, y_test_pred_lgb, y_test_proba_lgb, "Test")

# Evaluate Random Forest
y_test_pred_rf = rf_model.predict(X_test)
y_test_proba_rf = rf_model.predict_proba(X_test)[:, 1]
rf_test_metrics = evaluate_model(y_test, y_test_pred_rf, y_test_proba_rf, "Test")

print("XGBoost Performance:")
print(f"  Test - AUC: {xgb_test_metrics['AUC']}, AUPRC: {xgb_test_metrics['AUPRC']}, Brier: {xgb_test_metrics['Brier Score']}")
print("LightGBM Performance:")
print(f"  Test - AUC: {lgb_test_metrics['AUC']}, AUPRC: {lgb_test_metrics['AUPRC']}, Brier: {lgb_test_metrics['Brier Score']}")
print("Random Forest Performance:")
print(f"  Test - AUC: {rf_test_metrics['AUC']}, AUPRC: {rf_test_metrics['AUPRC']}, Brier: {rf_test_metrics['Brier Score']}")

# 5-fold Cross-Validation
print("\n5-fold Cross-Validation:")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_cv_auc = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
xgb_cv_auprc = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)

lgb_cv_auc = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
lgb_cv_auprc = cross_val_score(lgb_model, X_train, y_train, cv=cv, scoring='average_precision', n_jobs=-1)

print(f"  XGBoost CV AUC: {xgb_cv_auc.mean():.4f} (±{xgb_cv_auc.std():.4f})")
print(f"  XGBoost CV AUPRC: {xgb_cv_auprc.mean():.4f} (±{xgb_cv_auprc.std():.4f})")
print(f"  LightGBM CV AUC: {lgb_cv_auc.mean():.4f} (±{lgb_cv_auc.std():.4f})")
print(f"  LightGBM CV AUPRC: {lgb_cv_auprc.mean():.4f} (±{lgb_cv_auprc.std():.4f})")

# Feature importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 XGBoost Feature Importance:")
print(xgb_importance.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 8))
top_features = xgb_importance.head(20)
plt.barh(range(len(top_features)), top_features['Importance'].values[::-1])
plt.yticks(range(len(top_features)), top_features['Feature'].values[::-1], fontsize=9)
plt.xlabel('Feature Importance')
plt.title('XGBoost Top 20 Feature Importance', fontsize=14)
plt.tight_layout()
plt.savefig('../visualizations/modeling/xgb_feature_importance.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/xgb_feature_importance.png")
plt.close()

# ROC and PR curves for advanced models
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_proba_xgb)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_test_proba_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_test_proba_lgb)
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, y_test_proba_lgb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_test_proba_rf)

# ============================================================================
# Step 4: Model Ensemble (Stacking)
# ============================================================================
print("\n[Step 4] Model Ensemble (Stacking)")
print("-" * 80)

# Stacking ensemble (LR, XGBoost, LightGBM)
base_estimators = [
    ('lr', LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000, random_state=42)),
    ('xgb', xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, learning_rate=0.05, max_depth=5, 
                              n_estimators=300, random_state=42, eval_metric='logloss', use_label_encoder=False)),
    ('lgb', lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, learning_rate=0.05, max_depth=5,
                               n_estimators=300, random_state=42, verbosity=-1, force_col_wise=True))
]

stacking_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5
)

stacking_model.fit(X_train, y_train)

y_test_pred_stack = stacking_model.predict(X_test)
y_test_proba_stack = stacking_model.predict_proba(X_test)[:, 1]
stack_test_metrics = evaluate_model(y_test, y_test_pred_stack, y_test_proba_stack, "Test")

print("Stacking Ensemble Performance:")
print(f"  Test - AUC: {stack_test_metrics['AUC']}, AUPRC: {stack_test_metrics['AUPRC']}, Brier: {stack_test_metrics['Brier Score']}")

fpr_stack, tpr_stack, _ = roc_curve(y_test, y_test_proba_stack)
precision_stack, recall_stack, _ = precision_recall_curve(y_test, y_test_proba_stack)

# ============================================================================
# Step 5: Model Calibration
# ============================================================================
print("\n[Step 5] Model Calibration")
print("-" * 80)

# Calibrate XGBoost (best model)
xgb_calibrated_platt = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
xgb_calibrated_platt.fit(X_val, y_val)

xgb_calibrated_isotonic = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
xgb_calibrated_isotonic.fit(X_val, y_val)

# Calibrated predictions
y_test_proba_xgb_platt = xgb_calibrated_platt.predict_proba(X_test)[:, 1]
y_test_proba_xgb_isotonic = xgb_calibrated_isotonic.predict_proba(X_test)[:, 1]

# Brier scores
brier_original = brier_score_loss(y_test, y_test_proba_xgb)
brier_platt = brier_score_loss(y_test, y_test_proba_xgb_platt)
brier_isotonic = brier_score_loss(y_test, y_test_proba_xgb_isotonic)

print(f"XGBoost Calibration:")
print(f"  Original Brier Score: {brier_original:.4f}")
print(f"  Platt Scaling Brier: {brier_platt:.4f}")
print(f"  Isotonic Brier: {brier_isotonic:.4f}")

# Calibration curves
fraction_of_positives_orig, mean_predicted_value_orig = calibration_curve(
    y_test, y_test_proba_xgb, n_bins=10
)
fraction_of_positives_platt, mean_predicted_value_platt = calibration_curve(
    y_test, y_test_proba_xgb_platt, n_bins=10
)
fraction_of_positives_isotonic, mean_predicted_value_isotonic = calibration_curve(
    y_test, y_test_proba_xgb_isotonic, n_bins=10
)

# Plot calibration curve
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_predicted_value_orig, fraction_of_positives_orig, 'o-', label=f'Original (Brier={brier_original:.3f})')
plt.plot(mean_predicted_value_platt, fraction_of_positives_platt, 's-', label=f'Platt Scaling (Brier={brier_platt:.3f})')
plt.plot(mean_predicted_value_isotonic, fraction_of_positives_isotonic, '^-', label=f'Isotonic (Brier={brier_isotonic:.3f})')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curves (Reliability Diagram)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/modeling/calibration_plot.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/modeling/calibration_plot.png")
plt.close()

# Use best calibrated model
if brier_isotonic < brier_platt:
    best_model = xgb_calibrated_isotonic
    best_proba = y_test_proba_xgb_isotonic
    calibration_method = "Isotonic"
else:
    best_model = xgb_calibrated_platt
    best_proba = y_test_proba_xgb_platt
    calibration_method = "Platt"

# ============================================================================
# Step 6: SHAP Interpretability Analysis
# ============================================================================
print("\n[Step 6] SHAP Interpretability Analysis")
print("-" * 80)

# SHAP analysis on XGBoost (use uncalibrated for SHAP)
print("Computing SHAP values (this may take a few minutes)...")

# Use TreeExplainer for XGBoost (best model typically)
# You can change to lgb_model or xgb_model based on which performs better
best_tree_model = xgb_model if xgb_test_metrics['AUC'] >= lgb_test_metrics['AUC'] else lgb_model
explainer = shap.TreeExplainer(best_tree_model)
shap_values = explainer.shap_values(X_test[:1000])  # Sample for speed

# Summary plot
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

print("\nTop 20 SHAP Feature Importance:")
print(shap_importance.head(20).to_string(index=False))

# Dependence plots for top features
top_shap_features = shap_importance.head(5)['Feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, feature in enumerate(top_shap_features[:6]):
    shap.dependence_plot(
        feature, shap_values, X_test.iloc[:1000], 
        ax=axes[idx], show=False
    )
    axes[idx].set_title(f'{feature} SHAP Dependence', fontsize=10)

plt.tight_layout()
plt.savefig('../visualizations/modeling/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: visualizations/modeling/shap_dependence_plots.png")

# Save SHAP importance
shap_importance.to_csv('../logs/modeling/shap_feature_importance.csv', index=False)
print("[OK] Saved: logs/modeling/shap_feature_importance.csv")

# ============================================================================
# Step 7: Subgroup Analysis
# ============================================================================
print("\n[Step 7] Subgroup Analysis")
print("-" * 80)

# Reconstruct original infection source for subgroup analysis
# Check if infection_source columns exist
infection_cols = [col for col in X.columns if 'infection_source' in col]
if infection_cols:
    # Create infection_source variable
    X_test_with_source = X_test.copy()
    source_mapping = {}
    for col in infection_cols:
        source_name = col.replace('infection_source_', '')
        source_mapping[col] = source_name
    
    # Determine primary infection source
    X_test_with_source['infection_source'] = X_test_with_source[infection_cols].idxmax(axis=1)
    X_test_with_source['infection_source'] = X_test_with_source['infection_source'].map(source_mapping)
else:
    X_test_with_source = X_test.copy()
    X_test_with_source['infection_source'] = 'unknown'

# Age subgroup
X_test_with_source['age_group'] = X_test_with_source['age'].apply(
    lambda x: '≥65' if x >= 65 else '<65'
)

subgroup_results = []

# Age subgroups
for age_group in ['≥65', '<65']:
    mask = X_test_with_source['age_group'] == age_group
    if mask.sum() > 0:
        y_subgroup = y_test[mask]
        proba_subgroup = best_proba[mask]
        pred_subgroup = (proba_subgroup >= 0.5).astype(int)
        
        auc = roc_auc_score(y_subgroup, proba_subgroup)
        recall = recall_score(y_subgroup, pred_subgroup)
        auprc = average_precision_score(y_subgroup, proba_subgroup)
        
        subgroup_results.append({
            'Subgroup': f'Age {age_group}',
            'N': int(mask.sum()),
            'Events': int(y_subgroup.sum()),
            'AUC': f"{auc:.4f}",
            'AUPRC': f"{auprc:.4f}",
            'Recall': f"{recall:.4f}"
        })

# Infection source subgroups
unique_sources = X_test_with_source['infection_source'].unique()
for source in unique_sources:
    if pd.notna(source) and source != 'unknown':
        mask = X_test_with_source['infection_source'] == source
        if mask.sum() > 10:  # Minimum sample size
            y_subgroup = y_test[mask]
            proba_subgroup = best_proba[mask]
            pred_subgroup = (proba_subgroup >= 0.5).astype(int)
            
            auc = roc_auc_score(y_subgroup, proba_subgroup)
            recall = recall_score(y_subgroup, pred_subgroup)
            auprc = average_precision_score(y_subgroup, proba_subgroup)
            
            subgroup_results.append({
                'Subgroup': f'Infection: {source}',
                'N': int(mask.sum()),
                'Events': int(y_subgroup.sum()),
                'AUC': f"{auc:.4f}",
                'AUPRC': f"{auprc:.4f}",
                'Recall': f"{recall:.4f}"
            })

subgroup_df = pd.DataFrame(subgroup_results)
print("\nSubgroup Performance:")
print(subgroup_df.to_string(index=False))
subgroup_df.to_csv('../logs/modeling/subgroup_performance_table.csv', index=False)
print("[OK] Saved: logs/modeling/subgroup_performance_table.csv")

# ============================================================================
# Step 8: Final Results and Export
# ============================================================================
print("\n[Step 8] Final Results and Export")
print("-" * 80)

# Compile all model performance
performance_summary = pd.DataFrame([
    {**lr_test_metrics, 'Model': 'Logistic Regression'},
    {**xgb_test_metrics, 'Model': 'XGBoost'},
    {**lgb_test_metrics, 'Model': 'LightGBM'},
    {**rf_test_metrics, 'Model': 'Random Forest'},
    {**stack_test_metrics, 'Model': 'Stacking Ensemble'}
])

print("\nModel Performance Summary:")
print(performance_summary.to_string(index=False))
performance_summary.to_csv('../logs/modeling/model_performance_summary.csv', index=False)
print("[OK] Saved: logs/modeling/model_performance_summary.csv")

# ROC and PR Curves
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC={lr_test_metrics["AUC"]})', linewidth=2)
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
plt.plot(recall_lr, precision_lr, label=f'LR (AUPRC={lr_test_metrics["AUPRC"]})', linewidth=2)
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

# Save best model
joblib.dump(best_model, '../artifacts/best_sepsis_model.pkl')
print("[OK] Saved: artifacts/best_sepsis_model.pkl")

print("\n" + "=" * 80)
print("[COMPLETE] Modeling and Interpretability Analysis Complete!")
print("=" * 80)
print(f"\nBest Model: XGBoost with {calibration_method} Calibration")
print(f"  Test AUC: {roc_auc_score(y_test, best_proba):.4f}")
print(f"  Test AUPRC: {average_precision_score(y_test, best_proba):.4f}")
print(f"  Test Brier Score: {brier_score_loss(y_test, best_proba):.4f}")

print("\nGenerated files:")
print("  - logs/modeling/model_performance_summary.csv")
print("  - logs/modeling/shap_feature_importance.csv")
print("  - logs/modeling/subgroup_performance_table.csv")
print("  - logs/modeling/lr_coefficients.csv")
print("  - visualizations/modeling/calibration_plot.png")
print("  - visualizations/modeling/roc_pr_curves.png")
print("  - visualizations/modeling/xgb_feature_importance.png")
print("  - visualizations/modeling/shap_summary_plot.png")
print("  - visualizations/modeling/shap_dependence_plots.png")
print("  - artifacts/best_sepsis_model.pkl")

