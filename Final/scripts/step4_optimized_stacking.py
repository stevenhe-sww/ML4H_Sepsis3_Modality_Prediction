"""
Step 4 - Optimized Stacking Ensemble for Sepsis-3 Mortality Prediction
优化版Stacking集成模型：增强多样性、概率校准、阈值优化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    recall_score, f1_score, brier_score_loss, roc_curve,
    precision_recall_curve, fbeta_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
# Note: SMOTE can be optionally used for severe class imbalance (not used in current version)
# from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import joblib
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("Step 4 - Optimized Stacking Ensemble for Sepsis-3 Mortality Prediction")
print("=" * 80)

# ============================================================================
# 数据加载和准备
# ============================================================================
print("\n[Step 1] Data Loading and Preparation")
print("-" * 80)

# 加载清洗后的数据（支持从scripts目录或项目根目录运行）
data_path = Path("../data/processed/sepsis3_cleaned.csv")
if not data_path.exists():
    data_path = Path("data/processed/sepsis3_cleaned.csv")
df = pd.read_csv(data_path)

# 移除ID列
id_cols = ['subject_id', 'hadm_id', 'stay_id']
id_cols_present = [col for col in id_cols if col in df.columns]
if id_cols_present:
    df = df.drop(columns=id_cols_present)

target_col = 'mortality_30d'
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"数据形状: {X.shape[0]} 样本 × {X.shape[1]} 特征")
print(f"目标分布: {(y==1).sum()} 阳性 ({(y==1).mean()*100:.2f}%)")

# 分层划分数据 (70/15/15)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  验证集: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"  测试集: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# 计算类别权重
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  类别权重比例: {scale_pos_weight:.2f}:1")

# ============================================================================
# 评估函数
# ============================================================================
def evaluate_model(y_true, y_pred, y_proba, set_name: str = ""):
    """综合模型评估指标"""
    return {
        'Set': set_name,
        'AUC': roc_auc_score(y_true, y_proba),
        'AUPRC': average_precision_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'F2': fbeta_score(y_true, y_pred, beta=2),
        'Brier': brier_score_loss(y_true, y_proba)
    }

# ============================================================================
# 改进1: 多样性增强的Base Learners (LR + XGB + LGB + RF)
# ============================================================================
print("\n[Step 2] Building Enhanced Base Learners")
print("-" * 80)

# 定义Base Estimators（增强多样性）
print("构建Base Learners:")
base_estimators = []

# 1. Logistic Regression (线性模型，带预处理)
lr_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(
        class_weight='balanced',
        C=0.5,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
)
base_estimators.append(('lr', lr_pipeline))
print("  [OK] Logistic Regression (带预处理pipeline)")

# 2. XGBoost (梯度提升树)
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0
)
base_estimators.append(('xgb', xgb_model))
print("  [OK] XGBoost")

# 3. LightGBM (梯度提升树，不同算法)
lgb_model = lgb.LGBMClassifier(
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
base_estimators.append(('lgb', lgb_model))
print("  [OK] LightGBM")

# 4. Random Forest (Bagging机制，增加多样性)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
base_estimators.append(('rf', rf_model))
print("  [OK] Random Forest (Bagging多样性)")

print(f"\n总计 {len(base_estimators)} 个Base Learners: LR + XGBoost + LightGBM + RandomForest")

# ============================================================================
# 改进2: Meta Learner优化（比较不同选项）
# ============================================================================
print("\n[Step 3] Meta Learner Optimization")
print("-" * 80)

# 尝试不同的Meta Learner（移除RidgeClassifier，因为它不适合概率输出）
meta_learners = {
    'LR': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'LR_C0.1': LogisticRegression(class_weight='balanced', C=0.1, random_state=42, max_iter=1000),
    'LR_C2.0': LogisticRegression(class_weight='balanced', C=2.0, random_state=42, max_iter=1000),
    'LR_C0.5': LogisticRegression(class_weight='balanced', C=0.5, random_state=42, max_iter=1000)
}

meta_scores = {}

print("比较不同Meta Learner性能（使用5折CV）:")
with tqdm(total=len(meta_learners), desc="[Meta] Testing meta learners", unit="learner") as pbar:
    for meta_name, meta_learner in meta_learners.items():
        # 使用stack_method='predict_proba'获取概率
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            passthrough=False,  # 先测试不包含原始特征
            n_jobs=-1
        )
        
        # 5折交叉验证评估
        cv_scores = cross_val_score(
            stack, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        
        meta_scores[meta_name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'learner': meta_learner
        }
        
        print(f"  {meta_name:>10}: AUC = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        pbar.update(1)

# 选择最佳Meta Learner
best_meta_name = max(meta_scores.keys(), key=lambda x: meta_scores[x]['mean'])
best_meta_learner = meta_scores[best_meta_name]['learner']
print(f"\n[OK] 最佳Meta Learner: {best_meta_name} (AUC={meta_scores[best_meta_name]['mean']:.4f})")

# 测试passthrough=True的效果
print("\n测试passthrough=True/False差异:")
for passthrough in [False, True]:
    stack_test = StackingClassifier(
        estimators=base_estimators,
        final_estimator=best_meta_learner,
        cv=5,
        stack_method='predict_proba',
        passthrough=passthrough,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(
        stack_test, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1
    )
    
    print(f"  passthrough={passthrough}: AUC = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# 选择passthrough设置（选择更好的）
use_passthrough = False  # 可以根据CV结果调整

# ============================================================================
# 改进3: 构建优化Stacking模型
# ============================================================================
print("\n[Step 4] Building Optimized Stacking Model")
print("-" * 80)

# 构建最终Stacking模型
optimized_stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=best_meta_learner,
    cv=5,  # 5折交叉验证用于生成meta特征
    stack_method='predict_proba',
    passthrough=use_passthrough,
    n_jobs=-1
)

print(f"Stacking配置:")
print(f"  Base Learners: {len(base_estimators)} 个")
print(f"  Meta Learner: {best_meta_name}")
print(f"  CV折数: 5")
print(f"  Stack Method: predict_proba")
print(f"  Passthrough: {use_passthrough}")

# 训练Stacking模型（带进度条）
print("\n训练Stacking模型...")
with tqdm(total=1, desc="[Training] Stacking ensemble", unit="model") as pbar:
    optimized_stack.fit(X_train, y_train)
    pbar.update(1)

print("[OK] Stacking模型训练完成")

# ============================================================================
# 改进4: 概率校准（Calibration）
# ============================================================================
print("\n[Step 5] Probability Calibration")
print("-" * 80)

# 使用CalibratedClassifierCV进行外层校准
print("应用Isotonic校准（CV-level）...")
calibrated_stack = CalibratedClassifierCV(
    optimized_stack,
    method='isotonic',
    cv=5  # 5折交叉验证校准
)

with tqdm(total=1, desc="[Calibration] Isotonic calibration", unit="step") as pbar:
    calibrated_stack.fit(X_train, y_train)
    pbar.update(1)

# 评估校准前后效果
y_val_proba_uncalibrated = optimized_stack.predict_proba(X_val)[:, 1]
y_val_proba_calibrated = calibrated_stack.predict_proba(X_val)[:, 1]

brier_uncal = brier_score_loss(y_val, y_val_proba_uncalibrated)
brier_cal = brier_score_loss(y_val, y_val_proba_calibrated)

print(f"校准前Brier Score: {brier_uncal:.4f}")
print(f"校准后Brier Score: {brier_cal:.4f}")
print(f"改善: {((brier_uncal - brier_cal) / brier_uncal * 100):.2f}%")

# ============================================================================
# 改进5: 阈值优化（F2最大化 + Youden's J）
# ============================================================================
print("\n[Step 6] Threshold Optimization")
print("-" * 80)

def find_optimal_threshold(y_true, y_proba, strategy='f2'):
    """寻找最优分类阈值"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    if strategy == 'youden':
        # Youden's J = TPR - FPR
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
    elif strategy == 'f2':
        # F2 score (beta=2, 偏重召回率)
        f2_scores = []
        for t in tqdm(thresholds, desc="[Threshold] F2 optimization", leave=False):
            y_pred = (y_proba >= t).astype(int)
            if y_pred.sum() > 0:
                f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
            else:
                f2_scores.append(0)
        optimal_idx = np.argmax(f2_scores)
    else:
        optimal_idx = len(thresholds) // 2  # 默认0.5
    
    return thresholds[optimal_idx]

# 在验证集上寻找最优阈值
y_val_proba_cal = calibrated_stack.predict_proba(X_val)[:, 1]

threshold_youden = find_optimal_threshold(y_val, y_val_proba_cal, 'youden')
threshold_f2 = find_optimal_threshold(y_val, y_val_proba_cal, 'f2')

# 评估不同阈值下的性能
threshold_results = {}
for strategy, threshold in [('youden', threshold_youden), ('f2', threshold_f2), ('default', 0.5)]:
    y_pred_thresh = (y_val_proba_cal >= threshold).astype(int)
    threshold_results[strategy] = {
        'threshold': threshold,
        'f1': f1_score(y_val, y_pred_thresh),
        'f2': fbeta_score(y_val, y_pred_thresh, beta=2),
        'recall': recall_score(y_val, y_pred_thresh),
        'precision': (y_val[y_pred_thresh == 1].sum() / y_pred_thresh.sum()) if y_pred_thresh.sum() > 0 else 0
    }

print("\n阈值优化结果（验证集）:")
for strategy, metrics in threshold_results.items():
    print(f"  {strategy.capitalize():>10} (阈值={metrics['threshold']:.4f}): "
          f"F1={metrics['f1']:.4f}, F2={metrics['f2']:.4f}, "
          f"Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}")

# 选择F2最优阈值（优先召回率）
optimal_threshold = threshold_f2
print(f"\n[OK] 选择F2最优阈值: {optimal_threshold:.4f} (Recall优先)")

# ============================================================================
# 改进6: 测试集评估
# ============================================================================
print("\n[Step 7] Test Set Evaluation")
print("-" * 80)

# 测试集预测（使用校准后的概率）
y_test_proba_calibrated = calibrated_stack.predict_proba(X_test)[:, 1]

# 使用不同阈值评估
evaluation_results = {}

# 默认阈值0.5
y_test_pred_default = (y_test_proba_calibrated >= 0.5).astype(int)
evaluation_results['default'] = evaluate_model(y_test, y_test_pred_default, y_test_proba_calibrated, "Test")

# F2最优阈值
y_test_pred_f2 = (y_test_proba_calibrated >= optimal_threshold).astype(int)
evaluation_results['f2_optimized'] = evaluate_model(y_test, y_test_pred_f2, y_test_proba_calibrated, "Test")

print("\n测试集性能（不同阈值）:")
print(f"{'策略':<15} {'AUC':<10} {'AUPRC':<10} {'Recall':<10} {'F1':<10} {'F2':<10} {'Brier':<10}")
print("-" * 80)
for strategy, metrics in evaluation_results.items():
    print(f"{strategy:<15} {metrics['AUC']:<10.4f} {metrics['AUPRC']:<10.4f} "
          f"{metrics['Recall']:<10.4f} {metrics['F1']:<10.4f} {metrics['F2']:<10.4f} "
          f"{metrics['Brier']:<10.4f}")

# 选择最佳策略（F2优化通常更适合临床场景）
final_metrics = evaluation_results['f2_optimized']
final_predictions = y_test_pred_f2

print(f"\n[OK] 最终模型性能（F2优化阈值={optimal_threshold:.4f}）:")
print(f"  AUC: {final_metrics['AUC']:.4f} (目标: ≥0.81)")
print(f"  AUPRC: {final_metrics['AUPRC']:.4f}")
print(f"  Recall: {final_metrics['Recall']:.4f} (目标: ≥0.60)")
print(f"  F1-Score: {final_metrics['F1']:.4f}")
print(f"  F2-Score: {final_metrics['F2']:.4f}")
print(f"  Brier Score: {final_metrics['Brier']:.4f} (目标: ≤0.14)")

# 检查是否达到目标
targets_met = {
    'AUC': final_metrics['AUC'] >= 0.81,
    'Recall': final_metrics['Recall'] >= 0.60,
    'Brier': final_metrics['Brier'] <= 0.14
}

print("\n目标达成情况:")
for target, met in targets_met.items():
    status = "[OK]" if met else "[待改进]"
    print(f"  {status} {target}")

# ============================================================================
# 改进7: 可视化（ROC, PR, Calibration, DCA）
# ============================================================================
print("\n[Step 8] Visualization")
print("-" * 80)

# 确保输出目录存在（支持从scripts目录或项目根目录运行）
vis_dir = Path('../visualizations/modeling')
if not vis_dir.exists():
    vis_dir = Path('visualizations/modeling')
vis_dir.mkdir(parents=True, exist_ok=True)

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_test_proba_calibrated)
auc = roc_auc_score(y_test, y_test_proba_calibrated)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Optimized Stacking (AUC={auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Optimized Stacking Ensemble')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(vis_dir / 'optimized_stacking_roc.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: optimized_stacking_roc.png")

# Precision-Recall曲线
precision, recall, _ = precision_recall_curve(y_test, y_test_proba_calibrated)
auprc = average_precision_score(y_test, y_test_proba_calibrated)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Optimized Stacking (AUPRC={auprc:.4f})', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Optimized Stacking Ensemble')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(vis_dir / 'optimized_stacking_pr.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: optimized_stacking_pr.png")

# 校准曲线
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_test_proba_calibrated, n_bins=10
)

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, 'o-', 
         label=f'Optimized Stacking (Brier={final_metrics["Brier"]:.4f})', linewidth=2)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve - Optimized Stacking Ensemble')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(vis_dir / 'optimized_stacking_calibration.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: optimized_stacking_calibration.png")

# 决策曲线分析（DCA）
def decision_curve_analysis(y_true, y_proba, thresholds):
    """计算决策曲线的净效益"""
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
net_benefits = decision_curve_analysis(y_test, y_test_proba_calibrated, thresholds_dca)

plt.figure(figsize=(10, 8))
plt.plot(thresholds_dca, net_benefits, label='Optimized Stacking Model', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Treat None')
plt.axhline(y=(y_test == 1).mean(), color='gray', linestyle='--', alpha=0.5, label='Treat All')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('Decision Curve Analysis (DCA) - Optimized Stacking Ensemble')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(vis_dir / 'optimized_stacking_dca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved: optimized_stacking_dca.png")

# ============================================================================
# 改进8: Bootstrap置信区间
# ============================================================================
print("\n[Step 9] Bootstrap Confidence Intervals")
print("-" * 80)

def bootstrap_confidence_intervals(y_true, y_proba, n_bootstrap=1000, random_state=42):
    """计算Bootstrap置信区间"""
    np.random.seed(random_state)
    n = len(y_true)
    bootstrap_results = []
    
    with tqdm(total=n_bootstrap, desc="[Bootstrap] Computing CI", unit="iter") as pbar:
        for i in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_proba_boot = y_proba[indices]
            
            try:
                auc = roc_auc_score(y_true_boot, y_proba_boot)
                auprc = average_precision_score(y_true_boot, y_proba_boot)
                brier = brier_score_loss(y_true_boot, y_proba_boot)
                bootstrap_results.append({'AUC': auc, 'AUPRC': auprc, 'Brier': brier})
            except:
                continue
            pbar.update(1)
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    ci_results = {}
    for metric in ['AUC', 'AUPRC', 'Brier']:
        ci_results[metric] = {
            'Mean': bootstrap_df[metric].mean(),
            'Lower_CI_95': bootstrap_df[metric].quantile(0.025),
            'Upper_CI_95': bootstrap_df[metric].quantile(0.975)
        }
    
    return ci_results

bootstrap_ci = bootstrap_confidence_intervals(y_test, y_test_proba_calibrated, n_bootstrap=1000)

print("\nBootstrap置信区间（95% CI）:")
for metric, values in bootstrap_ci.items():
    print(f"  {metric:>8}: {values['Mean']:.4f} [{values['Lower_CI_95']:.4f}, {values['Upper_CI_95']:.4f}]")

# ============================================================================
# 保存结果
# ============================================================================
print("\n[Step 10] Saving Results")
print("-" * 80)

# 确保输出目录存在（支持从scripts目录或项目根目录运行）
artifacts_dir = Path('../artifacts')
if not artifacts_dir.exists():
    artifacts_dir = Path('artifacts')
artifacts_dir.mkdir(parents=True, exist_ok=True)

logs_dir = Path('../logs/modeling')
if not logs_dir.exists():
    logs_dir = Path('logs/modeling')
logs_dir.mkdir(parents=True, exist_ok=True)

# 保存模型
joblib.dump(calibrated_stack, str(artifacts_dir / 'optimized_stacking_model.pkl'))
print(f"[OK] Saved: {artifacts_dir / 'optimized_stacking_model.pkl'}")

# 保存性能指标
performance_summary = pd.DataFrame([
    {**final_metrics, 'Model': 'Optimized_Stacking', 'Threshold_Strategy': 'F2_Optimized'}
])
performance_summary.to_csv(str(logs_dir / 'optimized_stacking_performance.csv'), index=False)
print(f"[OK] Saved: {logs_dir / 'optimized_stacking_performance.csv'}")

# 保存阈值结果
threshold_df = pd.DataFrame([
    {'Strategy': k, 'Threshold': f"{v['threshold']:.4f}", 
     'F1': f"{v['f1']:.4f}", 'F2': f"{v['f2']:.4f}", 
     'Recall': f"{v['recall']:.4f}", 'Precision': f"{v['precision']:.4f}"}
    for k, v in threshold_results.items()
])
threshold_df.to_csv(str(logs_dir / 'optimized_stacking_thresholds.csv'), index=False)
print(f"[OK] Saved: {logs_dir / 'optimized_stacking_thresholds.csv'}")

# 保存Bootstrap CI
bootstrap_df = pd.DataFrame([
    {
        'Metric': metric,
        'Mean': f"{values['Mean']:.4f}",
        'Lower_CI_95': f"{values['Lower_CI_95']:.4f}",
        'Upper_CI_95': f"{values['Upper_CI_95']:.4f}"
    }
    for metric, values in bootstrap_ci.items()
])
bootstrap_df.to_csv(str(logs_dir / 'optimized_stacking_bootstrap_ci.csv'), index=False)
print(f"[OK] Saved: {logs_dir / 'optimized_stacking_bootstrap_ci.csv'}")

# 保存配置信息
config = {
    'base_estimators': [name for name, _ in base_estimators],
    'meta_learner': best_meta_name,
    'cv_folds': 5,
    'stack_method': 'predict_proba',
    'passthrough': use_passthrough,
    'calibration_method': 'isotonic',
    'calibration_cv': 5,
    'optimal_threshold': float(optimal_threshold),
    'threshold_strategy': 'F2_maximization'
}

with open(str(artifacts_dir / 'optimized_stacking_config.json'), 'w') as f:
    json.dump(config, f, indent=2)
print(f"[OK] Saved: {artifacts_dir / 'optimized_stacking_config.json'}")

# ============================================================================
# 最终总结
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Optimized Stacking Ensemble Training Complete!")
print("=" * 80)

print("\n最终模型配置:")
print(f"  Base Learners: {', '.join([name for name, _ in base_estimators])}")
print(f"  Meta Learner: {best_meta_name}")
print(f"  概率校准: Isotonic (CV=5)")
print(f"  最优阈值: {optimal_threshold:.4f} (F2最大化)")

print("\n测试集性能:")
print(f"  AUC: {final_metrics['AUC']:.4f} (目标: ≥0.81) {'✓' if targets_met['AUC'] else '✗'}")
print(f"  AUPRC: {final_metrics['AUPRC']:.4f}")
print(f"  Recall: {final_metrics['Recall']:.4f} (目标: ≥0.60) {'✓' if targets_met['Recall'] else '✗'}")
print(f"  F1-Score: {final_metrics['F1']:.4f}")
print(f"  F2-Score: {final_metrics['F2']:.4f}")
print(f"  Brier Score: {final_metrics['Brier']:.4f} (目标: ≤0.14) {'✓' if targets_met['Brier'] else '✗'}")

print("\nBootstrap 95% CI:")
print(f"  AUC: [{bootstrap_ci['AUC']['Lower_CI_95']:.4f}, {bootstrap_ci['AUC']['Upper_CI_95']:.4f}]")
print(f"  AUPRC: [{bootstrap_ci['AUPRC']['Lower_CI_95']:.4f}, {bootstrap_ci['AUPRC']['Upper_CI_95']:.4f}]")
print(f"  Brier: [{bootstrap_ci['Brier']['Lower_CI_95']:.4f}, {bootstrap_ci['Brier']['Upper_CI_95']:.4f}]")

print("\n生成的文件:")
print("  - artifacts/optimized_stacking_model.pkl")
print("  - artifacts/optimized_stacking_config.json")
print("  - logs/modeling/optimized_stacking_performance.csv")
print("  - logs/modeling/optimized_stacking_thresholds.csv")
print("  - logs/modeling/optimized_stacking_bootstrap_ci.csv")
print("  - visualizations/modeling/optimized_stacking_roc.png")
print("  - visualizations/modeling/optimized_stacking_pr.png")
print("  - visualizations/modeling/optimized_stacking_calibration.png")
print("  - visualizations/modeling/optimized_stacking_dca.png")

print("\n" + "=" * 80)
