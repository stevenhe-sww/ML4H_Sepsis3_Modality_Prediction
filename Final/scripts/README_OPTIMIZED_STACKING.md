# 优化版 Stacking 集成模型使用说明

## 概述

`step4_optimized_stacking.py` 是一个优化的 Stacking 集成模型脚本，针对 Sepsis-3 死亡预测任务进行了全面改进。

## 主要改进点

### 1. 多样性增强的 Base Learners
- **LR (Logistic Regression)**: 线性模型，带预处理pipeline
- **XGBoost**: 梯度提升树
- **LightGBM**: 梯度提升树（不同算法）
- **RandomForest**: Bagging机制，增加多样性

**改进**: 从3个base learners增加到4个，引入RF的bagging机制，提升模型异质性。

### 2. Meta Learner 优化
- 自动比较不同C参数的Logistic Regression
- 测试 `passthrough=True/False` 的差异
- 选择最佳Meta Learner配置

### 3. 概率校准
- 使用 `CalibratedClassifierCV(method='isotonic', cv=5)` 进行外层校准
- 改善Brier Score和概率可信度
- 输出校准前后的对比

### 4. 阈值优化
- **F2最大化**: 偏重召回率（降低漏诊）
- **Youden's J**: 平衡敏感性和特异性
- 在验证集上优化，在测试集上评估

### 5. 可视化
生成以下图表：
- ROC曲线
- Precision-Recall曲线
- 校准曲线（Calibration Curve）
- 决策曲线分析（DCA）

### 6. Bootstrap置信区间
- 1000次Bootstrap重采样
- 计算AUC、AUPRC、Brier Score的95%置信区间

## 使用方法

### 运行脚本

```bash
cd /Users/zixuanhe/Desktop/ML4H/Final
python3 scripts/step4_optimized_stacking.py
```

### 输出文件

**模型文件**:
- `artifacts/optimized_stacking_model.pkl`: 校准后的最终模型
- `artifacts/optimized_stacking_config.json`: 模型配置信息

**性能指标**:
- `logs/modeling/optimized_stacking_performance.csv`: 测试集性能指标
- `logs/modeling/optimized_stacking_thresholds.csv`: 阈值优化结果
- `logs/modeling/optimized_stacking_bootstrap_ci.csv`: Bootstrap置信区间

**可视化**:
- `visualizations/modeling/optimized_stacking_roc.png`: ROC曲线
- `visualizations/modeling/optimized_stacking_pr.png`: PR曲线
- `visualizations/modeling/optimized_stacking_calibration.png`: 校准曲线
- `visualizations/modeling/optimized_stacking_dca.png`: 决策曲线分析

## 性能目标

脚本会检查以下目标是否达成：
- **AUC**: ≥ 0.81
- **Recall**: ≥ 0.60 (优先降低漏诊)
- **Brier Score**: ≤ 0.14

## 关键特性

1. **进度条**: 使用 `tqdm` 显示训练进度
2. **中文注释**: 详细的中文注释说明每个步骤
3. **错误处理**: 自动处理路径问题（支持从不同目录运行）
4. **可复现性**: 固定随机种子（random_state=42）

## 与原始Stacking的对比

| 特性 | 原始Stacking | 优化Stacking |
|------|-------------|-------------|
| Base Learners | 3个 (LR, XGB, LGB) | 4个 (LR, XGB, LGB, RF) |
| Meta Learner | 固定LR | 自动选择最佳配置 |
| 概率校准 | 无 | Isotonic校准 (CV=5) |
| 阈值优化 | 固定0.5 | F2最大化 + Youden's J |
| Bootstrap CI | 无 | 1000次重采样 |
| DCA分析 | 无 | 完整决策曲线 |

## 注意事项

1. **运行时间**: 脚本需要较长时间运行（约5-10分钟），因为包含：
   - 4个meta learner的5折CV评估
   - Stacking模型的5折CV训练
   - 概率校准的5折CV
   - 1000次Bootstrap重采样

2. **内存需求**: 确保有足够内存加载数据和训练模型

3. **依赖项**: 确保已安装所有必需的包（见脚本开头的imports）

## 预期结果示例

```
[OK] 最终模型性能（F2优化阈值=0.1065）:
  AUC: 0.8033 (目标: ≥0.81) ✗
  AUPRC: 0.5784
  Recall: 0.7365 (目标: ≥0.60) ✓
  F1-Score: 0.5422
  F2-Score: 0.XXXX
  Brier Score: 0.1377 (目标: ≤0.14) ✓
```

## 下一步优化建议

如果模型未达到目标，可以考虑：
1. 增加Optuna超参数优化（目前使用固定参数）
2. 添加SMOTE重采样（脚本中已预留接口）
3. 使用CatBoost作为额外的base learner
4. 添加monotonic constraints（如SOFA ↑ → Mortality ↑）

