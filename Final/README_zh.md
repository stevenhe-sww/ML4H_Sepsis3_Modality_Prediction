Sepsis-3 队列 30 天病死率预测项目

简介

本项目实现了一个从数据到模型的完整机器学习工作流，用于预测 Sepsis-3 患者 30 天病死率。工作流包含数据探索分析、数据清洗与预处理、基线建模、超参数优化、模型解释性与稳健性评估等模块，并提供统一的主训练脚本实现端到端运行与结果导出。

主要功能

- 数据探索分析（EDA）：基础与高级统计分析与可视化
- 数据清洗与预处理：临床感知的异常值处理、缺失处理、特征工程与标准化
- 基线建模：逻辑回归、随机森林、XGBoost、LightGBM、Stacking 集成
- 模型优化：Optuna 超参搜索、阈值优化、概率校准
- 可解释性与稳健性：SHAP 特征重要性、分组分析、Bootstrap 置信区间、决策曲线分析
- 统一编排：主训练脚本整合各环节，提供统一日志与结果导出

项目结构（根目录）

ML4H/Final/
- README.md：英文版文档
- README_zh.md：本文档
- data/：数据目录（raw 原始数据；processed 清洗后数据）
- scripts/：分析与建模脚本
- visualizations/：EDA 和建模可视化输出
- logs/：EDA 与清洗过程日志与统计输出
- reports/：分析与方法学说明文档
- artifacts/：预处理对象与模型等工件
- outputs_*/：训练输出目录（按创建顺序编号，如 outputs_1, outputs_2, ...）
- docs/：原始任务说明与需求

数据目录说明

data/
- raw/
  - sepsis3_cohort_all_features.csv：原始 Sepsis-3 队列数据（含 mortality_30d 目标变量）
- processed/
  - sepsis3_cleaned.csv：清洗与特征工程后的建模数据

要点：
- raw/ 只读，作为数据源事实标准；processed/ 由 step2_data_cleaning.py 生成
- 所有脚本相对项目根路径读写数据

脚本目录说明（scripts/）

- step1_eda.py：基础 EDA
  - 加载数据、变量分类（连续/分类型）、分布图（直方图、提琴图）、相关性矩阵、异常值检查、汇总统计

- step1_advanced_eda.py：高级 EDA
  - 生存与死亡组比较（t 检验、箱线/提琴图）、缺失模式（missingno/热力图）、正态性与偏度、
    连续变量与病死率关联趋势（逻辑回归光滑曲线）、多变量交互散点图、生成文字总结

- step2_data_cleaning.py：数据清洗与预处理
  - 异常值处理：分位截尾、临床生理上限裁剪、负值裁剪（如尿量）
  - 偏度校正：log1p 与 QuantileTransformer 正态化
  - 缺失处理：中位数、KNNImputer（按变量组）、高缺失率变量缺失指示器
  - 多重共线：删除高相关变量（示例：sodium_max、platelets_max）
  - 编码与特征工程：One-Hot 编码、关键交互项（如 SOFA×乳酸、SOFA×年龄）、分箱
  - 标准化：StandardScaler 应用于连续变量
  - 质量检查与日志记录

- step3_modeling.py：基线建模
  - 训练/验证/测试划分（70/15/15，分层）
  - 基线模型：逻辑回归、随机森林、XGBoost、LightGBM、Stacking
  - 评估：AUC、AUPRC、Accuracy、Recall、F1、Brier；ROC/PR 图；概率校准；SHAP；分组分析

- step4_model_optimization.py：模型优化
  - 预处理流水线封装、阈值优化（Youden、F2、代价）、Optuna 超参搜索、Bootstrap 置信区间、
    交叉验证层级校准、子组稳健性评估、决策曲线分析

- main_training_pipeline.py：主训练流水线
  - 统一编排：数据加载、预处理、基线建模、Optuna 优化、阈值优化、校准、可解释性、导出
  - 命令行参数：--data、--out_dir、--seed、--optuna_trials、--cv_folds
  - 自动创建输出子目录、详细日志、进度条显示

可视化目录说明（visualizations/）

- eda/：基础 EDA 图
- advanced_eda/：高级 EDA 图（组间比较、缺失模式、偏度校正、交互等）
- modeling/：模型评估图（ROC、PR、校准、DCA、可选 SHAP）

日志目录说明（logs/）

- eda/：EDA 统计输出
- cleaning/：清洗过程日志与统计（异常值、变换、缺失、对比）

报告目录说明（reports/）

- advanced_eda_summary_report.md：高级 EDA 结论
- data_cleaning_report.md：数据清洗方法与结果
- data_cleaning_enhancements.md：清洗增强项与影响
- file_organization_index.md：文件索引与说明
- QUICK_REFERENCE.md：常用命令、路径与问题排查

工件目录说明（artifacts/）

- 预处理对象与特征映射，便于再现与部署（例如标准化器与特征清单）

训练输出目录说明（outputs_*）

- 说明：训练运行的所有结果按创建顺序编号保存为 outputs_1、outputs_2、…、outputs_N
- 每个 outputs_*/ 目录包含：
  - artifacts/：best_sepsis_model.pkl（最佳模型）、optuna_best_params.json（最优超参）
  - logs/：train.log、model_performance_summary.csv、bootstrap_confidence_intervals.csv、
    threshold_optimization_report.csv、subgroup_performance_table.csv
  - plots/：roc_pr_curves.png、calibration_plot.png、dca_plot.png

运行说明（Quick Start）

环境依赖（建议 Python 3.10+）：
- pandas numpy scikit-learn matplotlib seaborn
- xgboost lightgbm optuna shap missingno
- tqdm scipy joblib

示例流程：

1. 基础 EDA
  cd scripts
  python step1_eda.py

2. 高级 EDA
  python step1_advanced_eda.py

3. 数据清洗
  python step2_data_cleaning.py

4. 主训练流水线（推荐）
  python main_training_pipeline.py \
    --data data/processed/sepsis3_cleaned.csv \
    --out_dir outputs_7 \
    --seed 42 \
    --optuna_trials 80 \
    --cv_folds 5

关键结果（最近一次正式运行示例：outputs_7）

- 基线最佳：Stacking（逻辑回归 + XGBoost + LightGBM）
- 优化模型：XGBoost_Optimized，AUC 约 0.802，AUPRC 约 0.575，Brier 约 0.138
- 阈值优化：Youden、F2、代价最小化三种方案
- 置信区间：Bootstrap 1000 次（AUC、AUPRC、Brier）
- 分组分析：年龄分层、感染来源、升压药使用等
- 决策曲线分析：提供不同阈值下净获益对比

数据清洗与建模要点

- 清洗：临床上限裁剪、分位截尾、log1p 与分位正态化、缺失中位数与 KNN、缺失指示器、
  多重共线控制、One-Hot 编码、交互项与分箱、标准化
- 建模：五折分层、AUC/AUPRC/Accuracy/Recall/F1/Brier 统一评估、ROC/PR 与校准曲线、
  概率校准（Platt/Isotonic/CalibratedClassifierCV）、SHAP、Bootstrap 与子组稳健性
- 优化：Optuna TPE 搜索、早停、阈值优化（Youden/F2/代价）、可选单调约束

复现性与日志

- 随机种子固定为 42
- 统一日志输出到各 outputs_*/logs/train.log，同时在控制台打印
- 进度条显示关键阶段进度

联系方式与支持

- 详细方法说明：参见 reports/ 与脚本内文档字符串
- 运行日志与指标：参见各 outputs_*/logs/ 下的 CSV 与 train.log

更新信息

- 最后更新时间：以各 outputs_*/logs/train.log 为准
- Python 版本：建议 3.10 及以上


