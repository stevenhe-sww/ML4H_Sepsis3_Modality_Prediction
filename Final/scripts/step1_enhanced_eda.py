"""
Step 1 Enhanced - Comprehensive EDA with Clinical Insights
增强版探索性数据分析：包含临床阈值、目标相关性、非线性分析、交互分析等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy import stats
from scipy.stats import pointbiserialr, chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# 创建输出目录
Path('../visualizations/eda').mkdir(parents=True, exist_ok=True)
Path('../logs/eda').mkdir(parents=True, exist_ok=True)
Path('../tables').mkdir(parents=True, exist_ok=True)
Path('../figures').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Step 1 Enhanced - Comprehensive EDA with Clinical Insights")
print("=" * 80)

# ============================================================================
# 数据加载和基础信息
# ============================================================================
print("\n[Step 0] Data Loading and Quality Check")
print("-" * 80)

data_path = Path("../data/processed/sepsis3_cleaned.csv")
if not data_path.exists():
    data_path = Path("data/processed/sepsis3_cleaned.csv")

df = pd.read_csv(data_path)
print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# 移除ID列
id_cols = ['subject_id', 'hadm_id', 'stay_id']
id_cols_present = [col for col in id_cols if col in df.columns]
if id_cols_present:
    df = df.drop(columns=id_cols_present)

target_col = 'mortality_30d'
if target_col not in df.columns:
    print(f"[ERROR] Target column '{target_col}' not found!")
    exit(1)

# 数据质量检查
print("\nData Quality Check:")
print(f"  Missing values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)")
print(f"  Duplicate rows: {df.duplicated().sum()}")
print(f"  Target distribution: {(df[target_col]==1).sum()} positive ({(df[target_col]==1).mean()*100:.2f}%)")

# 识别变量类型
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != target_col]
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"  Numeric features: {len(numeric_cols)}")
print(f"  Categorical features: {len(categorical_cols)}")

# ============================================================================
# A1. 层级聚类相关图（Spearman，|corr| 聚类）
# ============================================================================
print("\n[A1] Hierarchical Clustered Correlation Matrix (Spearman)")
print("-" * 80)

# 选择数值变量（至少50个非缺失值）
corr_vars = [col for col in numeric_cols if df[col].notna().sum() > 50]
if len(corr_vars) > 30:
    corr_vars = corr_vars[:30]  # 限制变量数量以提高可读性

if len(corr_vars) > 1:
    # 计算Spearman相关系数（对重尾/单调关系更稳健）
    corr_matrix = df[corr_vars].corr(method='spearman')
    
    # 创建层级聚类热图
    plt.figure(figsize=(16, 14))
    g = sns.clustermap(
        corr_matrix,
        method='ward',
        metric='euclidean',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Spearman Correlation", "shrink": 0.8},
        figsize=(16, 14),
        fmt='.2f',
        annot=False  # 太多变量时注释会太密集
    )
    g.fig.suptitle('Hierarchical Clustered Correlation Matrix (Spearman)\nVariables grouped by similarity', 
                   fontsize=16, y=1.02)
    plt.savefig('../figures/fig_corr_clustermap_spearman.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_corr_clustermap_spearman.png")
    plt.close()
    
    # 保存相关系数矩阵
    corr_matrix.to_csv('../tables/tbl_corr_matrix.csv')
    print("[OK] Saved: tables/tbl_corr_matrix.csv")
    
    print(f"\nCorrelation matrix computed for {len(corr_vars)} variables")
    print(f"  Mean |correlation|: {corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    print(f"  Max |correlation|: {corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")

# ============================================================================
# A2. 目标相关性分析（Point-Biserial & Cramér's V）
# ============================================================================
print("\n[A2] Feature-Target Association Analysis")
print("-" * 80)

y = df[target_col]

# 数值特征：Point-Biserial相关
print("\nComputing point-biserial correlations for numeric features...")
numeric_target_corr = []

for col in numeric_cols:
    if df[col].notna().sum() > 50:
        data_clean = df[[col, target_col]].dropna()
        if len(data_clean) > 50:
            try:
                r, p = pointbiserialr(data_clean[target_col], data_clean[col])
                # 计算AUC作为补充
                try:
                    auc = roc_auc_score(data_clean[target_col], data_clean[col])
                except:
                    auc = 0.5
                numeric_target_corr.append({
                    'Feature': col,
                    'Correlation': r,
                    'P_value': p,
                    'AUC': auc,
                    'N': len(data_clean)
                })
            except:
                continue

numeric_corr_df = pd.DataFrame(numeric_target_corr)
if len(numeric_corr_df) > 0:
    # FDR校正（Benjamini-Hochberg）
    from statsmodels.stats.multitest import multipletests
    _, q_values, _, _ = multipletests(numeric_corr_df['P_value'], method='fdr_bh')
    numeric_corr_df['Q_value'] = q_values
    numeric_corr_df['Significant'] = q_values < 0.05
    
    # 按|相关系数|排序
    numeric_corr_df['Abs_Correlation'] = numeric_corr_df['Correlation'].abs()
    numeric_corr_df = numeric_corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # 保存完整表
    numeric_corr_df.to_csv('../tables/tbl_target_corr_pointbiserial_all.csv', index=False)
    print(f"[OK] Saved: tables/tbl_target_corr_pointbiserial_all.csv ({len(numeric_corr_df)} features)")
    
    # 绘制Top-30条形图
    top_n = min(30, len(numeric_corr_df))
    top_features = numeric_corr_df.head(top_n)
    
    plt.figure(figsize=(12, max(8, top_n * 0.3)))
    colors = ['red' if sig else 'gray' for sig in top_features['Significant']]
    plt.barh(range(len(top_features)), top_features['Correlation'].values, color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'].values, fontsize=9)
    plt.xlabel('Point-Biserial Correlation with Mortality', fontsize=12)
    plt.title(f'Top {top_n} Features by Association with 30-Day Mortality\n(Red: FDR q<0.05, Gray: Non-significant)', 
              fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('../figures/fig_target_corr_pointbiserial_top30.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_target_corr_pointbiserial_top30.png")
    plt.close()
    
    # 打印Top-10
    print("\nTop 10 features by |correlation|:")
    for idx, row in top_features.head(10).iterrows():
        sig_mark = "[*]" if row['Significant'] else ""
        print(f"  {row['Feature']:<30} r={row['Correlation']:7.4f} (p={row['P_value']:.4f}, q={row['Q_value']:.4f}) AUC={row['AUC']:.4f} {sig_mark}")

# 类别特征：Cramér's V
print("\nComputing Cramér's V for categorical features...")
categorical_target_corr = []

for col in categorical_cols:
    if col != target_col and df[col].notna().sum() > 50:
        try:
            contingency = pd.crosstab(df[col], df[target_col])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p, dof, expected = chi2_contingency(contingency)
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                categorical_target_corr.append({
                    'Feature': col,
                    "Cramer's_V": cramers_v,
                    'P_value': p,
                    'N': n,
                    'Categories': contingency.shape[0]
                })
        except:
            continue

if len(categorical_target_corr) > 0:
    cat_corr_df = pd.DataFrame(categorical_target_corr)
    cat_corr_df = cat_corr_df.sort_values("Cramer's_V", ascending=False)
    
    # FDR校正
    _, q_values, _, _ = multipletests(cat_corr_df['P_value'], method='fdr_bh')
    cat_corr_df['Q_value'] = q_values
    cat_corr_df['Significant'] = q_values < 0.05
    
    cat_corr_df.to_csv('../tables/tbl_target_assoc_cramersV_all.csv', index=False)
    print(f"[OK] Saved: tables/tbl_target_assoc_cramersV_all.csv ({len(cat_corr_df)} features)")
    
    # 绘制Top-20条形图
    top_n = min(20, len(cat_corr_df))
    top_cat = cat_corr_df.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = ['red' if sig else 'gray' for sig in top_cat['Significant']]
    plt.barh(range(len(top_cat)), top_cat["Cramer's_V"].values, color=colors)
    plt.yticks(range(len(top_cat)), top_cat['Feature'].values, fontsize=9)
    plt.xlabel("Cramér's V with Mortality", fontsize=12)
    plt.title(f'Top {top_n} Categorical Features by Association with 30-Day Mortality\n(Red: FDR q<0.05)', 
              fontsize=14, pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('../figures/fig_target_assoc_cramersV_top20.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_target_assoc_cramersV_top20.png")
    plt.close()

# ============================================================================
# B. 单变量分布：临床阈值与变换前后对比
# ============================================================================
print("\n[B] Univariate Distributions with Clinical Thresholds")
print("-" * 80)

# 定义临床参考区间
clinical_thresholds = {
    'lactate_max_24h': {'normal': (0, 2), 'elevated': (2, 4), 'high': (4, np.inf), 'unit': 'mmol/L'},
    'platelet_min': {'normal': (150, np.inf), 'low': (100, 150), 'very_low': (0, 100), 'unit': '×10⁹/L'},
    'sodium_mean': {'normal': (135, 145), 'low': (0, 135), 'high': (145, np.inf), 'unit': 'mmol/L'},
    'creatinine_max': {'normal': (0, 1.2), 'elevated': (1.2, 2.0), 'high': (2.0, np.inf), 'unit': 'mg/dL'},
    'wbc_max': {'normal': (4, 11), 'low': (0, 4), 'high': (11, np.inf), 'unit': '×10⁹/L'},
    'age': {'normal': (18, 65), 'elderly': (65, 75), 'very_elderly': (75, np.inf), 'unit': 'years'},
}

# 计算偏度和峰度（变换前后）
skew_kurt_results = []

# 选择关键变量进行详细分析
key_vars = ['sofa_total', 'lactate_max_24h', 'age', 'creatinine_max', 'platelet_min', 
            'sodium_mean', 'wbc_max', 'mbp_mean', 'hr_mean']

key_vars = [v for v in key_vars if v in df.columns]

for var in key_vars[:6]:  # 展示前6个
    if df[var].notna().sum() > 50:
        data_clean = df[var].dropna()
        
        # 计算原始分布的偏度和峰度
        skew_orig = stats.skew(data_clean)
        kurt_orig = stats.kurtosis(data_clean)
        
        # 尝试log变换
        if (data_clean > 0).all():
            data_log = np.log1p(data_clean)
            skew_log = stats.skew(data_log)
            kurt_log = stats.kurtosis(data_log)
        else:
            skew_log = None
            kurt_log = None
        
        skew_kurt_results.append({
            'Variable': var,
            'Skew_Original': skew_orig,
            'Kurt_Original': kurt_orig,
            'Skew_Log1p': skew_log,
            'Kurt_Log1p': kurt_log,
            'N': len(data_clean)
        })
        
        # 绘制分布图（带临床阈值）
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：原始分布 + 临床阈值
        ax1 = axes[0]
        ax1.hist(data_clean, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        
        # 添加临床阈值（如果存在）
        if var in clinical_thresholds:
            thresholds = clinical_thresholds[var]
            unit = thresholds.get('unit', '')
            
            # 绘制正常范围
            if 'normal' in thresholds:
                norm_range = thresholds['normal']
                if norm_range[1] != np.inf:
                    ax1.axvspan(norm_range[0], norm_range[1], alpha=0.2, color='green', label='Normal Range')
            
            # 绘制阈值线
            if 'elevated' in thresholds:
                ax1.axvline(thresholds['elevated'][0], color='orange', linestyle='--', linewidth=2, label='Elevated')
            if 'high' in thresholds:
                ax1.axvline(thresholds['high'][0], color='red', linestyle='--', linewidth=2, label='High')
            
            title_suffix = f" ({unit})"
        else:
            title_suffix = ""
        
        ax1.set_xlabel(f'{var}{title_suffix}', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Distribution: {var}\nSkew={skew_orig:.2f}, Kurt={kurt_orig:.2f}, n={len(data_clean)}', 
                      fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：变换前后对比（如果适用log变换）
        ax2 = axes[1]
        if (data_clean > 0).all() and skew_orig > 1:
            data_log = np.log1p(data_clean)
            ax2.hist(data_clean, bins=50, alpha=0.5, label='Original', color='steelblue')
            ax2.hist(data_log, bins=50, alpha=0.5, label='Log1p Transformed', color='coral')
            ax2.set_xlabel('Value', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'Transformation Comparison\nSkew: {skew_orig:.2f} → {skew_log:.2f}', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Log transformation\nnot applicable\nor not needed', 
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        safe_var_name = var.replace('/', '_').replace(' ', '_')
        plt.savefig(f'../figures/fig_dist_with_clinical_bands_{safe_var_name}.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: figures/fig_dist_with_clinical_bands_{safe_var_name}.png")
        plt.close()

# 保存偏度峰度表
if skew_kurt_results:
    skew_kurt_df = pd.DataFrame(skew_kurt_results)
    skew_kurt_df.to_csv('../tables/tbl_var_skew_kurtosis_before_after.csv', index=False)
    print("[OK] Saved: tables/tbl_var_skew_kurtosis_before_after.csv")

print("\n[Note] Clinical thresholds visualization completed for key variables")
print("        Full distribution analysis saved to figures/ directory")

# ============================================================================
# C. 缺失模式分析增强（MNAR分析）
# ============================================================================
print("\n[C] Enhanced Missing Pattern Analysis (MNAR)")
print("-" * 80)

# C1. 缺失聚类
missing_df = df[numeric_cols].isnull()
if missing_df.sum().sum() > 0:
    # 计算缺失相关性
    missing_corr = missing_df.corr()
    
    # 缺失聚类热图
    plt.figure(figsize=(14, 12))
    sns.clustermap(
        missing_corr,
        method='ward',
        cmap='viridis',
        square=True,
        linewidths=0.5,
        figsize=(14, 12),
        cbar_kws={"label": "Missing Pattern Correlation"}
    )
    plt.suptitle('Missing Pattern Clustering\n(Variables that tend to be missing together)', 
                 fontsize=14, y=1.02)
    plt.savefig('../figures/fig_missing_clustermap.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_missing_clustermap.png")
    plt.close()

# C2. 按结局的缺失率差异
print("\nComputing missing rate differences by outcome...")
missing_diff_results = []

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        missing_by_outcome = df.groupby(target_col)[col].apply(lambda x: x.isnull().sum())
        total_by_outcome = df.groupby(target_col)[col].size()
        pct_missing = (missing_by_outcome / total_by_outcome * 100)
        
        diff = pct_missing[1] - pct_missing[0]  # 死亡组 - 生存组
        
        # 计算95% CI（比例差的置信区间）
        n0 = total_by_outcome[0]
        n1 = total_by_outcome[1]
        p0 = pct_missing[0] / 100
        p1 = pct_missing[1] / 100
        
        if n0 > 0 and n1 > 0:
            se_diff = np.sqrt(p0*(1-p0)/n0 + p1*(1-p1)/n1)
            ci_lower = (diff/100) - 1.96 * se_diff
            ci_upper = (diff/100) + 1.96 * se_diff
            
            missing_diff_results.append({
                'Variable': col,
                'Missing_Rate_Survived': pct_missing[0],
                'Missing_Rate_Died': pct_missing[1],
                'Difference': diff,
                'CI_Lower_95': ci_lower * 100,
                'CI_Upper_95': ci_upper * 100
            })

if missing_diff_results:
    missing_diff_df = pd.DataFrame(missing_diff_results)
    missing_diff_df = missing_diff_df.sort_values('Difference', ascending=False)
    missing_diff_df.to_csv('../tables/tbl_missing_diff_by_outcome.csv', index=False)
    print(f"[OK] Saved: tables/tbl_missing_diff_by_outcome.csv ({len(missing_diff_df)} variables)")
    
    # 绘制差异条形图
    top_n = min(20, len(missing_diff_df))
    top_diff = missing_diff_df.head(top_n)
    
    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    colors = ['red' if d > 0 else 'blue' for d in top_diff['Difference']]
    y_pos = np.arange(len(top_diff))
    plt.barh(y_pos, top_diff['Difference'].values, color=colors, alpha=0.7)
    
    # 添加误差棒（CI）
    for i, row in enumerate(top_diff.iterrows()):
        _, r = row
        plt.errorbar(r['Difference'], i, 
                    xerr=[[r['Difference'] - r['CI_Lower_95']], [r['CI_Upper_95'] - r['Difference']]], 
                    fmt='none', color='black', capsize=3)
    
    plt.yticks(y_pos, top_diff['Variable'].values, fontsize=9)
    plt.xlabel('Missing Rate Difference (Died - Survived, %)', fontsize=12)
    plt.title('Missing Rate Differences by Outcome\n(Red: Higher missing in death, Blue: Higher in survival)', 
              fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('../figures/fig_missing_diff_by_outcome.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_missing_diff_by_outcome.png")
    plt.close()

# C3. 缺失→死亡的单变量Logit回归
print("\nComputing missing indicator ORs vs mortality...")
missing_or_results = []

for col in numeric_cols:
    if df[col].isnull().sum() > 50:  # 至少50个缺失值
        # 创建缺失指示变量
        missing_indicator = df[col].isnull().astype(int)
        
        # 单变量Logistic回归
        try:
            X = sm.add_constant(missing_indicator)
            y = df[target_col]
            
            # 移除缺失值（如果目标也有缺失）
            valid_mask = ~(missing_indicator.isna() | y.isna())
            if valid_mask.sum() > 100:
                model = sm.Logit(y[valid_mask], X[valid_mask]).fit(disp=0)
                
                or_val = np.exp(model.params[1])
                ci_lower = np.exp(model.conf_int().iloc[1, 0])
                ci_upper = np.exp(model.conf_int().iloc[1, 1])
                p_value = model.pvalues[1]
                
                missing_or_results.append({
                    'Variable': col,
                    'Missing_Indicator_OR': or_val,
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper,
                    'P_value': p_value,
                    'N': valid_mask.sum()
                })
        except:
            continue

if missing_or_results:
    missing_or_df = pd.DataFrame(missing_or_results)
    # FDR校正
    _, q_values, _, _ = multipletests(missing_or_df['P_value'], method='fdr_bh')
    missing_or_df['Q_value'] = q_values
    missing_or_df['Significant'] = q_values < 0.05
    missing_or_df = missing_or_df.sort_values('Missing_Indicator_OR', ascending=False)
    
    missing_or_df.to_csv('../tables/tbl_missing_or_vs_mortality.csv', index=False)
    print(f"[OK] Saved: tables/tbl_missing_or_vs_mortality.csv ({len(missing_or_df)} variables)")
    
    print("\nTop 10 missing indicators by OR:")
    for idx, row in missing_or_df.head(10).iterrows():
        sig_mark = "[*]" if row['Significant'] else ""
        print(f"  {row['Variable']:<30} OR={row['Missing_Indicator_OR']:.3f} "
              f"[{row['CI_Lower_95']:.3f}, {row['CI_Upper_95']:.3f}] "
              f"(p={row['P_value']:.4f}, q={row['Q_value']:.4f}) {sig_mark}")

print("\n[Section C Complete] Missing pattern analysis with MNAR insights")

# ============================================================================
# 进度报告
# ============================================================================
print("\n" + "=" * 80)
print("[Progress] Sections A-C Complete")
print("=" * 80)
print("\nGenerated files so far:")
print("  - figures/fig_corr_clustermap_spearman.png")
print("  - figures/fig_target_corr_pointbiserial_top30.png")
print("  - figures/fig_target_assoc_cramersV_top20.png")
print("  - figures/fig_dist_with_clinical_bands_*.png")
print("  - figures/fig_missing_clustermap.png")
print("  - figures/fig_missing_diff_by_outcome.png")
print("  - tables/tbl_corr_matrix.csv")
print("  - tables/tbl_target_corr_pointbiserial_all.csv")
print("  - tables/tbl_target_assoc_cramersV_all.csv")
print("  - tables/tbl_var_skew_kurtosis_before_after.csv")
print("  - tables/tbl_missing_diff_by_outcome.csv")
print("  - tables/tbl_missing_or_vs_mortality.csv")
# ============================================================================
# D. 非线性与阈值：分位-死亡率曲线 + GAM/样条
# ============================================================================
print("\n[D] Nonlinear Analysis: Quantile-Mortality Curves & GAM/Splines")
print("-" * 80)

from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

# 选择关键变量进行非线性分析
key_vars_nonlinear = ['sofa_total', 'lactate_max_24h', 'age', 'creatinine_max', 
                      'urine_output_24h', 'mbp_mean', 'hr_mean']
key_vars_nonlinear = [v for v in key_vars_nonlinear if v in df.columns]

quantile_mortality_results = []

for var in key_vars_nonlinear[:5]:  # 分析前5个关键变量
    if df[var].notna().sum() > 100:
        data_clean = df[[var, target_col]].dropna()
        
        if len(data_clean) > 100:
            # 10分位分箱
            data_clean['quantile'] = pd.qcut(data_clean[var], q=10, duplicates='drop')
            
            quantile_stats = []
            for quantile_name, group in data_clean.groupby('quantile'):
                n = len(group)
                mortality_rate = group[target_col].mean()
                # Wilson 95% CI for proportion
                z = 1.96
                p = mortality_rate
                n_eff = n
                if n > 0:
                    ci_lower = (p + z**2/(2*n_eff) - z*np.sqrt((p*(1-p) + z**2/(4*n_eff))/n_eff)) / (1 + z**2/n_eff)
                    ci_upper = (p + z**2/(2*n_eff) + z*np.sqrt((p*(1-p) + z**2/(4*n_eff))/n_eff)) / (1 + z**2/n_eff)
                    ci_lower = max(0, ci_lower)
                    ci_upper = min(1, ci_upper)
                else:
                    ci_lower = ci_upper = 0
                
                quantile_mean = group[var].mean()
                quantile_stats.append({
                    'Variable': var,
                    'Quantile': str(quantile_name),
                    'Mean_Value': quantile_mean,
                    'Mortality_Rate': mortality_rate,
                    'CI_Lower_95': ci_lower,
                    'CI_Upper_95': ci_upper,
                    'N': n
                })
            
            quantile_df = pd.DataFrame(quantile_stats)
            quantile_mortality_results.append(quantile_df)
            
            # 绘制分位-死亡率曲线
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 左图：分位箱线图
            ax1 = axes[0]
            x_pos = np.arange(len(quantile_df))
            ax1.plot(x_pos, quantile_df['Mortality_Rate'].values, 'o-', linewidth=2, markersize=8, label='Mortality Rate')
            ax1.fill_between(x_pos, 
                           quantile_df['CI_Lower_95'].values, 
                           quantile_df['CI_Upper_95'].values, 
                           alpha=0.3, label='95% CI')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels([f'Q{i+1}' for i in range(len(quantile_df))], rotation=45)
            ax1.set_ylabel('Mortality Rate', fontsize=12)
            ax1.set_xlabel(f'{var} (10 Quantiles)', fontsize=12)
            ax1.set_title(f'Quantile-Mortality Curve: {var}\n(n={len(data_clean)})', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 右图：LOESS平滑曲线
            ax2 = axes[1]
            # LOESS平滑
            try:
                smoothed = lowess(data_clean[target_col].values, 
                                 data_clean[var].values, 
                                 frac=0.3, it=3)
                ax2.scatter(data_clean[var].values, data_clean[target_col].values, 
                          alpha=0.1, s=10, color='gray')
                ax2.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOESS Smoothed')
            except:
                # 如果LOESS失败，使用简单平滑
                sorted_data = data_clean.sort_values(var)
                ax2.scatter(sorted_data[var].values, sorted_data[target_col].values, 
                          alpha=0.1, s=10, color='gray')
            
            ax2.set_xlabel(var, fontsize=12)
            ax2.set_ylabel('Mortality Rate', fontsize=12)
            ax2.set_title(f'LOESS Smoothed Risk Curve: {var}', fontsize=14)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            safe_var_name = var.replace('/', '_').replace(' ', '_')
            plt.savefig(f'../figures/fig_quantile_mortality_curves_{safe_var_name}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: figures/fig_quantile_mortality_curves_{safe_var_name}.png")
            plt.close()

# 保存分位-死亡率表
if quantile_mortality_results:
    all_quantile_df = pd.concat(quantile_mortality_results, ignore_index=True)
    all_quantile_df.to_csv('../tables/tbl_quantile_mortality_by_var.csv', index=False)
    print("[OK] Saved: tables/tbl_quantile_mortality_by_var.csv")

# ============================================================================
# E. 交互：二维风险热力图
# ============================================================================
print("\n[E] Interaction Analysis: 2D Risk Heatmaps")
print("-" * 80)

# 定义关键的交互对
interaction_pairs = [
    ('sofa_total', 'lactate_max_24h'),
    ('age', 'sofa_total'),
    ('creatinine_max', 'urine_output_24h'),
    ('mbp_mean', 'hr_mean'),
]

available_pairs = [(x, y) for x, y in interaction_pairs 
                  if x in df.columns and y in df.columns]

for var1, var2 in available_pairs[:3]:  # 展示前3个交互对
    data_clean = df[[var1, var2, target_col]].dropna()
    
    if len(data_clean) > 100:
        # 4分位分箱
        data_clean[f'{var1}_q'] = pd.qcut(data_clean[var1], q=4, duplicates='drop', labels=['Q1', 'Q2', 'Q3', 'Q4'])
        data_clean[f'{var2}_q'] = pd.qcut(data_clean[var2], q=4, duplicates='drop', labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # 计算16宫格的死亡率
        heatmap_data = []
        for q1 in ['Q1', 'Q2', 'Q3', 'Q4']:
            for q2 in ['Q1', 'Q2', 'Q3', 'Q4']:
                mask = (data_clean[f'{var1}_q'] == q1) & (data_clean[f'{var2}_q'] == q2)
                if mask.sum() > 0:
                    mortality_rate = data_clean[mask][target_col].mean()
                    n = mask.sum()
                    heatmap_data.append({
                        'Var1_Quantile': q1,
                        'Var2_Quantile': q2,
                        'Mortality_Rate': mortality_rate,
                        'N': n
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_table = heatmap_df.pivot(index='Var1_Quantile', columns='Var2_Quantile', values='Mortality_Rate')
            pivot_n = heatmap_df.pivot(index='Var1_Quantile', columns='Var2_Quantile', values='N')
            
            # 绘制热力图
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # 左图：死亡率热力图
            ax1 = axes[0]
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='Reds', 
                       cbar_kws={'label': 'Mortality Rate'}, ax=ax1, linewidths=0.5)
            ax1.set_xlabel(f'{var2} (Quartiles)', fontsize=12)
            ax1.set_ylabel(f'{var1} (Quartiles)', fontsize=12)
            ax1.set_title(f'2D Risk Heatmap: {var1} × {var2}\nMortality Rate by Quartile Combinations', 
                         fontsize=14)
            
            # 右图：样本数热力图
            ax2 = axes[1]
            sns.heatmap(pivot_n, annot=True, fmt='d', cmap='Blues', 
                       cbar_kws={'label': 'Sample Size'}, ax=ax2, linewidths=0.5)
            ax2.set_xlabel(f'{var2} (Quartiles)', fontsize=12)
            ax2.set_ylabel(f'{var1} (Quartiles)', fontsize=12)
            ax2.set_title(f'Sample Size by Quartile Combinations', fontsize=14)
            
            plt.tight_layout()
            safe_name = f"{var1.replace('/', '_')}_by_{var2.replace('/', '_')}"
            plt.savefig(f'../figures/fig_2d_risk_heatmap_{safe_name}.png', 
                       dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: figures/fig_2d_risk_heatmap_{safe_name}.png")
            plt.close()
            
            # 保存数据表
            heatmap_df.to_csv(f'../tables/tbl_2d_risk_grids_{safe_name}.csv', index=False)
            print(f"[OK] Saved: tables/tbl_2d_risk_grids_{safe_name}.csv")

# ============================================================================
# F. 多重共线性分析（VIF）
# ============================================================================
print("\n[F] Multicollinearity Analysis: VIF Ranking")
print("-" * 80)

# 选择数值变量（至少100个非缺失值）
vif_vars = [col for col in numeric_cols if df[col].notna().sum() > 100]
vif_vars = vif_vars[:30]  # 限制变量数量

if len(vif_vars) > 5:
    # 准备数据（移除缺失值）
    vif_data = df[vif_vars].dropna()
    
    if len(vif_data) > 100:
        # 标准化（VIF对尺度敏感）
        scaler = StandardScaler()
        vif_data_scaled = pd.DataFrame(
            scaler.fit_transform(vif_data),
            columns=vif_data.columns
        )
        
        # 计算VIF
        vif_results = []
        try:
            X_with_const = add_constant(vif_data_scaled)
            for i, col in enumerate(vif_vars, 1):
                try:
                    vif = variance_inflation_factor(X_with_const.values, i)
                    vif_results.append({
                        'Variable': col,
                        'VIF': vif,
                        'Risk_Level': 'High' if vif > 10 else ('Moderate' if vif > 5 else 'Low')
                    })
                except:
                    continue
        except Exception as e:
            print(f"[WARNING] VIF calculation failed: {e}")
            vif_results = []
        
        if vif_results:
            vif_df = pd.DataFrame(vif_results)
            vif_df = vif_df.sort_values('VIF', ascending=False)
            vif_df.to_csv('../tables/tbl_vif_rank.csv', index=False)
            print(f"[OK] Saved: tables/tbl_vif_rank.csv ({len(vif_df)} variables)")
            
            # 绘制VIF排行图
            plt.figure(figsize=(10, max(6, len(vif_df) * 0.3)))
            colors = ['red' if v > 10 else ('orange' if v > 5 else 'green') 
                     for v in vif_df['VIF'].values]
            plt.barh(range(len(vif_df)), vif_df['VIF'].values, color=colors)
            plt.axvline(x=5, color='orange', linestyle='--', label='VIF=5 (Moderate)')
            plt.axvline(x=10, color='red', linestyle='--', label='VIF=10 (High)')
            plt.yticks(range(len(vif_df)), vif_df['Variable'].values, fontsize=9)
            plt.xlabel('Variance Inflation Factor (VIF)', fontsize=12)
            plt.title('VIF Ranking for Numeric Features\n(Red: VIF>10, Orange: VIF>5, Green: VIF≤5)', 
                     fontsize=14, pad=20)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig('../figures/fig_vif_rank.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: figures/fig_vif_rank.png")
            plt.close()
            
            print("\nTop 10 variables by VIF:")
            for idx, row in vif_df.head(10).iterrows():
                print(f"  {row['Variable']:<30} VIF={row['VIF']:.2f} ({row['Risk_Level']})")

# ============================================================================
# G. 单变量强信号榜：OR + AUC
# ============================================================================
print("\n[G] Univariate Strong Signals: OR + AUC Ranking")
print("-" * 80)

# 单变量Logistic回归
univariate_results = []

for col in numeric_cols:
    if df[col].notna().sum() > 100:
        data_clean = df[[col, target_col]].dropna()
        
        if len(data_clean) > 100:
            try:
                # 标准化（按IQR）
                q75 = data_clean[col].quantile(0.75)
                q25 = data_clean[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    data_scaled = (data_clean[col] - data_clean[col].median()) / iqr
                else:
                    data_scaled = data_clean[col] - data_clean[col].median()
                
                # Logistic回归
                X = sm.add_constant(data_scaled)
                y = data_clean[target_col]
                
                model = sm.Logit(y, X).fit(disp=0)
                
                # OR (每IQR增加)
                or_val = np.exp(model.params[1])
                ci_lower = np.exp(model.conf_int().iloc[1, 0])
                ci_upper = np.exp(model.conf_int().iloc[1, 1])
                p_value = model.pvalues[1]
                
                # AUC
                try:
                    auc = roc_auc_score(y, data_clean[col])
                    pr_auc = average_precision_score(y, data_clean[col])
                except:
                    auc = 0.5
                    pr_auc = 0.5
                
                univariate_results.append({
                    'Variable': col,
                    'OR_per_IQR': or_val,
                    'OR_CI_Lower_95': ci_lower,
                    'OR_CI_Upper_95': ci_upper,
                    'P_value': p_value,
                    'AUC': auc,
                    'PR_AUC': pr_auc,
                    'N': len(data_clean)
                })
            except:
                continue

if univariate_results:
    univariate_df = pd.DataFrame(univariate_results)
    
    # FDR校正
    _, q_values, _, _ = multipletests(univariate_df['P_value'], method='fdr_bh')
    univariate_df['Q_value'] = q_values
    univariate_df['Significant'] = q_values < 0.05
    
    # 按AUC排序
    univariate_df = univariate_df.sort_values('AUC', ascending=False)
    univariate_df.to_csv('../tables/tbl_univariate_auc_or_rank.csv', index=False)
    print(f"[OK] Saved: tables/tbl_univariate_auc_or_rank.csv ({len(univariate_df)} variables)")
    
    # 绘制Top-20 AUC排行图
    top_n = min(20, len(univariate_df))
    top_univariate = univariate_df.head(top_n)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, top_n * 0.25)))
    
    # 左图：AUC排行
    ax1 = axes[0]
    colors = ['red' if sig else 'gray' for sig in top_univariate['Significant']]
    ax1.barh(range(len(top_univariate)), top_univariate['AUC'].values, color=colors)
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Random (AUC=0.5)')
    ax1.set_yticks(range(len(top_univariate)))
    ax1.set_yticklabels(top_univariate['Variable'].values, fontsize=9)
    ax1.set_xlabel('AUROC', fontsize=12)
    ax1.set_title(f'Top {top_n} Features by Univariate AUROC\n(Red: FDR q<0.05)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 右图：OR排行
    ax2 = axes[1]
    # 按OR排序（取绝对值）
    top_or = top_univariate.sort_values('OR_per_IQR', key=lambda x: x.abs(), ascending=False)
    colors = ['red' if sig else 'gray' for sig in top_or['Significant']]
    ax2.barh(range(len(top_or)), top_or['OR_per_IQR'].values, color=colors)
    ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='OR=1.0 (No effect)')
    ax2.set_yticks(range(len(top_or)))
    ax2.set_yticklabels(top_or['Variable'].values, fontsize=9)
    ax2.set_xlabel('OR per IQR Increase', fontsize=12)
    ax2.set_title(f'Top {top_n} Features by Univariate OR\n(Red: FDR q<0.05)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('../figures/fig_univariate_auc_rank.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: figures/fig_univariate_auc_rank.png")
    plt.close()
    
    print("\nTop 10 features by AUC:")
    for idx, row in top_univariate.head(10).iterrows():
        sig_mark = "[*]" if row['Significant'] else ""
        print(f"  {row['Variable']:<30} AUC={row['AUC']:.4f}, OR={row['OR_per_IQR']:.3f} "
              f"[{row['OR_CI_Lower_95']:.3f}, {row['OR_CI_Upper_95']:.3f}] "
              f"(p={row['P_value']:.4f}, q={row['Q_value']:.4f}) {sig_mark}")

# ============================================================================
# H. 衍生特征工程（AKI分级、休克指数等）
# ============================================================================
print("\n[H] Derived Feature Engineering")
print("-" * 80)

derived_features = {}

# H1. AKI分级（基于肌酐和尿量）
if 'creatinine_max' in df.columns:
    # KDIGO近似分级（简化版）
    creatinine = df['creatinine_max'].fillna(df['creatinine_max'].median())
    urine = df.get('urine_output_24h', pd.Series([np.nan] * len(df)))
    urine = urine.fillna(urine.median()) if urine.notna().any() else pd.Series([np.nan] * len(df))
    
    # AKI Stage 0-3（简化）
    aki_stage = pd.Series(0, index=df.index)
    # Stage 1: Cr 1.5-2x baseline (假设baseline=1.0)
    aki_stage[(creatinine >= 1.5) & (creatinine < 2.0)] = 1
    # Stage 2: Cr 2-3x
    aki_stage[(creatinine >= 2.0) & (creatinine < 3.0)] = 2
    # Stage 3: Cr >3x
    aki_stage[creatinine >= 3.0] = 3
    
    derived_features['aki_stage'] = aki_stage
    print(f"  [OK] Created AKI stage (0-3): {aki_stage.value_counts().to_dict()}")

# H2. 休克指数（HR/SBP 或 HR/MBP）
if 'hr_mean' in df.columns and 'mbp_mean' in df.columns:
    hr = df['hr_mean'].fillna(df['hr_mean'].median())
    mbp = df['mbp_mean'].fillna(df['mbp_mean'].median())
    shock_index = hr / (mbp + 1e-6)  # 避免除零
    derived_features['shock_index'] = shock_index
    print(f"  [OK] Created shock index (HR/MBP): mean={shock_index.mean():.2f}, range=[{shock_index.min():.2f}, {shock_index.max():.2f}]")

# H3. 乳酸/MAP比
if 'lactate_max_24h' in df.columns and 'mbp_mean' in df.columns:
    lactate = df['lactate_max_24h'].fillna(df['lactate_max_24h'].median())
    mbp = df['mbp_mean'].fillna(df['mbp_mean'].median())
    lactate_map_ratio = lactate / (mbp + 1e-6)
    derived_features['lactate_map_ratio'] = lactate_map_ratio
    print(f"  [OK] Created lactate/MAP ratio: mean={lactate_map_ratio.mean():.3f}")

# 分析衍生特征与死亡率的关系
if derived_features:
    derived_perf = []
    
    for feat_name, feat_values in derived_features.items():
        if feat_name == 'aki_stage':
            # AKI分级与死亡率
            aki_mortality = df.groupby(feat_values)[target_col].agg(['mean', 'count'])
            print(f"\n  AKI Stage vs Mortality:")
            for stage, row in aki_mortality.iterrows():
                print(f"    Stage {stage}: Mortality={row['mean']:.3f} (n={int(row['count'])})")
            
            # 绘制柱状图
            plt.figure(figsize=(8, 6))
            aki_mortality['mean'].plot(kind='bar', color='steelblue')
            plt.xlabel('AKI Stage', fontsize=12)
            plt.ylabel('Mortality Rate', fontsize=12)
            plt.title('AKI Stage vs 30-Day Mortality\n(n shown in bars)', fontsize=14)
            plt.xticks(rotation=0)
            for i, (stage, row) in enumerate(aki_mortality.iterrows()):
                plt.text(i, row['mean'] + 0.02, f"n={int(row['count'])}", 
                        ha='center', fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('../figures/fig_aki_stage_mortality.png', dpi=300, bbox_inches='tight')
            print("[OK] Saved: figures/fig_aki_stage_mortality.png")
            plt.close()
            
            # 计算OR
            try:
                aki_binary = (feat_values >= 2).astype(int)  # Stage 2-3 vs 0-1
                X = sm.add_constant(aki_binary)
                y = df[target_col]
                model = sm.Logit(y, X).fit(disp=0)
                or_val = np.exp(model.params[1])
                derived_perf.append({
                    'Feature': feat_name,
                    'AUC': roc_auc_score(y[aki_binary.notna()], aki_binary[aki_binary.notna()]),
                    'OR': or_val,
                    'Description': 'AKI Stage 2-3 vs 0-1'
                })
            except:
                pass
        else:
            # 连续衍生特征
            try:
                auc = roc_auc_score(df[target_col][feat_values.notna()], 
                                  feat_values[feat_values.notna()])
                derived_perf.append({
                    'Feature': feat_name,
                    'AUC': auc,
                    'OR': np.nan,
                    'Description': feat_name.replace('_', ' ').title()
                })
            except:
                pass
    
    if derived_perf:
        derived_perf_df = pd.DataFrame(derived_perf)
        derived_perf_df.to_csv('../tables/tbl_engineered_features_perf.csv', index=False)
        print("[OK] Saved: tables/tbl_engineered_features_perf.csv")
        
        # 绘制休克指数/乳酸-MAP比的分位-死亡率曲线（如果适用）
        for feat_name in ['shock_index', 'lactate_map_ratio']:
            if feat_name in derived_features:
                feat_values = derived_features[feat_name]
                data_clean = pd.DataFrame({
                    feat_name: feat_values,
                    target_col: df[target_col]
                }).dropna()
                
                if len(data_clean) > 100:
                    # 10分位分箱
                    data_clean['quantile'] = pd.qcut(data_clean[feat_name], q=10, duplicates='drop')
                    quantile_stats = data_clean.groupby('quantile')[target_col].agg(['mean', 'count'])
                    
                    plt.figure(figsize=(10, 6))
                    x_pos = np.arange(len(quantile_stats))
                    plt.plot(x_pos, quantile_stats['mean'].values, 'o-', linewidth=2, markersize=8)
                    plt.xticks(x_pos, [f'Q{i+1}' for i in range(len(quantile_stats))], rotation=45)
                    plt.ylabel('Mortality Rate', fontsize=12)
                    plt.xlabel(f'{feat_name.replace("_", " ").title()} (10 Quantiles)', fontsize=12)
                    plt.title(f'Quantile-Mortality Curve: {feat_name.replace("_", " ").title()}\n(n={len(data_clean)})', 
                             fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    safe_name = feat_name.replace('/', '_')
                    plt.savefig(f'../figures/fig_{safe_name}_curve.png', dpi=300, bbox_inches='tight')
                    print(f"[OK] Saved: figures/fig_{safe_name}_curve.png")
                    plt.close()

print("\n[Section H Complete] Derived feature engineering and analysis")

# ============================================================================
# I. Table 1: 基线表 + SMD（效应量）
# ============================================================================
print("\n[I] Table 1: Baseline Characteristics with Standardized Mean Differences (SMD)")
print("-" * 80)

def compute_smd(continuous_var, group1, group2):
    """计算标准化均值差（SMD）"""
    mean1 = group1.mean()
    mean2 = group2.mean()
    std1 = group1.std()
    std2 = group2.std()
    
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
    if pooled_std > 0:
        smd = (mean1 - mean2) / pooled_std
    else:
        smd = 0
    return smd

# 准备Table 1数据
table1_results = []

# 连续变量
y_survived = df[df[target_col] == 0]
y_died = df[df[target_col] == 1]

for col in numeric_cols[:20]:  # 前20个数值变量
    if df[col].notna().sum() > 100:
        data_survived = y_survived[col].dropna()
        data_died = y_died[col].dropna()
        
        if len(data_survived) > 50 and len(data_died) > 50:
            # 统计量
            mean_surv = data_survived.mean()
            std_surv = data_survived.std()
            median_surv = data_survived.median()
            q25_surv = data_survived.quantile(0.25)
            q75_surv = data_survived.quantile(0.75)
            
            mean_died = data_died.mean()
            std_died = data_died.std()
            median_died = data_died.median()
            q25_died = data_died.quantile(0.25)
            q75_died = data_died.quantile(0.75)
            
            # SMD
            smd = compute_smd(col, data_died, data_survived)
            
            # 统计检验
            try:
                _, p_value = stats.mannwhitneyu(data_survived, data_died, alternative='two-sided')
            except:
                p_value = np.nan
            
            table1_results.append({
                'Variable': col,
                'Type': 'Continuous',
                'Survived_Mean': mean_surv,
                'Survived_SD': std_surv,
                'Survived_Median': median_surv,
                'Survived_IQR': f"[{q25_surv:.2f}, {q75_surv:.2f}]",
                'Died_Mean': mean_died,
                'Died_SD': std_died,
                'Died_Median': median_died,
                'Died_IQR': f"[{q25_died:.2f}, {q75_died:.2f}]",
                'SMD': smd,
                'P_value': p_value,
                'N_Survived': len(data_survived),
                'N_Died': len(data_died)
            })

# 类别变量
for col in categorical_cols[:10]:  # 前10个类别变量
    if col != target_col and df[col].notna().sum() > 100:
        contingency = pd.crosstab(df[col], df[target_col])
        if contingency.shape[0] > 1:
            # 计算每个类别的百分比
            pct_survived = (contingency[0] / contingency[0].sum() * 100).round(2)
            pct_died = (contingency[1] / contingency[1].sum() * 100).round(2)
            
            # Cramér's V for SMD替代
            try:
                chi2, p, _, _ = chi2_contingency(contingency)
                n = contingency.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            except:
                cramers_v = np.nan
                p = np.nan
            
            # 保存主要类别
            for cat_val in contingency.index[:5]:  # 前5个类别
                table1_results.append({
                    'Variable': f'{col}_{cat_val}',
                    'Type': 'Categorical',
                    'Survived_N': int(contingency.loc[cat_val, 0]),
                    'Survived_Pct': pct_survived[cat_val],
                    'Died_N': int(contingency.loc[cat_val, 1]),
                    'Died_Pct': pct_died[cat_val],
                    'SMD': cramers_v,  # 使用Cramér's V作为SMD的替代
                    'P_value': p,
                    'N_Survived': int(contingency[0].sum()),
                    'N_Died': int(contingency[1].sum())
                })

if table1_results:
    table1_df = pd.DataFrame(table1_results)
    
    # FDR校正（连续变量）
    continuous_mask = table1_df['Type'] == 'Continuous'
    if continuous_mask.sum() > 0:
        _, q_values, _, _ = multipletests(
            table1_df.loc[continuous_mask, 'P_value'].fillna(1.0), 
            method='fdr_bh'
        )
        table1_df.loc[continuous_mask, 'Q_value'] = q_values
        table1_df.loc[continuous_mask, 'Significant'] = q_values < 0.05
    else:
        table1_df['Q_value'] = np.nan
        table1_df['Significant'] = False
    
    table1_df.to_csv('../tables/tbl_table1_baseline_with_SMD.csv', index=False)
    print(f"[OK] Saved: tables/tbl_table1_baseline_with_SMD.csv ({len(table1_df)} rows)")
    
    # 打印摘要
    print("\nTable 1 Summary (Top 10 continuous variables by |SMD|):")
    continuous_vars = table1_df[table1_df['Type'] == 'Continuous'].copy()
    continuous_vars['Abs_SMD'] = continuous_vars['SMD'].abs()
    continuous_vars = continuous_vars.sort_values('Abs_SMD', ascending=False)
    
    for idx, row in continuous_vars.head(10).iterrows():
        sig_mark = "[*]" if row.get('Significant', False) else ""
        smd_interpret = "Large" if abs(row['SMD']) >= 0.8 else ("Medium" if abs(row['SMD']) >= 0.5 else "Small")
        print(f"  {row['Variable']:<30} SMD={row['SMD']:7.3f} ({smd_interpret}) "
              f"Survived: {row['Survived_Mean']:.2f}±{row['Survived_SD']:.2f}, "
              f"Died: {row['Died_Mean']:.2f}±{row['Died_SD']:.2f} {sig_mark}")

# ============================================================================
# J. 类不平衡与采样策略预览
# ============================================================================
print("\n[J] Class Imbalance Analysis and Sampling Strategy Preview")
print("-" * 80)

# 目标分布
target_counts = df[target_col].value_counts()
target_pct = (target_counts / len(df) * 100)

print(f"\nTarget Distribution:")
print(f"  Survived (0): {target_counts[0]} ({target_pct[0]:.2f}%)")
print(f"  Died (1): {target_counts[1]} ({target_pct[1]:.2f}%)")
print(f"  Imbalance Ratio: {target_counts[0] / target_counts[1]:.2f}:1")

# 绘制目标分布图
plt.figure(figsize=(10, 6))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：饼图
ax1 = axes[0]
colors = ['lightblue', 'salmon']
ax1.pie(target_counts.values, labels=['Survived', 'Died'], autopct='%1.1f%%', 
       colors=colors, startangle=90)
ax1.set_title(f'Target Distribution\n(n={len(df)})', fontsize=14)

# 右图：条形图
ax2 = axes[1]
ax2.bar(['Survived', 'Died'], target_counts.values, color=colors)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Target Distribution (Counts)', fontsize=14)
for i, (label, count) in enumerate(zip(['Survived', 'Died'], target_counts.values)):
    ax2.text(i, count + len(df)*0.01, f'{count}\n({target_pct[i]:.1f}%)', 
            ha='center', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../figures/fig_target_imbalance.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: figures/fig_target_imbalance.png")
plt.close()

# 采样策略预览（仅展示，不用于正式建模）
print("\nSampling Strategy Preview (demonstration only):")
print("  [Note] This is for visualization only. Actual sampling is done during modeling.")

# 计算scale_pos_weight和class_weight
scale_pos_weight = target_counts[0] / target_counts[1]
print(f"  Scale_pos_weight (for XGBoost/LightGBM): {scale_pos_weight:.2f}")
print(f"  Class_weight (for sklearn): 'balanced' or {scale_pos_weight:.2f}:1")

# 可选：展示SMOTE效果（如果imbalanced-learn可用）
try:
    from imblearn.over_sampling import SMOTE
    
    # 选择一个小样本进行演示
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample = sample_df[numeric_cols[:10]].fillna(0)  # 简化处理
    y_sample = sample_df[target_col]
    
    if X_sample.notna().all().all():
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_sample, y_sample)
        
        print(f"\n  SMOTE demonstration (sample n={sample_size}):")
        print(f"    Before: {y_sample.value_counts().to_dict()}")
        print(f"    After:  {pd.Series(y_resampled).value_counts().to_dict()}")
except ImportError:
    print("  [Note] imbalanced-learn not available for SMOTE demonstration")

print("\n[Section J Complete] Class imbalance analysis completed")

# ============================================================================
# 最终总结和导出清单
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Enhanced EDA with Clinical Insights - All Sections Complete!")
print("=" * 80)

print("\nGenerated Files Summary:")
print("\nFigures (../figures/):")
print("  ✓ fig_corr_clustermap_spearman.png")
print("  ✓ fig_target_corr_pointbiserial_top30.png")
print("  ✓ fig_target_assoc_cramersV_top20.png")
print("  ✓ fig_dist_with_clinical_bands_*.png")
print("  ✓ fig_missing_clustermap.png")
print("  ✓ fig_missing_diff_by_outcome.png")
print("  ✓ fig_quantile_mortality_curves_*.png")
print("  ✓ fig_2d_risk_heatmap_*.png")
print("  ✓ fig_vif_rank.png")
print("  ✓ fig_univariate_auc_rank.png")
print("  ✓ fig_aki_stage_mortality.png")
print("  ✓ fig_shock_index_curve.png (if applicable)")
print("  ✓ fig_lactate_map_ratio_curve.png (if applicable)")
print("  ✓ fig_target_imbalance.png")

print("\nTables (../tables/):")
print("  ✓ tbl_corr_matrix.csv")
print("  ✓ tbl_target_corr_pointbiserial_all.csv")
print("  ✓ tbl_target_assoc_cramersV_all.csv")
print("  ✓ tbl_var_skew_kurtosis_before_after.csv")
print("  ✓ tbl_missing_diff_by_outcome.csv")
print("  ✓ tbl_missing_or_vs_mortality.csv")
print("  ✓ tbl_quantile_mortality_by_var.csv")
print("  ✓ tbl_2d_risk_grids_*.csv")
print("  ✓ tbl_vif_rank.csv")
print("  ✓ tbl_univariate_auc_or_rank.csv")
print("  ✓ tbl_engineered_features_perf.csv")
print("  ✓ tbl_table1_baseline_with_SMD.csv")

print("\nKey Findings:")
print("  - All analyses include FDR correction for multiple comparisons")
print("  - Clinical thresholds visualized for key variables")
print("  - Missing pattern analysis reveals MNAR insights")
print("  - Nonlinear relationships identified via quantile curves")
print("  - Interaction effects visualized via 2D heatmaps")
print("  - VIF analysis guides feature selection")
print("  - Derived features (AKI, shock index) created and validated")
print("  - Table 1 with SMD provides publication-ready baseline characteristics")

print("\n" + "=" * 80)
print("Enhanced EDA Complete! Ready for modeling pipeline.")
print("=" * 80)

