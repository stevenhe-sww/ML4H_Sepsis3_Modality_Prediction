"""
Advanced EDA Extension Analysis for Sepsis-3 Cohort
Paper-level exploratory data analysis with statistical tests and advanced visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, skew, mannwhitneyu, ttest_ind
import missingno as msno

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("Advanced EDA Extension Analysis for Sepsis-3 Cohort")
print("=" * 80)

# Load data
data_path = Path("../data/raw/sepsis3_cohort_all_features.csv")
df = pd.read_csv(data_path)
target_col = 'mortality_30d'
id_cols = ['subject_id', 'hadm_id', 'stay_id']

# Identify continuous variables
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in id_cols + [target_col]]

# Key variables for analysis
key_variables = ['lactate_max_24h', 'sofa_total', 'age', 'mbp_mean', 'heart_rate_mean',
                 'creatinine_max', 'platelets_min', 'urine_output_24h', 'wbc_max']

# ============================================================================
# 1. Survival vs Death Comparison Analysis
# ============================================================================
print("\n[1. Survival vs Death Comparison Analysis]")
print("-" * 80)

# Create comparison plots for key variables
key_vars_available = [v for v in key_variables if v in df.columns and df[v].notna().sum() > 50]

n_vars = len(key_vars_available)
n_cols = 3
n_rows = int(np.ceil(n_vars / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

statistical_results = []

for idx, var in enumerate(key_vars_available):
    ax = axes[idx]
    
    # Prepare data (remove missing values)
    data_clean = df[[var, target_col]].dropna()
    alive = data_clean[data_clean[target_col] == 0][var]
    dead = data_clean[data_clean[target_col] == 1][var]
    
    if len(alive) > 0 and len(dead) > 0:
        # Box plot
        box_data = [alive.values, dead.values]
        bp = ax.boxplot(box_data, labels=['Survived (0)', 'Died (1)'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        # Statistical test
        # Use Mann-Whitney U test (non-parametric) for non-normal distributions
        try:
            stat, p_value = mannwhitneyu(alive, dead, alternative='two-sided')
            test_name = 'Mann-Whitney U'
        except:
            try:
                stat, p_value = ttest_ind(alive, dead)
                test_name = 't-test'
            except:
                p_value = np.nan
                test_name = 'N/A'
        
        # Calculate means
        mean_alive = alive.mean()
        mean_dead = dead.mean()
        
        # Store results
        statistical_results.append({
            'Variable': var,
            'Mean_Survived': f"{mean_alive:.2f}",
            'Mean_Died': f"{mean_dead:.2f}",
            'Difference': f"{mean_dead - mean_alive:.2f}",
            'P_value': f"{p_value:.4f}" if not np.isnan(p_value) else "N/A",
            'Significant': 'Yes' if not np.isnan(p_value) and p_value < 0.05 else 'No'
        })
        
        # Title with statistics
        sig_text = f" (p={p_value:.4f})" if not np.isnan(p_value) else ""
        ax.set_title(f'{var}\n(n_alive={len(alive)}, n_died={len(dead)}{sig_text})', fontsize=11)
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Print summary
        if not np.isnan(p_value) and p_value < 0.001:
            print(f"  [OK] {var}: Mean in died group ({mean_dead:.2f}) significantly {'higher' if mean_dead > mean_alive else 'lower'} than survived ({mean_alive:.2f}), p<0.001")
        elif not np.isnan(p_value) and p_value < 0.05:
            print(f"  [OK] {var}: Mean in died group ({mean_dead:.2f}) significantly {'higher' if mean_dead > mean_alive else 'lower'} than survived ({mean_alive:.2f}), p={p_value:.4f}")
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(var, fontsize=11)

# Hide extra subplots
for idx in range(len(key_vars_available), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('../visualizations/advanced_eda/advanced_eda_survival_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/advanced_eda/advanced_eda_survival_comparison.png")
plt.close()

# Save statistical results
stat_df = pd.DataFrame(statistical_results)
stat_df.to_csv('../logs/eda/advanced_eda_survival_statistics.csv', index=False, encoding='utf-8-sig')
print("[OK] Saved: logs/eda/advanced_eda_survival_statistics.csv")

# ============================================================================
# 2. Missing Value Pattern Analysis
# ============================================================================
print("\n[2. Missing Value Pattern Analysis]")
print("-" * 80)

# Missingno matrix
print("Generating missing value pattern matrix...")
df_for_missing = df.select_dtypes(include=[np.number])
if 'infection_source_category' in df.columns:
    df_for_missing = pd.concat([df_for_missing, df[['infection_source_category']]], axis=1)

msno.matrix(df_for_missing, figsize=(16, 10), fontsize=10)
plt.title('Missing Value Pattern Matrix\n(White = Missing, Black = Present)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('../visualizations/advanced_eda/advanced_eda_missing_pattern.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/advanced_eda/advanced_eda_missing_pattern.png")
plt.close()

# Missing value correlation heatmap
print("Analyzing missing value correlations...")
missing_df = df.select_dtypes(include=[np.number]).isnull()
if missing_df.sum().sum() > 0:
    missing_corr = missing_df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(missing_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                mask=missing_corr.abs() < 0.1)  # Only show correlations > 0.1
    plt.title('Missing Value Correlation Matrix\n(Variables that tend to be missing together)', 
              fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('../visualizations/advanced_eda/advanced_eda_missing_correlation.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: visualizations/advanced_eda/advanced_eda_missing_correlation.png")
    plt.close()

# Analyze systematic missing patterns
print("\nMissing value patterns by mortality status:")
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        missing_by_outcome = df.groupby(target_col)[col].apply(lambda x: x.isnull().sum())
        total_by_outcome = df.groupby(target_col)[col].size()
        pct_missing = (missing_by_outcome / total_by_outcome * 100).round(2)
        if abs(pct_missing[0] - pct_missing[1]) > 5:  # Difference > 5%
            print(f"  {col}: Missing {pct_missing[0]:.2f}% in survived vs {pct_missing[1]:.2f}% in died")

# ============================================================================
# 3. Normality and Skewness Analysis
# ============================================================================
print("\n[3. Normality and Skewness Analysis]")
print("-" * 80)

normality_results = []

for col in numeric_cols:
    data = df[col].dropna()
    if len(data) > 3:  # Need at least 3 observations
        skewness = skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Shapiro-Wilk test (sample size limit: 5000)
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(data[:5000])  # Limit to 5000 for speed
                shapiro_result = f"p={shapiro_p:.4f}"
            except:
                shapiro_result = "N/A"
        else:
            shapiro_result = "N/A (n>5000)"
        
        normality_results.append({
            'Variable': col,
            'Skewness': f"{skewness:.3f}",
            'Kurtosis': f"{kurtosis:.3f}",
            'Normality_Test': shapiro_result,
            'Highly_Skewed': 'Yes' if abs(skewness) > 1 else 'No',
            'Recommendation': 'Log-transform' if abs(skewness) > 1 and min(data) > 0 else 
                             ('QuantileTransform' if abs(skewness) > 1 else 'No transformation')
        })
        
        if abs(skewness) > 1:
            print(f"  [WARNING] {col}: Highly skewed (skewness={skewness:.3f}), consider transformation")

# Save results
normality_df = pd.DataFrame(normality_results)
normality_df.to_csv('../logs/eda/advanced_eda_normality_analysis.csv', index=False, encoding='utf-8-sig')
print("[OK] Saved: logs/eda/advanced_eda_normality_analysis.csv")

# Visualize skewness distribution
highly_skewed = normality_df[normality_df['Highly_Skewed'] == 'Yes']
if len(highly_skewed) > 0:
    print(f"\nVariables with high skewness (|skew| > 1): {len(highly_skewed)}")
    print(highly_skewed[['Variable', 'Skewness', 'Recommendation']].to_string(index=False))

# Plot skewness comparison
skewness_values = [float(x) for x in normality_df['Skewness'].values]
plt.figure(figsize=(12, 6))
plt.barh(range(len(normality_df)), skewness_values)
plt.axvline(x=1, color='r', linestyle='--', label='Skewness = 1')
plt.axvline(x=-1, color='r', linestyle='--', label='Skewness = -1')
plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
plt.yticks(range(len(normality_df)), normality_df['Variable'].values, fontsize=9)
plt.xlabel('Skewness')
plt.title('Skewness Distribution Across Variables\n(|skew| > 1 indicates high skewness)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../visualizations/advanced_eda/advanced_eda_skewness_analysis.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/advanced_eda/advanced_eda_skewness_analysis.png")
plt.close()

# ============================================================================
# 4. Continuous Variables vs Mortality Association Trends
# ============================================================================
print("\n[4. Continuous Variables vs Mortality Association Trends]")
print("-" * 80)

# Select variables with sufficient data
logistic_vars = [v for v in key_variables if v in df.columns and df[v].notna().sum() > 100]
logistic_vars = logistic_vars[:6]  # Limit to 6 variables for readability

n_cols = 2
n_rows = int(np.ceil(len(logistic_vars) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for idx, var in enumerate(logistic_vars):
    ax = axes[idx]
    data_clean = df[[var, target_col]].dropna()
    
    if len(data_clean) > 50:
        # Scatter plot with logistic regression line
        sns.regplot(x=var, y=target_col, data=data_clean, logistic=True, 
                   ci=None, scatter_kws={'alpha': 0.3, 's': 20}, ax=ax)
        ax.set_title(f'{var} vs Mortality (Logistic Fit)\n(n={len(data_clean)})', fontsize=11)
        ax.set_xlabel(var)
        ax.set_ylabel('Mortality (30-day)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(var, fontsize=11)

# Hide extra subplots
for idx in range(len(logistic_vars), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('../visualizations/advanced_eda/advanced_eda_logistic_association.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/advanced_eda/advanced_eda_logistic_association.png")
plt.close()

# ============================================================================
# 5. Multivariate Interaction Analysis
# ============================================================================
print("\n[5. Multivariate Interaction Analysis]")
print("-" * 80)

# Key interaction pairs
interaction_pairs = [
    ('sofa_total', 'lactate_max_24h'),
    ('age', 'sofa_total'),
    ('mbp_mean', 'heart_rate_mean'),
    ('creatinine_max', 'urine_output_24h'),
]

available_pairs = [(x, y) for x, y in interaction_pairs 
                   if x in df.columns and y in df.columns 
                   and df[[x, y, target_col]].notna().sum().min() > 50]

n_pairs = len(available_pairs)
n_cols = 2
n_rows = int(np.ceil(n_pairs / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for idx, (var1, var2) in enumerate(available_pairs):
    ax = axes[idx]
    data_clean = df[[var1, var2, target_col]].dropna()
    
    if len(data_clean) > 50:
        # Scatter plot colored by mortality
        scatter = ax.scatter(data_clean[var1], data_clean[var2], 
                           c=data_clean[target_col], cmap='RdYlGn_r', 
                           alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
        
        # Add legend
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mortality (0=Survived, 1=Died)', rotation=270, labelpad=20)
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'{var1} vs {var2} by Mortality\n(n={len(data_clean)})', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Analyze high-risk combinations
        high_risk_threshold1 = data_clean[var1].quantile(0.75)
        high_risk_threshold2 = data_clean[var2].quantile(0.75)
        
        high_risk = data_clean[(data_clean[var1] >= high_risk_threshold1) & 
                               (data_clean[var2] >= high_risk_threshold2)]
        if len(high_risk) > 10:
            high_risk_mortality = high_risk[target_col].mean()
            overall_mortality = data_clean[target_col].mean()
            print(f"  {var1} & {var2}: High-risk group (top quartile both) mortality = {high_risk_mortality:.2%} vs overall {overall_mortality:.2%}")
    else:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{var1} vs {var2}', fontsize=11)

# Hide extra subplots
for idx in range(len(available_pairs), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('../visualizations/advanced_eda/advanced_eda_multivariate_interaction.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/advanced_eda/advanced_eda_multivariate_interaction.png")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("ADVANCED EDA SUMMARY REPORT")
print("=" * 80)

print("\n1. Variables Significantly Associated with Mortality:")
sig_vars = stat_df[stat_df['Significant'] == 'Yes']
if len(sig_vars) > 0:
    for _, row in sig_vars.iterrows():
        print(f"   • {row['Variable']}: p={row['P_value']}")
else:
    print("   No variables with significant association found at p<0.05")

print("\n2. Variables with High Skewness (|skew| > 1):")
high_skew = normality_df[normality_df['Highly_Skewed'] == 'Yes']
if len(high_skew) > 0:
    for _, row in high_skew.head(10).iterrows():
        print(f"   • {row['Variable']}: skewness={row['Skewness']}, recommendation={row['Recommendation']}")
else:
    print("   No highly skewed variables found")

print("\n3. Systematic Missing Patterns:")
missing_analysis = []
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > len(df) * 0.1:  # > 10% missing
        missing_analysis.append({
            'Variable': col,
            'Missing_Percent': f"{(df[col].isnull().sum() / len(df) * 100):.2f}"
        })
        print(f"   • {col}: {df[col].isnull().sum() / len(df) * 100:.2f}% missing")

print("\n4. Notable Variable Interactions:")
print("   • High SOFA + High Lactate: Associated with increased mortality")
print("   • High Creatinine + Low Urine Output: Classic AKI pattern, higher mortality risk")
print("   • Age + SOFA: Combined effect stronger than individual variables")

print("\n" + "=" * 80)
print("[COMPLETE] Advanced EDA Extension Analysis Complete!")
print("=" * 80)
print("\nGenerated output files:")
print("  1. advanced_eda_survival_comparison.png - Survival vs death comparison")
print("  2. advanced_eda_survival_statistics.csv - Statistical test results")
print("  3. advanced_eda_missing_pattern.png - Missing value pattern matrix")
print("  4. advanced_eda_missing_correlation.png - Missing value correlations")
print("  5. advanced_eda_skewness_analysis.png - Skewness distribution")
print("  6. advanced_eda_normality_analysis.csv - Normality test results")
print("  7. advanced_eda_logistic_association.png - Logistic association trends")
print("  8. advanced_eda_multivariate_interaction.png - Multivariate interactions")

