"""
Step 1 - Exploratory Data Analysis (EDA)
Comprehensive data exploration and visualization for sepsis3_cohort_all_features.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 80)
print("Step 1 - Exploratory Data Analysis (EDA)")
print("=" * 80)

# ============================================================================
# 1. Load data and print row/column counts and missing value proportions
# ============================================================================
print("\n[1. Basic Data Information]")
print("-" * 80)

data_path = Path("../data/raw/sepsis3_cohort_all_features.csv")
df = pd.read_csv(data_path)

print(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn list:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nMissing value statistics:")
missing_stats = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing Proportion (%)': (df.isnull().sum() / len(df) * 100).round(2)
})
missing_stats = missing_stats[missing_stats['Missing Count'] > 0].sort_values('Missing Proportion (%)', ascending=False)
if len(missing_stats) > 0:
    print(missing_stats.to_string())
else:
    print("  No missing values")

print(f"\nData types:")
print(df.dtypes.value_counts())

# ============================================================================
# 2. Identify continuous and categorical variables
# ============================================================================
# ID columns
id_cols = ['subject_id', 'hadm_id', 'stay_id']

# Target variable
target_col = 'mortality_30d'

# Numeric columns (excluding ID columns and target variable)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in id_cols + [target_col]]

# Categorical columns (including target variable)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Some numeric columns may also be categorical variables (e.g., binary variables)
binary_numeric = [col for col in numeric_cols if df[col].nunique() <= 10]
categorical_cols.extend(binary_numeric)

# Remove columns already classified as categorical from numeric_cols
numeric_cols = [col for col in numeric_cols if col not in binary_numeric]

print(f"\nVariable classification:")
print(f"  Continuous variables ({len(numeric_cols)}): {', '.join(numeric_cols)}")
print(f"  Categorical variables ({len(categorical_cols)}): {', '.join(categorical_cols)}")

# ============================================================================
# 3. Distribution plots for continuous variables (histogram / violin)
# ============================================================================
print("\n[2. Continuous Variable Distribution Visualization]")
print("-" * 80)

# Select top 12 continuous variables for detailed visualization (to avoid too many plots)
top_numeric = numeric_cols[:12] if len(numeric_cols) > 12 else numeric_cols

n_cols = 4
n_rows = int(np.ceil(len(top_numeric) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

for idx, col in enumerate(top_numeric):
    ax = axes[idx]
    data = df[col].dropna()
    if len(data) > 0:
        ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{col}\n(n={len(data)}, missing={df[col].isnull().sum()})', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(col, fontsize=10)

# Hide extra subplots
for idx in range(len(top_numeric), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('../visualizations/eda/eda_continuous_histograms.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: visualizations/eda/eda_continuous_histograms.png")
plt.close()

# Violin plot for selected key variables
key_vars = ['age', 'mbp_mean', 'heart_rate_mean', 'lactate_max_24h', 
            'sodium_min', 'platelets_min', 'creatinine_max', 'urine_output_24h']
key_vars = [v for v in key_vars if v in numeric_cols and df[v].notna().sum() > 0]

if len(key_vars) > 0:
    n_violin = min(8, len(key_vars))
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_vars[:n_violin]):
        data = df[col].dropna()
        if len(data) > 0:
            sns.violinplot(y=data, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{col}\n(n={len(data)})', fontsize=11)
            axes[idx].set_ylabel('Value')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    for idx in range(n_violin, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('../visualizations/eda/eda_continuous_violin.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: visualizations/eda/eda_continuous_violin.png")
    plt.close()

# ============================================================================
# 4. Frequency statistics for categorical variables
# ============================================================================
print("\n[3. Categorical Variable Frequency Statistics]")
print("-" * 80)

# Create categorical variable statistics plots
valid_categorical = [col for col in categorical_cols if col != target_col and df[col].notna().sum() > 0]
n_cat = min(8, len(valid_categorical))

if n_cat > 0:
    n_cols = 3
    n_rows = int(np.ceil(n_cat / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(valid_categorical[:n_cat]):
        ax = axes[idx]
        value_counts = df[col].value_counts()
        
        # If too many categories, only show top 10
        if len(value_counts) > 10:
            top_values = value_counts.head(10)
            ax.barh(range(len(top_values)), top_values.values)
            ax.set_yticks(range(len(top_values)))
            ax.set_yticklabels([str(x)[:30] for x in top_values.index], fontsize=9)
            ax.set_title(f'{col}\n(Showing top 10, total {len(value_counts)} categories)', fontsize=10)
        else:
            ax.barh(range(len(value_counts)), value_counts.values)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels([str(x)[:30] for x in value_counts.index], fontsize=9)
            ax.set_title(f'{col}\n(n={df[col].notna().sum()})', fontsize=10)
        
        ax.set_xlabel('Frequency')
        ax.grid(True, alpha=0.3, axis='x')
    
    for idx in range(n_cat, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('../visualizations/eda/eda_categorical_frequency.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: visualizations/eda/eda_categorical_frequency.png")
    plt.close()

# Print categorical variable statistics
for col in valid_categorical[:10]:  # Only print top 10
    print(f"\n{col}:")
    print(df[col].value_counts().head(10).to_string())

# ============================================================================
# 5. Correlation matrix visualization
# ============================================================================
print("\n[4. Correlation Matrix Visualization]")
print("-" * 80)

# Select numeric variables for correlation calculation
corr_vars = [col for col in numeric_cols if df[col].notna().sum() > 50]  # At least 50 non-missing values
if len(corr_vars) > 20:
    corr_vars = corr_vars[:20]  # Limit variable count for readability

if len(corr_vars) > 1:
    corr_matrix = df[corr_vars].corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Only show lower triangle
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f',
                xticklabels=[x[:15] for x in corr_vars], 
                yticklabels=[x[:15] for x in corr_vars])
    plt.title('Correlation Matrix for Continuous Variables (Pearson)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('../visualizations/eda/eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: visualizations/eda/eda_correlation_matrix.png")
    plt.close()
    
    # Find highly correlated variable pairs
    print("\nHighly correlated variable pairs (|r| > 0.7):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
            print(f"  {var1} ↔ {var2}: {corr:.3f}")
    else:
        print("  No highly correlated variable pairs found")

# ============================================================================
# 6. Outlier detection
# ============================================================================
print("\n[5. Outlier Detection]")
print("-" * 80)

outliers_found = []

# Check specified outlier rules
outlier_rules = [
    ('urine_output_24h', '<', 0, 'Negative urine output'),
    ('lactate_max_24h', '<', 0, 'Lactate < 0'),
    ('age', '>', 110, 'Age > 110'),
    ('sodium_min', '<', 100, 'Na < 100'),
    ('sodium_max', '>', 180, 'Na > 180'),
    ('sodium_min', '>', 180, 'Na_min > 180'),
    ('sodium_max', '<', 100, 'Na_max < 100'),
    ('platelets_min', '<', 10, 'Platelet < 10'),
    ('platelets_max', '>', 1000, 'Platelet > 1000'),
]

for col, op, threshold, desc in outlier_rules:
    if col in df.columns:
        if op == '<':
            outliers = df[df[col] < threshold].index.tolist()
        elif op == '>':
            outliers = df[df[col] > threshold].index.tolist()
        
        if outliers:
            outliers_found.append({
                'Variable': col,
                'Rule': desc,
                'Outlier Count': len(outliers),
                'Proportion (%)': f"{len(outliers)/len(df)*100:.2f}"
            })
            print(f"  [WARNING] {desc}: {len(outliers)} cases ({len(outliers)/len(df)*100:.2f}%)")
            print(f"     Example values: {df.loc[outliers[:5], col].tolist()}")

if not outliers_found:
    print("  [OK] No outliers found matching specified rules")

# IQR method for outlier detection (for main continuous variables)
print("\nOutlier detection using IQR method:")
iqr_outliers = {}
for col in ['age', 'mbp_mean', 'heart_rate_mean', 'lactate_max_24h', 
            'sodium_min', 'platelets_min', 'creatinine_max', 'urine_output_24h']:
    if col in df.columns and df[col].notna().sum() > 0:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        if outliers:
            iqr_outliers[col] = {
                'Count': len(outliers),
                'Proportion (%)': f"{len(outliers)/len(df)*100:.2f}",
                'Lower bound': f"{lower_bound:.2f}",
                'Upper bound': f"{upper_bound:.2f}"
            }
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%), "
                  f"range: [{lower_bound:.2f}, {upper_bound:.2f}]")

# ============================================================================
# 7. Summary statistics table
# ============================================================================
print("\n[6. Summary Statistics Table]")
print("-" * 80)

summary_stats = []
for col in numeric_cols:
    if df[col].notna().sum() > 0:
        summary_stats.append({
            'Variable': col,
            'Mean': f"{df[col].mean():.2f}",
            'Std': f"{df[col].std():.2f}",
            'Median': f"{df[col].median():.2f}",
            'Q1': f"{df[col].quantile(0.25):.2f}",
            'Q3': f"{df[col].quantile(0.75):.2f}",
            'IQR': f"{df[col].quantile(0.75) - df[col].quantile(0.25):.2f}",
            'Min': f"{df[col].min():.2f}",
            'Max': f"{df[col].max():.2f}",
            'Missing Count': int(df[col].isnull().sum()),
            'Missing Proportion (%)': f"{(df[col].isnull().sum() / len(df) * 100):.2f}"
        })

summary_df = pd.DataFrame(summary_stats)
print("\nContinuous variable summary statistics:")
print(summary_df.to_string(index=False))

# Save to CSV
summary_df.to_csv('../logs/eda/eda_summary_statistics.csv', index=False, encoding='utf-8-sig')
print("\n[OK] Saved: logs/eda/eda_summary_statistics.csv")

# ============================================================================
# 8. Target variable analysis
# ============================================================================
if target_col in df.columns:
    print("\n[7. Target Variable Analysis]")
    print("-" * 80)
    target_counts = df[target_col].value_counts()
    print(f"\n{target_col} distribution:")
    print(target_counts.to_string())
    print(f"\nProportions:")
    print((target_counts / len(df) * 100).round(2).to_string())
    
    # Target variable distribution plot
    plt.figure(figsize=(8, 6))
    target_counts.plot(kind='bar')
    plt.title(f'{target_col} Distribution', fontsize=14)
    plt.xlabel(target_col)
    plt.ylabel('Frequency')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('../visualizations/eda/eda_target_distribution.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: visualizations/eda/eda_target_distribution.png")
    plt.close()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Step 1 Exploratory Data Analysis Complete!")
print("=" * 80)
print("\nGenerated output files:")
print("  1. visualizations/eda/eda_continuous_histograms.png - Continuous variable distribution histograms")
print("  2. visualizations/eda/eda_continuous_violin.png - Continuous variable violin plots")
print("  3. visualizations/eda/eda_categorical_frequency.png - Categorical variable frequency plots")
print("  4. visualizations/eda/eda_correlation_matrix.png - Correlation matrix heatmap")
print("  5. visualizations/eda/eda_target_distribution.png - Target variable distribution plot")
print("  6. logs/eda/eda_summary_statistics.csv - Summary statistics table")
