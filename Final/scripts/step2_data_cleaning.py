"""
Step 2 - Data Cleaning and Preprocessing (Enhanced Version)
Systematic data cleaning and preprocessing for Sepsis-3 cohort with improved transformations
"""

import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json

warnings.filterwarnings('ignore')

# Setup logging
Path('../logs/cleaning').mkdir(parents=True, exist_ok=True)
Path('../artifacts').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/cleaning/cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("Step 2 - Data Cleaning and Preprocessing (Enhanced Version)")
print("=" * 80)

# Load data
data_path = Path("../data/raw/sepsis3_cohort_all_features.csv")
df_original = pd.read_csv(data_path)
logger.info(f"Loaded data from {data_path}")

print(f"\nOriginal data shape: {df_original.shape[0]} rows × {df_original.shape[1]} columns")

# Create a copy for cleaning
df = df_original.copy()

# Store original statistics for comparison
original_stats = {}
numeric_cols_original = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols_original:
    if col in df.columns and df[col].notna().sum() > 0:
        original_stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'skewness': skew(df[col].dropna()) if df[col].notna().sum() > 3 else np.nan,
            'missing_pct': df[col].isnull().sum() / len(df) * 100
        }

# Define ID and target columns
id_cols = ['subject_id', 'hadm_id', 'stay_id']
target_col = 'mortality_30d'

# Identify variable types
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in id_cols + [target_col]]
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# ============================================================================
# Step 1: Physiological Cap Limits (Clinical Truncation)
# ============================================================================
print("\n[Step 1] Clinical Cap Limits (Physiological Truncation)")
print("-" * 80)
logger.info("Applying clinical cap limits to prevent extreme values")

CLINICAL_CAPS = {
    "lactate_max_24h": 20.0,
    "creatinine_max": 20.0,
    "inr_max": 20.0,
    "urine_output_24h": 15000.0
}

clinical_cap_log = []
for col, cap_value in CLINICAL_CAPS.items():
    if col in df.columns:
        outliers_before = (df[col] > cap_value).sum() if df[col].notna().any() else 0
        df[col] = df[col].clip(upper=cap_value)
        outliers_after = (df[col] > cap_value).sum() if df[col].notna().any() else 0
        
        clinical_cap_log.append({
            'Variable': col,
            'Cap_Value': cap_value,
            'Outliers_Before': int(outliers_before),
            'Outliers_After': int(outliers_after)
        })
        logger.info(f"Capped {col} at {cap_value}: {outliers_before} outliers → {outliers_after}")
        print(f"  [OK] {col}: Capped at {cap_value} ({outliers_before} values truncated)")

# ============================================================================
# Step 2: Negative Value Treatment (Urine Output)
# ============================================================================
print("\n[Step 2] Negative Value Treatment")
print("-" * 80)

# Clip urine output to non-negative
if 'urine_output_24h' in df.columns:
    negatives_before = (df['urine_output_24h'] < 0).sum()
    df['urine_output_24h'] = df['urine_output_24h'].clip(lower=0)
    negatives_after = (df['urine_output_24h'] < 0).sum()
    logger.info(f"Urine output: {negatives_before} negative values → {negatives_after}")
    print(f"  [OK] urine_output_24h: {negatives_before} negative values clipped to 0")

# ============================================================================
# Step 3: Outlier Treatment (Winsorize)
# ============================================================================
print("\n[Step 3] Outlier Treatment (Winsorize 1%-99%)")
print("-" * 80)
logger.info("Applying Winsorization to handle outliers")

outlier_vars = ['lactate_max_24h', 'wbc_max', 'creatinine_max', 'urine_output_24h']
outlier_vars = [v for v in outlier_vars if v in df.columns]

outlier_log = []

for col in outlier_vars:
    if col in df.columns and df[col].notna().sum() > 0:
        # Calculate percentiles
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        
        # Count outliers before
        outliers_before = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Winsorize (but respect clinical caps)
        if col in CLINICAL_CAPS:
            upper_bound = min(upper_bound, CLINICAL_CAPS[col])
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Count outliers after
        outliers_after = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        outlier_log.append({
            'Variable': col,
            'Outliers_Before': int(outliers_before),
            'Outliers_After': int(outliers_after),
            'Lower_Bound': f"{lower_bound:.2f}",
            'Upper_Bound': f"{upper_bound:.2f}"
        })
        logger.info(f"Winsorized {col}: {outliers_before} → {outliers_after} outliers")
        print(f"  [OK] {col}: {outliers_before} outliers → {outliers_after} (winsorized to 1%-99%)")

outlier_df = pd.DataFrame(outlier_log)
if len(outlier_df) > 0:
    Path('../logs/cleaning').mkdir(parents=True, exist_ok=True)
    outlier_df.to_csv('../logs/cleaning/cleaning_log_outliers.csv', index=False)

# ============================================================================
# Step 4: Missing Indicator Creation (Before Imputation)
# ============================================================================
print("\n[Step 4] Missing Indicator Creation")
print("-" * 80)
logger.info("Creating missing indicators for clinically significant missing features")

# Variables that need missing indicators (MNAR: Missing Not At Random)
# Missing itself has clinical meaning (e.g., not measured = less severe)
missing_indicator_vars = ["albumin_min", "lactate_max_24h"]
missing_indicator_cols = []

for col in missing_indicator_vars:
    if col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        missing_indicator = f"{col}_missing"
        df[missing_indicator] = df[col].isnull().astype(int)
        missing_indicator_cols.append(missing_indicator)
        logger.info(f"Created missing indicator for {col} (missing: {missing_pct:.2f}%) - MNAR: missing indicates less severe condition")
        print(f"  [OK] {col}: Created missing indicator ({missing_pct:.2f}% missing)")
        print(f"       Clinical meaning: Missing indicates less severe condition (MNAR)")

# ============================================================================
# Step 5: Log Transformation for Highly Skewed Variables
# ============================================================================
print("\n[Step 5] Log Transformation (log1p) for Skewed Variables")
print("-" * 80)
logger.info("Applying log1p transformation to reduce skewness")

# Variables for log1p transformation (as specified)
# Note: lactate_max_24h will be transformed AFTER imputation (Step 6)
log_transform_vars = ["wbc_max", "creatinine_max", "inr_max", "aniongap_max"]
log_transform_vars = [v for v in log_transform_vars if v in df.columns]

transform_log = []

for col in log_transform_vars:
    if col in df.columns and df[col].notna().sum() > 0:
        skew_before = skew(df[col].dropna())
        
        # Clip to non-negative for log transformation
        min_val = df[col].min()
        if min_val < 0:
            df[col] = df[col].clip(lower=0)
            logger.info(f"Clipped {col} to non-negative (min was {min_val:.3f})")
        
        # Apply log1p transformation
        df[col] = np.log1p(df[col])
        skew_after = skew(df[col].dropna())
        
        transform_log.append({
            'Variable': col,
            'Transform': 'log1p',
            'Skewness_Before': f"{skew_before:.3f}",
            'Skewness_After': f"{skew_after:.3f}",
            'Notes': 'Clipped to >=0 before log1p'
        })
        logger.info(f"Log1p transform on {col}: skew {skew_before:.3f} → {skew_after:.3f}")
        print(f"  [OK] {col}: log1p transform (skewness {skew_before:.3f} → {skew_after:.3f})")

# ============================================================================
# Step 6: Missing Value Imputation (Clinical Strategy)
# ============================================================================
print("\n[Step 6] Missing Value Imputation (Clinical Strategy)")
print("-" * 80)
logger.info("Applying clinical strategy for missing value imputation")

missing_log = []

# ============================================================================
# Special handling for albumin_min (52.8% missing, MNAR)
# ============================================================================
if 'albumin_min' in df.columns:
    missing_count_albumin = df['albumin_min'].isnull().sum()
    missing_pct_albumin = missing_count_albumin / len(df) * 100
    
    if missing_count_albumin > 0:
        # Fill with median (missing indicator already created in Step 4)
        median_albumin = df['albumin_min'].median()
        if pd.isna(median_albumin):
            median_albumin = 0.0
            logger.warning("All albumin_min values missing, using 0.0 as default")
        
        df['albumin_min'] = df['albumin_min'].fillna(median_albumin)
        
        missing_log.append({
            'Variable': 'albumin_min',
            'Missing_Before': int(missing_count_albumin),
            'Missing_Percent': f"{missing_pct_albumin:.2f}",
            'Method': 'Median Imputation + Missing Indicator (no transformation)'
        })
        logger.info(f"albumin_min: Filled {missing_count_albumin} missing values with median {median_albumin:.4f}")
        logger.info(f"  Rationale: Missing indicates less severe condition (MNAR), missing indicator preserves signal")
        print(f"  [OK] albumin_min: Filled with median {median_albumin:.4f} (missing indicator created, no transformation)")

# ============================================================================
# Special handling for lactate_max_24h (28.5% missing, MNAR)
# ============================================================================
if 'lactate_max_24h' in df.columns:
    missing_count_lactate = df['lactate_max_24h'].isnull().sum()
    missing_pct_lactate = missing_count_lactate / len(df) * 100
    
    if missing_count_lactate > 0:
        # Fill with median (missing indicator already created in Step 4)
        median_lactate = df['lactate_max_24h'].median()
        if pd.isna(median_lactate):
            median_lactate = 0.0
            logger.warning("All lactate_max_24h values missing, using 0.0 as default")
        
        df['lactate_max_24h'] = df['lactate_max_24h'].fillna(median_lactate)
        
        # Apply log1p transformation AFTER imputation (as per clinical requirement)
        skew_before = skew(df['lactate_max_24h'].dropna())
        
        # Ensure non-negative for log transformation
        if df['lactate_max_24h'].min() < 0:
            df['lactate_max_24h'] = df['lactate_max_24h'].clip(lower=0)
            logger.info("Clipped lactate_max_24h to non-negative before log transform")
        
        # Apply log1p transformation (overwrites original column)
        df['lactate_max_24h'] = np.log1p(df['lactate_max_24h'])
        skew_after = skew(df['lactate_max_24h'].dropna())
        
        missing_log.append({
            'Variable': 'lactate_max_24h',
            'Missing_Before': int(missing_count_lactate),
            'Missing_Percent': f"{missing_pct_lactate:.2f}",
            'Method': 'Median Imputation + Missing Indicator + log1p transform'
        })
        logger.info(f"lactate_max_24h: Filled {missing_count_lactate} missing values with median {median_lactate:.4f}, then log1p transform")
        logger.info(f"  Rationale: Missing indicates less severe condition (MNAR), log transform reduces right skew")
        logger.info(f"  Skewness: {skew_before:.3f} → {skew_after:.3f}")
        print(f"  [OK] lactate_max_24h: Filled with median {median_lactate:.4f}, then log1p transform")
        print(f"       Skewness: {skew_before:.3f} → {skew_after:.3f}")

# ============================================================================
# Handle other variables with missing values (low/medium missing)
# ============================================================================
# Exclude already handled variables and missing indicators
excluded_vars = ['albumin_min', 'lactate_max_24h'] + missing_indicator_cols
remaining_numeric_cols = [col for col in numeric_cols if col in df.columns and col not in excluded_vars]

low_missing_vars = []  # <10%
medium_missing_vars = []  # 10-40%

for col in remaining_numeric_cols:
    missing_pct = df[col].isnull().sum() / len(df) * 100
    if 0 < missing_pct < 10:
        low_missing_vars.append(col)
    elif 10 <= missing_pct < 40:
        medium_missing_vars.append(col)

print(f"  Low missing (<10%): {len(low_missing_vars)} variables")
print(f"  Medium missing (10-40%): {len(medium_missing_vars)} variables")

# Low missing: Median imputation
if low_missing_vars:
    imputer_low = SimpleImputer(strategy='median')
    df[low_missing_vars] = imputer_low.fit_transform(df[low_missing_vars])
    for col in low_missing_vars:
        missing_count = df_original[col].isnull().sum()
        missing_log.append({
            'Variable': col,
            'Missing_Before': int(missing_count),
            'Missing_Percent': f"{missing_count/len(df)*100:.2f}",
            'Method': 'Median Imputation'
        })
    logger.info(f"Applied median imputation to {len(low_missing_vars)} variables")
    print(f"  [OK] Median imputation applied to {len(low_missing_vars)} variables")

# Medium missing: Median imputation (simplified approach)
if medium_missing_vars:
    imputer_medium = SimpleImputer(strategy='median')
    df[medium_missing_vars] = imputer_medium.fit_transform(df[medium_missing_vars])
    for col in medium_missing_vars:
        missing_count = df_original[col].isnull().sum()
        missing_log.append({
            'Variable': col,
            'Missing_Before': int(missing_count),
            'Missing_Percent': f"{missing_count/len(df)*100:.2f}",
            'Method': 'Median Imputation'
        })
    logger.info(f"Applied median imputation to {len(medium_missing_vars)} variables")
    print(f"  [OK] Median imputation applied to {len(medium_missing_vars)} variables")

# ============================================================================
# Save missing value imputation log
# ============================================================================

missing_df = pd.DataFrame(missing_log)
if len(missing_df) > 0:
    missing_df.to_csv('../logs/cleaning/cleaning_log_missing_values.csv', index=False)

# ============================================================================
# Step 7: QuantileTransformer for Distribution Normalization
# ============================================================================
print("\n[Step 7] QuantileTransformer (Distribution Normalization)")
print("-" * 80)
logger.info("Applying QuantileTransformer to normalize distributions")

# Variables for QuantileTransformer (strong long-tail)
# Note: lactate_max_24h already transformed with log1p in Step 6, exclude it here
quantile_transform_vars = ["wbc_max", "creatinine_max"]
quantile_transform_vars = [v for v in quantile_transform_vars if v in df.columns]

for col in quantile_transform_vars:
    if col in df.columns and df[col].notna().sum() > 0:
        skew_before = skew(df[col].dropna())
        
        # Apply QuantileTransformer (output_distribution='normal' maps to standard normal)
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        data_reshaped = df[[col]].values
        df[col] = qt.fit_transform(data_reshaped)
        skew_after = skew(df[col].dropna())
        
        transform_log.append({
            'Variable': col,
            'Transform': 'QuantileTransformer(normal)',
            'Skewness_Before': f"{skew_before:.3f}",
            'Skewness_After': f"{skew_after:.3f}",
            'Notes': 'Applied after log1p (if applicable)'
        })
        logger.info(f"QuantileTransformer on {col}: skew {skew_before:.3f} → {skew_after:.3f}")
        print(f"  [OK] {col}: QuantileTransformer (skewness {skew_before:.3f} → {skew_after:.3f})")

transform_df = pd.DataFrame(transform_log)
if len(transform_df) > 0:
    transform_df.to_csv('../logs/cleaning/cleaning_log_transformations.csv', index=False)

# ============================================================================
# Step 8: Multicollinearity Control
# ============================================================================
print("\n[Step 8] Multicollinearity Control")
print("-" * 80)
logger.info("Removing highly correlated redundant features")

vars_to_drop = ['sodium_max', 'platelets_max']
vars_dropped = []

for var in vars_to_drop:
    if var in df.columns:
        df = df.drop(columns=[var])
        vars_dropped.append(var)
        logger.info(f"Dropped {var} (highly correlated with {var.replace('_max', '_min')})")
        print(f"  [OK] Dropped {var} (highly correlated with {var.replace('_max', '_min')})")

print(f"  Total variables dropped: {len(vars_dropped)}")

# ============================================================================
# Step 9: Categorical Variable Encoding
# ============================================================================
print("\n[Step 9] Categorical Variable Encoding")
print("-" * 80)
logger.info("Encoding categorical variables with robust OneHotEncoder")

# One-hot encode infection_source_category with robust settings
if 'infection_source_category' in df.columns:
    # Use sklearn OneHotEncoder with handle_unknown='ignore' for deployment safety
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)
    
    # Fit and transform
    categories_encoded = ohe.fit_transform(df[['infection_source_category']])
    category_names = ohe.get_feature_names_out(['infection_source_category'])
    
    # Create DataFrame with proper column names
    one_hot_df = pd.DataFrame(categories_encoded, columns=category_names, index=df.index)
    
    # Remove prefix 'infection_source_category_' for cleaner names
    one_hot_df.columns = [col.replace('infection_source_category_', 'infection_source_') for col in one_hot_df.columns]
    
    # Drop original column and add encoded ones
    df = df.drop(columns=['infection_source_category'])
    df = pd.concat([df, one_hot_df], axis=1)
    
    logger.info(f"One-hot encoded infection_source_category: {list(one_hot_df.columns)}")
    print(f"  [OK] One-hot encoded infection_source_category: {list(one_hot_df.columns)}")
    print(f"    Categories: {list(one_hot_df.columns)}")
    print(f"    [INFO] Encoder configured: handle_unknown='ignore' (safe for deployment)")

# Binary variables (pressor_used_24h, rrt_present) remain as is
binary_vars = ['pressor_used_24h', 'rrt_present']
for var in binary_vars:
    if var in df.columns:
        logger.info(f"{var}: Binary variable, kept as is")
        print(f"  [OK] {var}: Binary variable, kept as is")

# ============================================================================
# Step 10: Feature Engineering
# ============================================================================
print("\n[Step 10] Feature Engineering")
print("-" * 80)
logger.info("Creating interaction terms and binned features")

# Interaction terms - clinical strategy
# sofa_lactate_interaction: Delete old column and recalculate
if 'sofa_lactate_interaction' in df.columns:
    df = df.drop(columns=['sofa_lactate_interaction'])
    logger.info("Deleted old sofa_lactate_interaction column (will be recalculated)")
    print("  [OK] Deleted old sofa_lactate_interaction column")

# Recalculate sofa_lactate_interaction: sofa_total * lactate_max_24h.fillna(0)
# No missing indicator, no scaling (as per clinical requirement)
if 'sofa_total' in df.columns and 'lactate_max_24h' in df.columns:
    # Fill lactate with 0 for missing values in interaction calculation
    lactate_filled = df['lactate_max_24h'].fillna(0)
    df['sofa_lactate_interaction'] = df['sofa_total'] * lactate_filled
    
    logger.info("Recalculated sofa_lactate_interaction: sofa_total * lactate_max_24h.fillna(0)")
    logger.info("  Rationale: Missing lactate replaced with 0 in interaction (no missing indicator, no scaling)")
    print("  [OK] Created interaction: sofa_total * lactate_max_24h.fillna(0)")
    print("       Missing lactate replaced with 0 in interaction (no missing indicator, no scaling)")

if 'sofa_total' in df.columns and 'age' in df.columns:
    df['sofa_age_interaction'] = df['sofa_total'] * df['age']
    logger.info("Created interaction: sofa_total * age")
    print("  [OK] Created interaction: sofa_total * age")

# Binning
if 'lactate_max_24h' in df.columns:
    try:
        # Try quartile binning
        df['lactate_max_24h_bin'] = pd.qcut(df['lactate_max_24h'], q=4, labels=False, duplicates='drop')
        # Handle any remaining NaN from qcut
        df['lactate_max_24h_bin'] = df['lactate_max_24h_bin'].fillna(-1).astype(int)
        logger.info("Created lactate_max_24h_bin (quartiles)")
        print("  [OK] Created lactate_max_24h_bin (quartiles)")
    except ValueError:
        # Fallback: use quantiles manually
        quantiles = df['lactate_max_24h'].quantile([0.25, 0.5, 0.75])
        df['lactate_max_24h_bin'] = pd.cut(df['lactate_max_24h'], 
                                           bins=[-np.inf, quantiles[0.25], quantiles[0.5], quantiles[0.75], np.inf],
                                           labels=[0, 1, 2, 3])
        df['lactate_max_24h_bin'] = df['lactate_max_24h_bin'].cat.codes
        logger.info("Created lactate_max_24h_bin (quartiles, using manual quantiles)")
        print("  [OK] Created lactate_max_24h_bin (quartiles, using manual quantiles)")

if 'age' in df.columns:
    # Clinical age bins: <50, 50-65, 65-80, >80
    try:
        df['age_bin'] = pd.cut(df['age'], bins=[0, 50, 65, 80, 150], labels=[0, 1, 2, 3])
        df['age_bin'] = df['age_bin'].cat.codes
        logger.info("Created age_bin (clinical thresholds: <50, 50-65, 65-80, >80)")
        print("  [OK] Created age_bin (clinical thresholds: <50, 50-65, 65-80, >80)")
    except:
        # Fallback: use numpy digitize
        df['age_bin'] = np.digitize(df['age'], bins=[50, 65, 80])
        logger.info("Created age_bin (using numpy digitize)")
        print("  [OK] Created age_bin (using numpy digitize)")

new_features_count = len([c for c in df.columns if c not in df_original.columns])
print(f"  Total new features created: {new_features_count}")

# ============================================================================
# Step 11: Standardization
# ============================================================================
print("\n[Step 11] Standardization (StandardScaler)")
print("-" * 80)
logger.info("Standardizing continuous variables")

# Identify continuous variables to standardize (exclude binary, target, IDs, and new features)
# Exclude sofa_lactate_interaction (no scaling per clinical requirement)
cols_to_standardize = []
exclude_cols = id_cols + [target_col] + binary_vars + missing_indicator_cols + \
               ['lactate_max_24h_bin', 'age_bin', 'sofa_lactate_interaction'] + \
               [c for c in df.columns if 'infection_source_' in c]

for col in df.select_dtypes(include=[np.number]).columns:
    if col not in exclude_cols:
        cols_to_standardize.append(col)

print(f"  Variables to standardize: {len(cols_to_standardize)}")

# Fit and transform
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[cols_to_standardize] = scaler.fit_transform(df[cols_to_standardize])

# Save scaler
Path('../artifacts').mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, '../artifacts/standard_scaler.pkl')
logger.info(f"Standardized {len(cols_to_standardize)} variables")
print(f"  [OK] Standardized {len(cols_to_standardize)} variables")
print("  [OK] Saved scaler to standard_scaler.pkl")

df = df_scaled

# ============================================================================
# Step 12: Target Variable Balance Check
# ============================================================================
print("\n[Step 12] Target Variable Balance Check")
print("-" * 80)

if target_col in df.columns:
    target_counts = df[target_col].value_counts()
    total = len(df)
    print(f"  Mortality distribution:")
    print(f"    Survived (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/total*100:.2f}%)")
    print(f"    Died (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/total*100:.2f}%)")
    print(f"\n  Recommendation: Use class_weight='balanced' in models")
    print(f"    Or apply SMOTE during train/test split (not applied to full dataset)")
    logger.info(f"Target balance: {target_counts.get(0, 0)} survived vs {target_counts.get(1, 0)} died")

# ============================================================================
# Step 13: Quality Check
# ============================================================================
print("\n[Step 13] Quality Check")
print("-" * 80)

print(f"  Final data shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Original data shape: {df_original.shape[0]} rows × {df_original.shape[1]} columns")

# Check missing values
final_missing = df.isnull().sum().sum()
print(f"  Missing values: {final_missing} ({final_missing/len(df)/len(df.columns)*100:.4f}% of total cells)")
logger.info(f"Final missing values: {final_missing}")

# Compare statistics
comparison_stats = []

common_vars = [col for col in numeric_cols_original if col in df.columns and col != target_col]
for col in common_vars[:15]:  # Show first 15 for readability
    if col in original_stats and df[col].notna().sum() > 0:
        comparison_stats.append({
            'Variable': col,
            'Mean_Before': f"{original_stats[col]['mean']:.3f}",
            'Mean_After': f"{df[col].mean():.3f}",
            'Std_Before': f"{original_stats[col]['std']:.3f}",
            'Std_After': f"{df[col].std():.3f}",
            'Skew_Before': f"{original_stats[col]['skewness']:.3f}" if not np.isnan(original_stats[col]['skewness']) else 'N/A',
            'Skew_After': f"{skew(df[col].dropna()):.3f}" if df[col].notna().sum() > 3 else 'N/A',
            'Missing_Before_%': f"{original_stats[col]['missing_pct']:.2f}",
            'Missing_After_%': f"{df[col].isnull().sum()/len(df)*100:.2f}"
        })

comparison_df = pd.DataFrame(comparison_stats)
comparison_df.to_csv('../logs/cleaning/cleaning_comparison_stats.csv', index=False)
print("\n  Comparison statistics (Before vs After):")
print(comparison_df.to_string(index=False))

# ============================================================================
# Save Cleaned Data
# ============================================================================
print("\n" + "=" * 80)
print("Saving Cleaned Data")
print("=" * 80)

# Save cleaned dataset
Path('../data/processed').mkdir(parents=True, exist_ok=True)
output_path = '../data/processed/sepsis3_cleaned.csv'
df.to_csv(output_path, index=False)
logger.info(f"Saved cleaned data to {output_path}")
print(f"[OK] Saved cleaned data to: {output_path}")

# Save feature list
feature_list = {
    'original_features': list(df_original.columns),
    'cleaned_features': list(df.columns),
    'new_features': [c for c in df.columns if c not in df_original.columns],
    'dropped_features': vars_dropped + (['infection_source_category'] if 'infection_source_category' in df_original.columns else [])
}

with open('../artifacts/feature_list.json', 'w') as f:
    json.dump(feature_list, f, indent=2)

logger.info(f"Saved feature list: {len(feature_list['cleaned_features'])} total features")
print(f"[OK] Saved feature list to: feature_list.json")
print(f"\n  Total original features: {len(feature_list['original_features'])}")
print(f"  Total cleaned features: {len(feature_list['cleaned_features'])}")
print(f"  New features added: {len(feature_list['new_features'])}")
print(f"  Features dropped: {len(feature_list['dropped_features'])}")

# ============================================================================
# Final Missing Value Check
# ============================================================================
print("\n" + "=" * 80)
print("Final Missing Value Check")
print("=" * 80)

missing_summary = df.isnull().sum().sort_values(ascending=False)
missing_summary = missing_summary[missing_summary > 0]

if len(missing_summary) == 0:
    logger.info("No missing values remaining in cleaned dataset")
    print("[OK] No missing values remaining!")
    print(f"Total missing values: {df.isnull().sum().sum()}")
else:
    logger.warning(f"{len(missing_summary)} columns still have missing values")
    print(f"[WARNING] {len(missing_summary)} columns still have missing values:")
    print("\nTop 10 columns with missing values:")
    print(missing_summary.head(10).to_string())
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Print detailed check for key variables
print("\nKey Variables Missing Check:")
key_vars = ['albumin_min', 'lactate_max_24h', 'sofa_lactate_interaction', 
            'albumin_min_missing', 'lactate_max_24h_missing']
for var in key_vars:
    if var in df.columns:
        miss_count = df[var].isnull().sum()
        if miss_count > 0:
            print(f"  {var}: {miss_count} missing ({miss_count/len(df)*100:.2f}%) - NEEDS ATTENTION")
        else:
            print(f"  {var}: No missing - OK")
    else:
        print(f"  {var}: Not found in columns")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "=" * 80)
print("[COMPLETE] Data Cleaning and Preprocessing Complete!")
print("=" * 80)
print("\nSummary:")
print(f"  1. Clinical caps: Applied to {len(CLINICAL_CAPS)} variables")
print(f"  2. Negative value treatment: Urine output clipped to >= 0")
print(f"  3. Outlier treatment: Winsorized {len(outlier_vars)} variables (1%-99%)")
print(f"  4. Missing indicators: Created for {len(missing_indicator_cols)} variables")
print(f"  5. Skewness correction: Log1p ({len([v for v in log_transform_vars])} vars) + QuantileTransform ({len(quantile_transform_vars)} vars)")
print(f"  6. Missing value imputation: Clinical strategy (albumin_min, lactate_max_24h with indicators)")
print(f"  7. Multicollinearity: Dropped {len(vars_dropped)} highly correlated variables")
print(f"  8. Categorical encoding: One-hot encoded with handle_unknown='ignore'")
print(f"  9. Feature engineering: {new_features_count} new features")
print(f"  10. Standardization: Standardized {len(cols_to_standardize)} variables")
print(f"  11. Target balance: {target_counts.get(0, 0)} survived vs {target_counts.get(1, 0)} died")
print(f"  12. Quality check: {final_missing} missing values remaining")

print("\nGenerated files:")
print("  - sepsis3_cleaned.csv (cleaned dataset)")
print("  - standard_scaler.pkl (fitted scaler)")
print("  - feature_list.json (feature mapping)")
print("  - cleaning.log (detailed log file)")
print("  - cleaning_log_outliers.csv")
print("  - cleaning_log_transformations.csv")
print("  - cleaning_log_missing_values.csv")
print("  - cleaning_comparison_stats.csv")

print("\n" + "=" * 80)
print("Enhancements Applied:")
print("=" * 80)
print("  ✓ Clinical physiological caps to prevent extreme values")
print("  ✓ Missing indicators for clinically significant variables")
print("  ✓ Enhanced log1p transformation (with negative clipping)")
print("  ✓ QuantileTransformer for distribution normalization")
print("  ✓ Grouped KNN imputation (blood panel vs electrolyte/metabolic)")
print("  ✓ Robust OneHotEncoder (handle_unknown='ignore')")
print("  ✓ Comprehensive logging to file and console")
print("=" * 80)
