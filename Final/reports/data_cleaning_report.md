# Data Cleaning and Preprocessing Report
## Sepsis-3 Cohort Dataset

---

## Executive Summary

This report documents the systematic data cleaning and preprocessing steps applied to the Sepsis-3 cohort dataset. The cleaning pipeline successfully processed 20,391 patients with 24 original features, resulting in a cleaned dataset with 30 features and zero missing values.

**Dataset**: `sepsis3_cohort_all_features.csv` → `sepsis3_cleaned.csv`  
**Final Shape**: 20,391 rows × 30 columns  
**Missing Values**: 0 (0.00%)  
**Target Distribution**: 76.94% survived, 23.06% died

---

## Step-by-Step Cleaning Process

### Step 1: Outlier Treatment (Winsorize 1%-99%)

**Method**: Winsorization to 1st and 99th percentiles

| Variable | Outliers Before | Outliers After | Lower Bound | Upper Bound |
|----------|----------------|----------------|-------------|-------------|
| lactate_max_24h | 267 | 0 | 0.70 | 15.10 |
| wbc_max | 394 | 0 | 2.10 | 48.70 |
| creatinine_max | 310 | 0 | 0.40 | 9.20 |
| urine_output_24h | 377 | 0 | 10.00 | 5756.00 |

**Impact**: Removed 1,348 extreme values that could bias model training.

---

### Step 2: Skewness Correction

**Methods Applied**:
- **Log1p Transformation** (5 variables): For variables with extreme positive skewness
- **QuantileTransformer** (2 variables): For variables that may have zero/negative values

#### Log1p Transformations

| Variable | Skewness Before | Skewness After | Improvement |
|----------|----------------|----------------|-------------|
| wbc_max | 10.143 → 1.403 | -0.304 | ✓ Normalized |
| creatinine_max | 6.996 → 2.516 | 1.303 | ✓ Reduced |
| inr_max | 6.084 | 2.622 | ✓ Reduced |
| platelets_min | 1.625 | -0.882 | ✓ Normalized |
| platelets_max | 1.838 | -0.684 | ✓ Normalized |

#### QuantileTransformer

| Variable | Skewness Before | Skewness After | Improvement |
|----------|----------------|----------------|-------------|
| lactate_max_24h | 2.748 | -0.485 | ✓ Normalized |
| urine_output_24h | 1.133 | -0.072 | ✓ Normalized |

**Impact**: Significantly improved distribution normality for linear models.

---

### Step 3: Missing Value Imputation

**Strategy**: Tiered approach based on missing percentage

#### Low Missing (<10%) - Median Imputation
- **Variables**: 17 variables
- **Method**: SimpleImputer with median strategy
- **Examples**: mbp_mean, sbp_mean, heart_rate_mean, sodium_min, etc.

#### Medium Missing (10-40%) - KNN Imputation
- **Variables**: 1 variable (lactate_max_24h)
- **Method**: KNNImputer with k=5 neighbors
- **Missing Rate**: 28.53%

#### High Missing (>40%) - Missing Indicators
- **Variables**: 1 variable (albumin_min)
- **Missing Rate**: 52.81%
- **Method**: 
  - Created `albumin_min_missing` indicator (1 if missing, 0 otherwise)
  - Filled missing values with median

**Final Result**: 0 missing values (100% complete dataset)

---

### Step 4: Multicollinearity Control

**Variables Dropped**:
- `sodium_max` (highly correlated with `sodium_min`, r=0.808)
- `platelets_max` (highly correlated with `platelets_min`, r=0.916)

**Rationale**: High correlation (>0.7) indicates redundant information that can cause multicollinearity in linear models.

---

### Step 5: Categorical Variable Encoding

#### One-Hot Encoding
- **Variable**: `infection_source_category`
- **Categories Encoded**:
  - `infection_source_blood`
  - `infection_source_other`
  - `infection_source_respiratory`
  - `infection_source_urine`

#### Binary Variables (Kept as-is)
- `pressor_used_24h` (0/1)
- `rrt_present` (0/1)

---

### Step 6: Feature Engineering

#### Interaction Terms
1. `sofa_lactate_interaction` = `sofa_total` × `lactate_max_24h`
   - Captures synergistic effect of high SOFA and high lactate

2. `sofa_age_interaction` = `sofa_total` × `age`
   - Captures interaction between disease severity and age

#### Binning
1. `lactate_max_24h_bin`: Quartile-based binning (Q1-Q4)
   - Q1: Lowest quartile
   - Q4: Highest quartile

2. `age_bin`: Clinical age categories
   - 0: <50 years
   - 1: 50-65 years
   - 2: 65-80 years
   - 3: >80 years

**Total New Features**: 9

---

### Step 7: Standardization

**Method**: StandardScaler (z-score normalization)

**Variables Standardized**: 17 continuous variables

**Result**: 
- Mean ≈ 0 for all standardized variables
- Standard deviation = 1 for all standardized variables

**Saved Artifact**: `standard_scaler.pkl` (for inference)

---

### Step 8: Target Variable Balance Check

**Distribution**:
- Survived (0): 15,688 (76.94%)
- Died (1): 4,703 (23.06%)

**Recommendation**: 
- Use `class_weight='balanced'` in scikit-learn models
- OR apply SMOTE during train/test split (not on full dataset)

**Imbalance Ratio**: ~3.3:1 (moderate imbalance)

---

### Step 9: Quality Check

#### Data Integrity
- ✅ No missing values remaining
- ✅ All features have valid numeric values
- ✅ Data shape maintained (20,391 rows)
- ✅ Target variable intact

#### Statistical Validation

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Missing Values | Variable | 0 | ✓ Resolved |
| Highly Skewed Variables | 9 | 2 | ✓ Reduced |
| Features | 24 | 30 | +6 new |
| Multicollinearity | 2 pairs | 0 | ✓ Resolved |

---

## Before vs After Comparison

### Selected Variables (Sample)

| Variable | Mean Before | Mean After | Std Before | Std After | Skew Before | Skew After |
|----------|-------------|------------|------------|-----------|-------------|------------|
| age | 65.168 | 0.000 | 15.877 | 1.000 | -0.588 | -0.588 |
| wbc_max | 15.936 | 0.000 | 11.914 | 1.000 | 10.143 | -0.304 |
| lactate_max_24h | 3.185 | -0.000 | 2.878 | 1.000 | 2.974 | -0.485 |
| creatinine_max | 1.804 | 0.000 | 1.865 | 1.000 | 6.996 | 1.304 |

**Note**: Mean ≈ 0 and Std = 1 after standardization is expected and correct.

---

## Feature Summary

### Original Features: 24
- 3 ID columns (subject_id, hadm_id, stay_id)
- 17 continuous variables
- 1 categorical variable (infection_source_category)
- 2 binary variables (pressor_used_24h, rrt_present)
- 1 target variable (mortality_30d)

### Cleaned Features: 30
- 3 ID columns (unchanged)
- 17 continuous variables (standardized)
- 4 one-hot encoded variables (infection_source)
- 1 missing indicator (albumin_min_missing)
- 2 interaction terms
- 2 binned variables
- 2 binary variables (unchanged)
- 1 target variable (unchanged)

### New Features Added: 9
1. albumin_min_missing
2. infection_source_blood
3. infection_source_other
4. infection_source_respiratory
5. infection_source_urine
6. sofa_lactate_interaction
7. sofa_age_interaction
8. lactate_max_24h_bin
9. age_bin

### Features Dropped: 3
1. sodium_max (multicollinearity)
2. platelets_max (multicollinearity)
3. infection_source_category (replaced by one-hot)

---

## Generated Artifacts

### Data Files
1. **sepsis3_cleaned.csv** - Cleaned and preprocessed dataset
2. **standard_scaler.pkl** - Fitted StandardScaler for inference
3. **feature_list.json** - Feature mapping (original → cleaned)

### Log Files
1. **cleaning_log_outliers.csv** - Outlier treatment log
2. **cleaning_log_transformations.csv** - Transformation log
3. **cleaning_log_missing_values.csv** - Missing value imputation log
4. **cleaning_comparison_stats.csv** - Before/after statistics

---

## Recommendations for Modeling

### 1. Feature Selection
- **Remove ID columns** before training
- **Consider feature importance** analysis (XGBoost, SHAP) to identify top predictors
- **Interaction terms** (sofa_lactate, sofa_age) may be important

### 2. Model Configuration
- **Class weights**: Use `class_weight='balanced'` for linear models
- **SMOTE**: Apply during train/test split (not to full dataset)
- **Cross-validation**: Use stratified k-fold to maintain class distribution

### 3. Model Types Suited for This Data
- **Linear Models** (Logistic Regression): ✓ Well-suited (standardized, normalized)
- **Tree-based Models** (Random Forest, XGBoost): ✓ Well-suited (handles interactions)
- **Neural Networks**: ✓ Well-suited (standardized features)

### 4. Validation Strategy
- Use **stratified train/validation/test split** (70-15-15)
- Ensure **temporal/patient-level** split if needed
- Monitor for **overfitting** due to class imbalance

---

## Limitations and Considerations

1. **Missing Indicators**: The `albumin_min_missing` indicator may carry predictive signal
2. **Transformations**: Log and QuantileTransformer need to be applied consistently in inference
3. **Binning**: Age and lactate bins are based on training data distributions
4. **Standardization**: All continuous features are standardized; binary/categorical features are not

---

## Next Steps

1. ✅ Data cleaning complete
2. ⏭️ Feature selection (optional)
3. ⏭️ Train/test split (stratified)
4. ⏭️ Model training
5. ⏭️ Model evaluation
6. ⏭️ Model interpretation (SHAP)

---

**Report Generated**: Data Cleaning and Preprocessing Pipeline  
**Date**: Processing completed  
**Status**: ✅ All cleaning steps completed successfully

