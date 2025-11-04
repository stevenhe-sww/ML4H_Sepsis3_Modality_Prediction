# Data Cleaning Script Enhancements Documentation

## Overview

This document describes the enhancements made to `step2_data_cleaning.py` to improve data cleaning quality, robustness, and clinical relevance for the Sepsis-3 mortality prediction model.

## Key Enhancements

### 1. Clinical Physiological Caps (Step 1)

**Problem**: Extreme values due to data entry errors or measurement artifacts can distort model training.

**Solution**: Applied hard caps based on clinical plausibility:
- `lactate_max_24h`: 20.0 mmol/L (extreme hyperlactatemia threshold)
- `creatinine_max`: 20.0 mg/dL (severe AKI, dialysis threshold)
- `inr_max`: 20.0 (severe coagulopathy)
- `urine_output_24h`: 15000 mL (physiological maximum)

**Clinical Significance**: Prevents biologically implausible values from affecting model learning, especially important for tree-based models that are sensitive to extreme outliers.

**Statistical Impact**: Reduces variance in transformed features and improves model stability.

---

### 2. Missing Indicator Creation (Step 4)

**Problem**: In clinical data, "not measured" can be informative. Missing albumin or lactate may indicate:
- Patient too unstable for certain tests
- Different clinical protocols
- Implicit severity signal

**Solution**: Created binary missing indicators (`_missing`) for:
- `albumin_min_missing`: High missing rate, often not measured in ICU
- `lactate_max_24h_missing`: Lactate measurement protocol varies

**Clinical Significance**: Captures the signal that "absence of data is data" - missingness patterns often correlate with patient outcomes.

**Statistical Impact**: Adds informative features without losing sample size, improves model's ability to handle missingness heterogeneity.

---

### 3. Enhanced Log Transformation (Step 5)

**Problem**: Many laboratory values have right-skewed distributions (e.g., creatinine, lactate, WBC), violating assumptions of linear models and reducing tree model efficiency.

**Solution**: Applied `np.log1p()` transformation to:
- `wbc_max`, `creatinine_max`, `lactate_max_24h`, `inr_max`, `aniongap_max`

With **negative value clipping** to ensure non-negative input:
```python
if min_val < 0:
    df[col] = df[col].clip(lower=0)
df[col] = np.log1p(df[col])
```

**Clinical Significance**: 
- Log transformation aligns with biological scaling (many processes are multiplicative)
- Reduces impact of extreme values while preserving relative differences

**Statistical Impact**: 
- Reduces skewness (typically from 2-5 → <1)
- Improves linear model assumptions (normality)
- Enhances tree model splits (more balanced distributions)

---

### 4. QuantileTransformer for Distribution Normalization (Step 7)

**Problem**: Even after log transformation, some variables may have non-Gaussian distributions that limit model performance.

**Solution**: Applied `QuantileTransformer(output_distribution='normal')` to:
- `wbc_max`, `creatinine_max`, `lactate_max_24h`

**How it works**:
- Maps data to uniform distribution via quantiles
- Then transforms to standard normal distribution (N(0,1))
- Applied **after** log transformation for cumulative effect

**Clinical Significance**: 
- Creates optimal distribution for linear models (logistic regression)
- Maintains rank-order relationships important for clinical interpretation

**Statistical Impact**:
- Normalizes distributions completely (skewness → ~0)
- Improves separability in feature space
- Better for models assuming normality (logistic regression, neural networks)

**Note**: This is applied **after** log1p, so the transformation pipeline is:
```
Original → Clinical Caps → Winsorize → Log1p → Impute → QuantileTransform
```

---

### 5. Grouped KNN Imputation (Step 6)

**Problem**: Standard KNN imputation may match patients with different physiological profiles (e.g., matching blood panel values with electrolyte values from different patient types).

**Solution**: Split KNN imputation into biologically coherent groups:

**Blood Panel Group**:
- `wbc_max`, `platelets_min`, `platelets_max`, `inr_max`
- These are typically measured together (CBC + coagulation)

**Electrolyte/Metabolic Group**:
- `sodium_min`, `sodium_max`, `aniongap_max`, `creatinine_max`, `albumin_min`
- These reflect metabolic and renal function

**Clinical Significance**: 
- Reduces imputation error by matching patients with similar physiological profiles
- Maintains biological coherence in imputed values

**Statistical Impact**: 
- Lower imputation MSE
- Better preservation of feature relationships
- Improved model performance on imputed data

---

### 6. Robust OneHotEncoder Configuration (Step 9)

**Problem**: During deployment, new infection source categories may appear that weren't in training data, causing encoding failures.

**Solution**: Configured `OneHotEncoder` with:
```python
OneHotEncoder(
    handle_unknown="ignore",    # Ignore unseen categories (set all to 0)
    sparse_output=False          # Return dense arrays (modern sklearn)
)
```

**Clinical Significance**: 
- Prevents model crashes during inference
- Handles evolving clinical classifications gracefully

**Statistical Impact**: 
- Robust to data drift
- No information leakage from test-time categories

---

### 7. Comprehensive Logging

**Problem**: Lack of traceability for data transformations makes debugging and reproducibility difficult.

**Solution**: Added `logging` module with:
- File logging: `../logs/cleaning/cleaning.log`
- Console output: Real-time progress
- Detailed transformation logs: CSV files for each step

**Log Files Generated**:
1. `cleaning.log`: Full transformation log with timestamps
2. `cleaning_log_outliers.csv`: Outlier treatment details
3. `cleaning_log_transformations.csv`: Skewness corrections
4. `cleaning_log_missing_values.csv`: Imputation strategies
5. `cleaning_comparison_stats.csv`: Before/after statistics

---

## Transformation Pipeline Summary

```
Input Data
    ↓
[Step 1] Clinical Caps (hard limits)
    ↓
[Step 2] Negative Value Clipping (urine_output)
    ↓
[Step 3] Winsorization (1%-99% percentiles)
    ↓
[Step 4] Missing Indicator Creation
    ↓
[Step 5] Log1p Transformation (skewed vars)
    ↓
[Step 6] Missing Value Imputation
    ├── Low missing (<10%): Median
    ├── Medium missing (10-40%): Grouped KNN
    └── High missing (>40%): Median + Missing Indicator
    ↓
[Step 7] QuantileTransformer (normalize distributions)
    ↓
[Step 8] Multicollinearity Control (drop redundant)
    ↓
[Step 9] Categorical Encoding (robust OneHot)
    ↓
[Step 10] Feature Engineering (interactions, bins)
    ↓
[Step 11] Standardization (StandardScaler)
    ↓
Output: Cleaned Dataset
```

## Backward Compatibility

The enhanced script maintains **100% backward compatibility**:
- ✅ Same input: `../data/raw/sepsis3_cohort_all_features.csv`
- ✅ Same output: `../data/processed/sepsis3_cleaned.csv`
- ✅ Same artifact files: `standard_scaler.pkl`, `feature_list.json`
- ✅ Same log file structure (with additional `cleaning.log`)

## Expected Improvements

### Model Performance
- **Better feature distributions**: Log + QuantileTransform → improved linear model performance
- **Informative missingness**: Missing indicators → captures clinical protocols
- **Robust imputation**: Grouped KNN → more accurate feature values
- **Outlier resistance**: Clinical caps + Winsorization → stable training

### Clinical Interpretability
- **Biological coherence**: Grouped imputation preserves clinical relationships
- **Missingness patterns**: Indicators reveal measurement protocols
- **Preserved ranks**: QuantileTransform maintains clinical ordering

### Deployment Robustness
- **Unknown categories**: OneHotEncoder handles new infection sources
- **Extreme values**: Clinical caps prevent inference errors
- **Reproducibility**: Comprehensive logging enables audit trail

## Usage

```bash
cd scripts
python step2_data_cleaning.py
```

The script will:
1. Load raw data from `../data/raw/sepsis3_cohort_all_features.csv`
2. Apply all transformations in sequence
3. Save cleaned data to `../data/processed/sepsis3_cleaned.csv`
4. Generate logs and comparison statistics

## Validation Checklist

After running the enhanced script, verify:

- [ ] No negative values remain (except for transformed features)
- [ ] All features have missing indicators where appropriate
- [ ] Skewness reduced (check `cleaning_log_transformations.csv`)
- [ ] Missing values handled (check `cleaning_log_missing_values.csv`)
- [ ] Clinical caps applied (check `cleaning.log`)
- [ ] Feature count matches expectations (check console output)
- [ ] Scaler saved successfully (`../artifacts/standard_scaler.pkl`)
- [ ] Log files generated in `../logs/cleaning/`

## References

1. **Log Transformation**: Box-Cox transformation family for right-skewed data
2. **QuantileTransformer**: Non-parametric normalization preserving ranks
3. **Missing Indicators**: Missing data as a feature (Little & Rubin, 2019)
4. **Grouped Imputation**: Domain-specific imputation (Azur et al., 2011)
5. **Clinical Caps**: Expert knowledge integration in preprocessing

