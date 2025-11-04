# Advanced EDA Summary Report
## Sepsis-3 Cohort Analysis

---

## Executive Summary

This report summarizes the advanced exploratory data analysis (EDA) conducted on the Sepsis-3 cohort dataset. The analysis includes survival comparison, missing value patterns, normality assessments, logistic associations, and multivariate interactions.

**Dataset**: `sepsis3_cohort_all_features.csv`  
**Sample Size**: 20,391 patients  
**Target Variable**: `mortality_30d` (30-day mortality)  
**Mortality Rate**: 23.06% (4,703 deaths / 20,391 total)

---

## 1. Survival vs Death Comparison Analysis

### Key Findings

All nine key clinical variables showed **highly significant differences** (p<0.001) between survived and died groups:

| Variable | Mean (Survived) | Mean (Died) | Difference | Significance |
|----------|----------------|-------------|------------|--------------|
| **lactate_max_24h** | 2.81 | 4.31 | +1.50 | p<0.001 ⬆️ |
| **sofa_total** | 5.87 | 8.41 | +2.53 | p<0.001 ⬆️ |
| **age** | 63.88 | 69.46 | +5.58 | p<0.001 ⬆️ |
| **heart_rate_mean** | 86.70 | 90.86 | +4.16 | p<0.001 ⬆️ |
| **creatinine_max** | 1.69 | 2.18 | +0.49 | p<0.001 ⬆️ |
| **wbc_max** | 15.46 | 17.51 | +2.05 | p<0.001 ⬆️ |
| **mbp_mean** | 77.22 | 74.85 | -2.37 | p<0.001 ⬇️ |
| **platelets_min** | 181.58 | 175.34 | -6.24 | p<0.001 ⬇️ |
| **urine_output_24h** | 1814.11 | 1263.59 | -550.52 | p<0.001 ⬇️ |

### Clinical Interpretation

- **Increased in death group**: Lactate, SOFA score, age, heart rate, creatinine, WBC
- **Decreased in death group**: Mean blood pressure, platelets, urine output

**These variables are strong predictors for mortality and should be prioritized in feature selection.**

---

## 2. Missing Value Pattern Analysis

### Systematic Missing Patterns

| Variable | Missing % | Pattern |
|----------|-----------|---------|
| **albumin_min** | 52.81% | Systematically missing (likely not measured for all patients) |
| **lactate_max_24h** | 28.53% | More missing in survived group (30.30%) vs died (22.65%) |
| **inr_max** | 5.48% | Relatively low missingness |

### Findings

- **Differential missingness**: Lactate and albumin show different missing rates between survival groups, suggesting potential selection bias
- **Non-random missing**: Missing patterns may indicate clinical decisions (e.g., sicker patients more likely to have labs drawn)
- **Recommendation**: Use advanced imputation methods (e.g., KNNImputer, MissForest) that account for missing patterns

---

## 3. Normality and Skewness Analysis

### Highly Skewed Variables (|skew| > 1)

| Variable | Skewness | Recommendation |
|----------|----------|----------------|
| **wbc_max** | 10.143 | Log-transform |
| **creatinine_max** | 6.996 | Log-transform |
| **inr_max** | 6.084 | Log-transform |
| **lactate_max_24h** | 2.974 | QuantileTransform |
| **platelets_max** | 1.838 | Log-transform |
| **urine_output_24h** | 1.785 | QuantileTransform |
| **platelets_min** | 1.625 | Log-transform |
| **aniongap_max** | 1.627 | Log-transform |

### Recommendations

- **Log-transform**: Variables with extreme positive skewness and all positive values (WBC, creatinine, INR, platelets)
- **QuantileTransform**: Variables that may have zero or negative values (lactate, urine output)
- **Impact**: Transformation will improve model performance for linear models (Logistic Regression) and help normalize distributions

---

## 4. Continuous Variables vs Mortality Association Trends

### Logistic Regression Associations

All key continuous variables showed **non-linear associations** with mortality:

- **lactate_max_24h**: Strong positive association (higher lactate → higher mortality)
- **sofa_total**: Strong positive association (higher SOFA → higher mortality)
- **age**: Moderate positive association (older → higher mortality)
- **urine_output_24h**: Strong negative association (lower output → higher mortality)

### Clinical Significance

- Non-linear relationships suggest that simple linear models may miss important threshold effects
- Consider creating categorical bins or using splines for better capture of non-linear associations
- Logistic regression plots show clear dose-response relationships for most variables

---

## 5. Multivariate Interaction Analysis

### Key Interaction Findings

#### High-Risk Combinations

1. **SOFA + Lactate**:
   - Top quartile both: **50.64% mortality** vs overall 24.96%
   - **2.03x higher risk** when both elevated

2. **Age + SOFA**:
   - Top quartile both: **49.88% mortality** vs overall 23.06%
   - **2.16x higher risk** in elderly with high SOFA

3. **Creatinine + Urine Output**:
   - High creatinine + Low urine output: Classic AKI pattern
   - Requires careful interpretation due to physiological relationship

### Clinical Interpretation

- **Synergistic effects**: Combinations of risk factors multiply mortality risk
- **Boundary effects**: Patients at the extremes (high-high or low-low) show distinct patterns
- **Recommendation**: Consider interaction terms or tree-based models that naturally capture interactions

---

## Recommendations for Modeling

### Feature Engineering

1. **Transformations**:
   - Log-transform: WBC, creatinine, INR, platelets
   - QuantileTransform: lactate, urine output

2. **Missing Value Handling**:
   - Use advanced imputation (KNNImputer or MissForest)
   - Consider missingness indicators for variables with >20% missing

3. **Feature Creation**:
   - Interaction terms: SOFA × Lactate, Age × SOFA
   - Risk stratification bins based on quartiles or clinical thresholds

### Model Selection

1. **Tree-based models** (Random Forest, XGBoost) may capture:
   - Non-linear associations
   - Variable interactions
   - Missing value patterns

2. **Linear models** (Logistic Regression) require:
   - Variable transformation
   - Interaction terms
   - Careful handling of missingness

### Priority Variables for Feature Selection

**Top predictors** (all p<0.001):
1. SOFA total
2. Lactate max (24h)
3. Age
4. Urine output (24h)
5. Creatinine max
6. Heart rate mean
7. Mean blood pressure
8. WBC max
9. Platelets min

---

## Limitations

1. **Missing data**: Systematic missingness may introduce bias
2. **Temporal relationships**: Analysis is cross-sectional; temporal patterns not explored
3. **Clinical context**: Some associations may reflect treatment rather than disease severity

---

## Generated Output Files

1. `advanced_eda_survival_comparison.png` - Box plots comparing survived vs died groups
2. `advanced_eda_survival_statistics.csv` - Statistical test results
3. `advanced_eda_missing_pattern.png` - Missing value pattern matrix
4. `advanced_eda_missing_correlation.png` - Missing value correlations
5. `advanced_eda_skewness_analysis.png` - Skewness distribution across variables
6. `advanced_eda_normality_analysis.csv` - Detailed normality test results
7. `advanced_eda_logistic_association.png` - Logistic regression associations
8. `advanced_eda_multivariate_interaction.png` - Multivariate interaction scatter plots

---

**Report Generated**: Advanced EDA Extension Analysis  
**Date**: Analysis completed  
**Analyst**: Clinical Data Science Team

