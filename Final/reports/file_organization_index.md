# File Organization Index
## Sepsis-3 Cohort Analysis Project

This document provides a comprehensive classification and organization of all files in the project directory.

---

## 📊 Data Files

### Original Data
- **sepsis3_cohort_all_features.csv**
  - Type: Raw dataset
  - Size: ~20,391 rows × 24 columns
  - Description: Original Sepsis-3 cohort data with all features
  - Status: Input file for all analyses

### Processed Data
- **sepsis3_cleaned.csv**
  - Type: Cleaned and preprocessed dataset
  - Size: ~20,391 rows × 30 columns
  - Description: Fully cleaned dataset after outlier treatment, imputation, transformation, standardization
  - Status: Ready for modeling
  - Related: Output from `step2_data_cleaning.py`

---

## 🐍 Python Scripts

### Step 1: Exploratory Data Analysis (EDA)
- **step1_eda.py**
  - Purpose: Basic exploratory data analysis
  - Outputs: 
    - eda_continuous_histograms.png
    - eda_continuous_violin.png
    - eda_categorical_frequency.png
    - eda_correlation_matrix.png
    - eda_target_distribution.png
    - eda_summary_statistics.csv
  - Functions:
    - Load data and basic statistics
    - Distribution visualization
    - Correlation analysis
    - Missing value analysis
    - Outlier detection
    - Summary statistics

### Step 1: Advanced EDA
- **step1_advanced_eda.py**
  - Purpose: Advanced statistical analysis and paper-level EDA
  - Outputs:
    - advanced_eda_survival_comparison.png
    - advanced_eda_survival_statistics.csv
    - advanced_eda_missing_pattern.png
    - advanced_eda_missing_correlation.png
    - advanced_eda_skewness_analysis.png
    - advanced_eda_normality_analysis.csv
    - advanced_eda_logistic_association.png
    - advanced_eda_multivariate_interaction.png
  - Functions:
    - Survival vs death comparison with statistical tests
    - Missing value pattern analysis
    - Normality and skewness testing
    - Logistic association trends
    - Multivariate interaction analysis

### Step 2: Data Cleaning
- **step2_data_cleaning.py**
  - Purpose: Comprehensive data cleaning and preprocessing pipeline
  - Outputs:
    - sepsis3_cleaned.csv
    - standard_scaler.pkl
    - feature_list.json
    - cleaning_log_outliers.csv
    - cleaning_log_transformations.csv
    - cleaning_log_missing_values.csv
    - cleaning_comparison_stats.csv
  - Functions:
    - Outlier treatment (Winsorize)
    - Skewness correction (log1p, QuantileTransformer)
    - Missing value imputation (Median, KNN, Indicators)
    - Multicollinearity control
    - Categorical encoding
    - Feature engineering
    - Standardization
    - Quality checks

---

## 📈 Visualization Files (PNG Images)

### Basic EDA Visualizations
1. **eda_continuous_histograms.png**
   - Content: Distribution histograms for continuous variables
   - Purpose: Visualize data distributions
   - Related: step1_eda.py

2. **eda_continuous_violin.png**
   - Content: Violin plots for key continuous variables
   - Purpose: Show distribution density and quartiles
   - Related: step1_eda.py

3. **eda_categorical_frequency.png**
   - Content: Frequency bar charts for categorical variables
   - Purpose: Show category distributions
   - Related: step1_eda.py

4. **eda_correlation_matrix.png**
   - Content: Heatmap of correlation matrix
   - Purpose: Identify variable relationships
   - Related: step1_eda.py

5. **eda_target_distribution.png**
   - Content: Bar chart of mortality distribution
   - Purpose: Show target variable balance
   - Related: step1_eda.py

### Advanced EDA Visualizations
6. **advanced_eda_survival_comparison.png**
   - Content: Box plots comparing survived vs died groups
   - Purpose: Statistical comparison with p-values
   - Related: step1_advanced_eda.py

7. **advanced_eda_missing_pattern.png**
   - Content: Missingno matrix visualization
   - Purpose: Identify missing data patterns
   - Related: step1_advanced_eda.py

8. **advanced_eda_missing_correlation.png**
   - Content: Correlation heatmap of missing patterns
   - Purpose: Find systematic missing relationships
   - Related: step1_advanced_eda.py

9. **advanced_eda_skewness_analysis.png**
   - Content: Bar chart of skewness values
   - Purpose: Identify variables needing transformation
   - Related: step1_advanced_eda.py

10. **advanced_eda_logistic_association.png**
    - Content: Logistic regression curves for key variables
    - Purpose: Show association trends with mortality
    - Related: step1_advanced_eda.py

11. **advanced_eda_multivariate_interaction.png**
    - Content: Scatter plots with mortality coloring
    - Purpose: Explore variable interactions
    - Related: step1_advanced_eda.py

---

## 📋 CSV Log and Statistics Files

### EDA Statistics
- **eda_summary_statistics.csv**
  - Content: Mean, std, median, IQR for all continuous variables
  - Purpose: Descriptive statistics summary
  - Related: step1_eda.py

- **advanced_eda_survival_statistics.csv**
  - Content: Statistical test results (p-values) for survival comparison
  - Columns: Variable, Mean_Survived, Mean_Died, Difference, P_value, Significant
  - Related: step1_advanced_eda.py

- **advanced_eda_normality_analysis.csv**
  - Content: Skewness, kurtosis, normality test results
  - Purpose: Identify variables needing transformation
  - Related: step1_advanced_eda.py

### Data Cleaning Logs
- **cleaning_log_outliers.csv**
  - Content: Outlier treatment log
  - Columns: Variable, Outliers_Before, Outliers_After, Lower_Bound, Upper_Bound
  - Related: step2_data_cleaning.py

- **cleaning_log_transformations.csv**
  - Content: Transformation log (log1p, QuantileTransformer)
  - Columns: Variable, Transform, Skewness_Before, Skewness_After
  - Related: step2_data_cleaning.py

- **cleaning_log_missing_values.csv**
  - Content: Missing value imputation log
  - Columns: Variable, Missing_Before, Missing_Percent, Method
  - Related: step2_data_cleaning.py

- **cleaning_comparison_stats.csv**
  - Content: Before/after comparison statistics
  - Columns: Variable, Mean_Before, Mean_After, Std_Before, Std_After, Skew_Before, Skew_After, Missing_Before_%, Missing_After_%
  - Related: step2_data_cleaning.py

---

## 📄 Documentation Files

- **instruciton.txt**
  - Type: Project requirements and instructions
  - Content: Step-by-step workflow for Sepsis-3 prediction modeling
  - Status: Original project specification

- **advanced_eda_summary_report.md**
  - Type: Markdown report
  - Content: Comprehensive summary of advanced EDA findings
  - Sections:
    - Survival vs death comparison
    - Missing value patterns
    - Normality analysis
    - Variable associations
    - Multivariate interactions
    - Recommendations

- **data_cleaning_report.md**
  - Type: Markdown report
  - Content: Complete data cleaning pipeline documentation
  - Sections:
    - Step-by-step cleaning process
    - Before/after comparisons
    - Feature summary
    - Recommendations for modeling

- **file_organization_index.md** (this file)
  - Type: Project organization index
  - Purpose: Classify and organize all project files

---

## 🔧 Model Artifacts

### Preprocessing Artifacts
- **standard_scaler.pkl**
  - Type: Pickle file
  - Content: Fitted StandardScaler object
  - Purpose: Used for inference/new data standardization
  - Related: step2_data_cleaning.py

- **feature_list.json**
  - Type: JSON file
  - Content: Feature mapping (original → cleaned)
  - Sections:
    - original_features
    - cleaned_features
    - new_features
    - dropped_features
  - Related: step2_data_cleaning.py

---

## 📁 Recommended Folder Organization

For better organization, consider creating the following folder structure:

```
ML4H/Final/
├── data/
│   ├── raw/
│   │   └── sepsis3_cohort_all_features.csv
│   └── processed/
│       └── sepsis3_cleaned.csv
├── scripts/
│   ├── step1_eda.py
│   ├── step1_advanced_eda.py
│   └── step2_data_cleaning.py
├── visualizations/
│   ├── eda/
│   │   ├── eda_continuous_histograms.png
│   │   ├── eda_continuous_violin.png
│   │   ├── eda_categorical_frequency.png
│   │   ├── eda_correlation_matrix.png
│   │   └── eda_target_distribution.png
│   └── advanced_eda/
│       ├── advanced_eda_survival_comparison.png
│       ├── advanced_eda_missing_pattern.png
│       ├── advanced_eda_missing_correlation.png
│       ├── advanced_eda_skewness_analysis.png
│       ├── advanced_eda_logistic_association.png
│       └── advanced_eda_multivariate_interaction.png
├── logs/
│   ├── eda_summary_statistics.csv
│   ├── advanced_eda_survival_statistics.csv
│   ├── advanced_eda_normality_analysis.csv
│   ├── cleaning_log_outliers.csv
│   ├── cleaning_log_transformations.csv
│   ├── cleaning_log_missing_values.csv
│   └── cleaning_comparison_stats.csv
├── artifacts/
│   ├── standard_scaler.pkl
│   └── feature_list.json
├── reports/
│   ├── advanced_eda_summary_report.md
│   ├── data_cleaning_report.md
│   └── file_organization_index.md
└── docs/
    └── instruciton.txt
```

---

## 🔄 File Dependency Flow

```
Raw Data
  ↓
[sepsis3_cohort_all_features.csv]
  ↓
  ├──→ step1_eda.py → Basic EDA outputs
  │      ├── Visualizations (5 PNGs)
  │      └── eda_summary_statistics.csv
  │
  └──→ step1_advanced_eda.py → Advanced EDA outputs
         ├── Visualizations (6 PNGs)
         ├── Statistics CSVs (2)
         └── advanced_eda_summary_report.md
  ↓
  └──→ step2_data_cleaning.py → Cleaning pipeline
         ├── sepsis3_cleaned.csv (processed data)
         ├── standard_scaler.pkl (artifact)
         ├── feature_list.json (metadata)
         ├── Cleaning logs (4 CSVs)
         └── data_cleaning_report.md
```

---

## 📊 File Count Summary

| Category | Count | File Types |
|----------|-------|------------|
| **Data Files** | 2 | CSV |
| **Python Scripts** | 3 | .py |
| **Visualizations** | 11 | PNG |
| **Statistics/Logs** | 8 | CSV |
| **Documentation** | 4 | MD, TXT |
| **Artifacts** | 2 | PKL, JSON |
| **Total** | 30 | - |

---

## 🎯 Quick Reference

### To view basic EDA results:
1. Open `eda_*.png` files for visualizations
2. Check `eda_summary_statistics.csv` for statistics

### To view advanced EDA results:
1. Open `advanced_eda_*.png` files
2. Check `advanced_eda_survival_statistics.csv` for test results
3. Read `advanced_eda_summary_report.md` for comprehensive findings

### To understand data cleaning:
1. Read `data_cleaning_report.md`
2. Check `cleaning_*.csv` logs for details
3. Review `cleaning_comparison_stats.csv` for before/after comparison

### To use cleaned data for modeling:
1. Load `sepsis3_cleaned.csv`
2. Use `standard_scaler.pkl` for new data standardization
3. Reference `feature_list.json` for feature mapping

---

**Last Updated**: File organization index created  
**Total Files**: 30 files across 6 categories

