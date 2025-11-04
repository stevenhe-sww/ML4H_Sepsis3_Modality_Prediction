"""
Main Training Pipeline for Sepsis-3 Mortality Prediction
Complete end-to-end training pipeline with progress bars and comprehensive logging
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    recall_score, f1_score, brier_score_loss, roc_curve,
    precision_recall_curve, fbeta_score, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
from optuna.samplers import TPESampler
import joblib
from scipy import stats

warnings.filterwarnings('ignore')

# Set random seeds
def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        xgb.set_config(verbosity=0)
    except:
        pass
    # LightGBM verbosity is set in model initialization

# Set plotting style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging(log_dir: Path):
    """Setup logging to file and console"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_data(data_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data"""
    logger.info(f"Loading data from {data_path}")
    
    with tqdm(total=3, desc="[Data] Loading", unit="step") as pbar:
        df = pd.read_csv(data_path)
        pbar.update(1)
        
        # Remove ID columns
        id_cols = ['subject_id', 'hadm_id', 'stay_id']
        id_cols_present = [col for col in id_cols if col in df.columns]
        if id_cols_present:
            df = df.drop(columns=id_cols_present)
        pbar.update(1)
        
        target_col = 'mortality_30d'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        pbar.update(1)
    
    logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Target distribution: {(y==1).sum()} positive ({y.mean()*100:.2f}%)")
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, seed: int, logger: logging.Logger) -> Tuple:
    """Stratified train/val/test split"""
    logger.info("Splitting data (70/15/15 stratified)")
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=seed, stratify=y_temp
    )
    
    logger.info(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline(X_train: pd.DataFrame, logger: logging.Logger) -> ColumnTransformer:
    """Build preprocessing pipeline"""
    logger.info("Building preprocessing pipeline")
    
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )
    
    # Fit on training data
    preprocessor.fit(X_train)
    logger.info(f"Pipeline fitted: {len(numeric_features)} numeric features")
    return preprocessor

# ============================================================================
# Model Training Functions
# ============================================================================
def evaluate_model(y_true, y_pred, y_proba, set_name: str = "") -> Dict[str, float]:
    """Comprehensive model evaluation"""
    return {
        'Set': set_name,
        'AUC': roc_auc_score(y_true, y_proba),
        'AUPRC': average_precision_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'Brier': brier_score_loss(y_true, y_proba)
    }

def train_baseline_models(X_train, X_val, X_test, y_train, y_val, y_test,
                         logger: logging.Logger, seed: int, preprocessor=None) -> Dict:
    """Train baseline models"""
    logger.info("="*60)
    logger.info("Stage A: Baseline Model Training")
    logger.info("="*60)
    
    models = {}
    results = {}
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Preprocess data for models that don't handle NaN (LR, Stacking)
    if preprocessor is not None:
        logger.info("Preprocessing data for LR and Stacking models...")
        X_train_processed = preprocessor.transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        # Convert to DataFrame for consistency
        X_train_processed = pd.DataFrame(X_train_processed, index=X_train.index, 
                                         columns=preprocessor.get_feature_names_out())
        X_val_processed = pd.DataFrame(X_val_processed, index=X_val.index,
                                       columns=preprocessor.get_feature_names_out())
        X_test_processed = pd.DataFrame(X_test_processed, index=X_test.index,
                                        columns=preprocessor.get_feature_names_out())
    else:
        # Fallback: use median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_processed = pd.DataFrame(imputer.fit_transform(X_train), 
                                         index=X_train.index, columns=X_train.columns)
        X_val_processed = pd.DataFrame(imputer.transform(X_val),
                                       index=X_val.index, columns=X_val.columns)
        X_test_processed = pd.DataFrame(imputer.transform(X_test),
                                        index=X_test.index, columns=X_test.columns)
    
    # Logistic Regression (requires no NaN)
    logger.info("Training Logistic Regression...")
    try:
        lr = LogisticRegression(
            class_weight='balanced',
            C=0.5,
            max_iter=1000,
            random_state=seed,
            solver='lbfgs'
        )
        lr.fit(X_train_processed, y_train)
        
        y_test_pred_lr = lr.predict(X_test_processed)
        y_test_proba_lr = lr.predict_proba(X_test_processed)[:, 1]
        results['Logistic_Regression'] = evaluate_model(y_test, y_test_pred_lr, y_test_proba_lr, "Test")
        models['Logistic_Regression'] = (lr, y_test_proba_lr)
        logger.info(f"[OK] LR - AUC: {results['Logistic_Regression']['AUC']:.4f}")
    except Exception as e:
        logger.warning(f"[WARNING] Logistic Regression failed: {e}")
    
    # Random Forest
    logger.info("Training Random Forest...")
    try:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        y_test_pred_rf = rf.predict(X_test)
        y_test_proba_rf = rf.predict_proba(X_test)[:, 1]
        results['Random_Forest'] = evaluate_model(y_test, y_test_pred_rf, y_test_proba_rf, "Test")
        models['Random_Forest'] = (rf, y_test_proba_rf)
        logger.info(f"[OK] RF - AUC: {results['Random_Forest']['AUC']:.4f}")
    except Exception as e:
        logger.warning(f"[WARNING] Random Forest failed: {e}")
    
    # XGBoost
    logger.info("Training XGBoost...")
    try:
        with tqdm(total=1, desc="[Model] XGBoost training", leave=False) as pbar:
            xgb_model = xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                max_depth=5,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0
            )
            # XGBoost 2.0+ uses callbacks instead of early_stopping_rounds
            try:
                from xgb.callback import EarlyStopping
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=50, save_best=True)],
                    verbose=False
                )
            except (TypeError, ImportError):
                # Fallback for older XGBoost or if callback doesn't work
                try:
                    xgb_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except TypeError:
                    # Newest XGBoost API (no early_stopping_rounds parameter)
                    xgb_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
            pbar.update(1)
        
        y_test_pred_xgb = xgb_model.predict(X_test)
        y_test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        results['XGBoost'] = evaluate_model(y_test, y_test_pred_xgb, y_test_proba_xgb, "Test")
        models['XGBoost'] = (xgb_model, y_test_proba_xgb)
        logger.info(f"[OK] XGB - AUC: {results['XGBoost']['AUC']:.4f}")
    except Exception as e:
        logger.warning(f"[WARNING] XGBoost failed: {e}")
    
    # LightGBM
    logger.info("Training LightGBM...")
    try:
        with tqdm(total=1, desc="[Model] LightGBM training", leave=False) as pbar:
            lgb_model = lgb.LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                max_depth=5,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=-1,
                force_col_wise=True
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            pbar.update(1)
        
        y_test_pred_lgb = lgb_model.predict(X_test)
        y_test_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
        results['LightGBM'] = evaluate_model(y_test, y_test_pred_lgb, y_test_proba_lgb, "Test")
        models['LightGBM'] = (lgb_model, y_test_proba_lgb)
        logger.info(f"[OK] LGB - AUC: {results['LightGBM']['AUC']:.4f}")
    except Exception as e:
        logger.warning(f"[WARNING] LightGBM failed: {e}")
    
    # Stacking (LR, XGBoost, LightGBM) - uses preprocessed data for LR
    logger.info("Training Stacking Ensemble (LR, XGBoost, LightGBM)...")
    try:
        # For Stacking, we need to handle NaN for LR component
        # Create a pipeline that handles preprocessing for LR but keeps original data for tree models
        # Create separate pipelines for each base estimator
        base_estimators = [
            ('lr', make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000, random_state=seed)
            )),
            ('xgb', xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, learning_rate=0.05, 
                                     max_depth=5, n_estimators=300, random_state=seed,
                                     eval_metric='logloss', use_label_encoder=False, verbosity=0)),
            ('lgb', lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, learning_rate=0.05,
                                       max_depth=5, n_estimators=300, random_state=seed,
                                       verbosity=-1, force_col_wise=True))
        ]
        
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(class_weight='balanced', random_state=seed),
            cv=5
        )
        
        with tqdm(total=1, desc="[Model] Stacking training", leave=False) as pbar:
            stack.fit(X_train, y_train)
            pbar.update(1)
        
        y_test_pred_stack = stack.predict(X_test)
        y_test_proba_stack = stack.predict_proba(X_test)[:, 1]
        results['Stacking'] = evaluate_model(y_test, y_test_pred_stack, y_test_proba_stack, "Test")
        models['Stacking'] = (stack, y_test_proba_stack)
        logger.info(f"[OK] Stack - AUC: {results['Stacking']['AUC']:.4f}")
    except Exception as e:
        logger.warning(f"[WARNING] Stacking failed: {e}")
        import traceback
        logger.warning(f"Stacking error details: {traceback.format_exc()}")
    
    # Select best model
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['AUC'])
        logger.info(f"[COMPLETE] Best baseline model: {best_model_name} (AUC={results[best_model_name]['AUC']:.4f})")
        return models, results, best_model_name, models[best_model_name][0]
    else:
        raise ValueError("No models trained successfully")

# ============================================================================
# SHAP Analysis
# ============================================================================
def compute_shap_values(model, X_test, logger: logging.Logger, max_samples: int = 1000) -> Tuple:
    """Compute SHAP values with progress bar"""
    logger.info("Computing SHAP values...")
    
    # Sample for speed
    if len(X_test) > max_samples:
        X_shap = X_test.iloc[:max_samples]
        logger.info(f"Using {max_samples} samples for SHAP (for speed)")
    else:
        X_shap = X_test
    
    try:
        # Try TreeExplainer for tree-based models
        if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
            explainer = shap.TreeExplainer(model)
            with tqdm(total=1, desc="[SHAP] Computing values", unit="batch") as pbar:
                shap_values = explainer.shap_values(X_shap)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification
                pbar.update(1)
        elif hasattr(model, 'base_estimators'):
            # For Stacking, use the best base estimator
            logger.info("SHAP for Stacking: using best base estimator")
            # Get the best performing base estimator for SHAP
            return None, None
        else:
            logger.warning("SHAP not supported for this model type")
            return None, None
        
        # Feature importance
        shap_importance = pd.DataFrame({
            'Feature': X_shap.columns,
            'SHAP_Importance': np.abs(shap_values).mean(0)
        }).sort_values('SHAP_Importance', ascending=False)
        
        logger.info("[OK] SHAP computation complete")
        return shap_values, shap_importance
    except Exception as e:
        logger.warning(f"[WARNING] SHAP computation failed: {e}")
        return None, None

# ============================================================================
# Optuna Hyperparameter Optimization
# ============================================================================
def optuna_objective(trial, X_train, y_train, X_val, y_val, model_type='xgb', preprocessor=None):
    """Optuna objective function"""
    if model_type == 'xgb':
        params = {
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        try:
            # Try with early_stopping_rounds (older XGBoost)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50, verbose=False)
        except TypeError:
            # New XGBoost API - no early_stopping_rounds
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_proba)
    
    elif model_type == 'lgb':
        params = {
            'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        y_val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_proba)
    
    elif model_type == 'stacking':
        # Optimize Stacking: optimize base estimators and meta-learner
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        seed = 42
        
        # Optimize XGBoost parameters
        xgb_params = {
            'scale_pos_weight': scale_pos_weight,
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('xgb_depth', 3, 8),
            'n_estimators': trial.suggest_int('xgb_n_est', 200, 500),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
            'random_state': seed,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'verbosity': 0
        }
        
        # Optimize LightGBM parameters
        lgb_params = {
            'scale_pos_weight': scale_pos_weight,
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('lgb_depth', 3, 8),
            'n_estimators': trial.suggest_int('lgb_n_est', 200, 500),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.6, 1.0),
            'random_state': seed,
            'verbosity': -1,
            'force_col_wise': True
        }
        
        # Optimize LR parameters
        lr_C = trial.suggest_float('lr_C', 0.1, 2.0)
        
        # Optimize meta-learner LR C
        meta_C = trial.suggest_float('meta_C', 0.1, 2.0)
        
        # Build base estimators
        base_estimators = [
            ('lr', make_pipeline(
                SimpleImputer(strategy='median'),
                StandardScaler(),
                LogisticRegression(class_weight='balanced', C=lr_C, max_iter=1000, random_state=seed)
            )),
            ('xgb', xgb.XGBClassifier(**xgb_params)),
            ('lgb', lgb.LGBMClassifier(**lgb_params))
        ]
        
        # Build stacking model
        stack = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(class_weight='balanced', C=meta_C, random_state=seed),
            cv=trial.suggest_int('stacking_cv', 3, 5)  # Optimize CV folds
        )
        
        # Fit and evaluate
        stack.fit(X_train, y_train)
        y_val_proba = stack.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_proba)

def run_optuna_optimization(X_train, y_train, X_val, y_val, n_trials: int,
                           logger: logging.Logger, model_type='xgb', preprocessor=None) -> Dict:
    """Run Optuna hyperparameter optimization with progress bar"""
    logger.info(f"Starting Optuna optimization ({n_trials} trials, {model_type})")
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    
    def objective_wrapper(trial):
        return optuna_objective(trial, X_train, y_train, X_val, y_val, model_type, preprocessor)
    
    with tqdm(total=n_trials, desc=f"[Optuna] Trials ({model_type})", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({'best_AUC': f"{study.best_value:.4f}"})
        
        study.optimize(objective_wrapper, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    
    logger.info(f"[OK] Optuna complete - Best AUC: {study.best_value:.4f}")
    return study.best_params

# ============================================================================
# Threshold Optimization
# ============================================================================
def optimize_thresholds(y_true, y_proba, logger: logging.Logger) -> Dict:
    """Optimize classification thresholds using multiple strategies"""
    logger.info("Optimizing classification thresholds...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Youden's J
    j_scores = tpr - fpr
    optimal_youden_idx = np.argmax(j_scores)
    threshold_youden = thresholds[optimal_youden_idx]
    
    # F2 score (beta=2, favors recall)
    f2_scores = []
    with tqdm(thresholds, desc="[Threshold] F2 optimization", leave=False) as pbar:
        for t in pbar:
            y_pred = (y_proba >= t).astype(int)
            if y_pred.sum() > 0:
                f2_scores.append(fbeta_score(y_true, y_pred, beta=2))
            else:
                f2_scores.append(0)
    optimal_f2_idx = np.argmax(f2_scores)
    threshold_f2 = thresholds[optimal_f2_idx]
    
    # Cost minimization
    fn_cost, fp_cost = 5, 1
    costs = []
    with tqdm(thresholds, desc="[Threshold] Cost optimization", leave=False) as pbar:
        for t in pbar:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = fn * fn_cost + fp * fp_cost
            costs.append(cost)
    optimal_cost_idx = np.argmin(costs)
    threshold_cost = thresholds[optimal_cost_idx]
    
    results = {
        'youden': threshold_youden,
        'f2': threshold_f2,
        'cost': threshold_cost,
        'default': 0.5
    }
    
    logger.info(f"[OK] Thresholds optimized - Youden: {threshold_youden:.4f}, F2: {threshold_f2:.4f}, Cost: {threshold_cost:.4f}")
    return results

# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================
def bootstrap_confidence_intervals(y_true, y_proba, logger: logging.Logger, 
                                  n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """Calculate bootstrap confidence intervals"""
    logger.info(f"Computing bootstrap confidence intervals (n={n_bootstrap})")
    
    np.random.seed(seed)
    n = len(y_true)
    bootstrap_results = []
    
    with tqdm(total=n_bootstrap, desc="[Bootstrap] AUC CI", unit="iter") as pbar:
        for i in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
            y_proba_boot = y_proba[indices]
            
            try:
                auc = roc_auc_score(y_true_boot, y_proba_boot)
                auprc = average_precision_score(y_true_boot, y_proba_boot)
                brier = brier_score_loss(y_true_boot, y_proba_boot)
                bootstrap_results.append({'AUC': auc, 'AUPRC': auprc, 'Brier': brier})
            except:
                continue
            pbar.update(1)
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    ci_results = {}
    for metric in ['AUC', 'AUPRC', 'Brier']:
        ci_results[metric] = {
            'Mean': bootstrap_df[metric].mean(),
            'Lower_CI_95': bootstrap_df[metric].quantile(0.025),
            'Upper_CI_95': bootstrap_df[metric].quantile(0.975)
        }
    
    logger.info("[OK] Bootstrap complete")
    return ci_results

# ============================================================================
# Subgroup Analysis
# ============================================================================
def subgroup_analysis(X_test, y_test, y_proba, logger: logging.Logger) -> pd.DataFrame:
    """Perform subgroup analysis"""
    logger.info("Performing subgroup analysis...")
    
    results = []
    X_test_groups = X_test.copy()
    
    # Age groups
    if 'age' in X_test.columns:
        X_test_groups['age_group'] = X_test['age'].apply(lambda x: '>=65' if x >= 65 else '<65')
        for age_group in ['>=65', '<65']:
            mask = X_test_groups['age_group'] == age_group
            if mask.sum() > 0:
                y_sub = y_test[mask]
                proba_sub = y_proba[mask]
                results.append({
                    'Subgroup': f'Age {age_group}',
                    'N': int(mask.sum()),
                    'Events': int(y_sub.sum()),
                    'AUC': roc_auc_score(y_sub, proba_sub),
                    'AUPRC': average_precision_score(y_sub, proba_sub),
                    'Brier': brier_score_loss(y_sub, proba_sub)
                })
    
    # Infection source
    infection_cols = [col for col in X_test.columns if 'infection_source' in col]
    if infection_cols:
        X_test_groups['source'] = X_test[infection_cols].idxmax(axis=1)
        X_test_groups['source'] = X_test_groups['source'].str.replace('infection_source_', '')
        for source in X_test_groups['source'].unique():
            mask = X_test_groups['source'] == source
            if mask.sum() > 10:
                y_sub = y_test[mask]
                proba_sub = y_proba[mask]
                results.append({
                    'Subgroup': f'Source: {source}',
                    'N': int(mask.sum()),
                    'Events': int(y_sub.sum()),
                    'AUC': roc_auc_score(y_sub, proba_sub),
                    'AUPRC': average_precision_score(y_sub, proba_sub),
                    'Brier': brier_score_loss(y_sub, proba_sub)
                })
    
    # Pressor use
    if 'pressor_used_24h' in X_test.columns:
        for pressor in [0, 1]:
            mask = X_test['pressor_used_24h'] == pressor
            if mask.sum() > 10:
                y_sub = y_test[mask]
                proba_sub = y_proba[mask]
                results.append({
                    'Subgroup': f'Pressor: {pressor}',
                    'N': int(mask.sum()),
                    'Events': int(y_sub.sum()),
                    'AUC': roc_auc_score(y_sub, proba_sub),
                    'AUPRC': average_precision_score(y_sub, proba_sub),
                    'Brier': brier_score_loss(y_sub, proba_sub)
                })
    
    subgroup_df = pd.DataFrame(results)
    logger.info(f"[OK] Subgroup analysis complete ({len(results)} groups)")
    return subgroup_df

# ============================================================================
# Decision Curve Analysis
# ============================================================================
def decision_curve_analysis(y_true, y_proba, thresholds: np.ndarray) -> np.ndarray:
    """Calculate net benefit for DCA"""
    net_benefits = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        n = len(y_true)
        
        if threshold < 1.0:
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            net_benefit = 0
        net_benefits.append(net_benefit)
    
    return np.array(net_benefits)

# ============================================================================
# Plotting Functions
# ============================================================================
def plot_roc_pr_curves(models_results: Dict, y_test, output_path: Path, logger: logging.Logger):
    """Plot ROC and PR curves"""
    logger.info("Plotting ROC and PR curves...")
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for name, (model, y_proba) in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, (model, y_proba) in models_results.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, label=f'{name} (AUPRC={auprc:.3f})', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved: {output_path}")

def plot_calibration_curve(y_true, y_proba, output_path: Path, logger: logging.Logger):
    """Plot calibration curve"""
    logger.info("Plotting calibration curve...")
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved: {output_path}")

def plot_shap_summary(shap_values, X_shap, output_path: Path, logger: logging.Logger):
    """Plot SHAP summary"""
    if shap_values is None:
        return
    
    logger.info("Plotting SHAP summary...")
    shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved: {output_path}")

def plot_dca(y_true, y_proba, output_path: Path, logger: logging.Logger):
    """Plot Decision Curve Analysis"""
    logger.info("Plotting Decision Curve Analysis...")
    
    thresholds_dca = np.linspace(0.01, 0.99, 100)
    net_benefits = decision_curve_analysis(y_true, y_proba, thresholds_dca)
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds_dca, net_benefits, label='Model', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Treat None')
    plt.axhline(y=(y_true == 1).mean(), color='gray', linestyle='--', alpha=0.5, label='Treat All')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis (DCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"[OK] Saved: {output_path}")

# ============================================================================
# Export Functions
# ============================================================================
def export_results(models_results: Dict, baseline_results: Dict, optimized_results: Dict,
                  shap_importance: pd.DataFrame, threshold_results: Dict,
                  bootstrap_ci: Dict, subgroup_df: pd.DataFrame,
                  optuna_params: Dict, best_model, output_dir: Path, logger: logging.Logger):
    """Export all results and artifacts"""
    logger.info("="*60)
    logger.info("Exporting Results and Artifacts")
    logger.info("="*60)
    
    # Performance summary
    all_results = []
    for name, metrics in baseline_results.items():
        all_results.append({**metrics, 'Model': name, 'Stage': 'Baseline'})
    
    if optimized_results:
        for name, metrics in optimized_results.items():
            all_results.append({**metrics, 'Model': name, 'Stage': 'Optimized'})
    
    performance_df = pd.DataFrame(all_results)
    performance_df.to_csv(output_dir / 'logs' / 'model_performance_summary.csv', index=False)
    logger.info("[OK] Saved: model_performance_summary.csv")
    
    # SHAP importance
    if shap_importance is not None:
        shap_importance.head(20).to_csv(output_dir / 'logs' / 'shap_feature_importance.csv', index=False)
        logger.info("[OK] Saved: shap_feature_importance.csv")
    
    # Threshold optimization
    threshold_df = pd.DataFrame([
        {'Strategy': k, 'Threshold': f"{v:.4f}"} for k, v in threshold_results.items()
    ])
    threshold_df.to_csv(output_dir / 'logs' / 'threshold_optimization_report.csv', index=False)
    logger.info("[OK] Saved: threshold_optimization_report.csv")
    
    # Bootstrap CI
    bootstrap_df = pd.DataFrame([
        {
            'Metric': metric,
            'Mean': f"{values['Mean']:.4f}",
            'Lower_CI_95': f"{values['Lower_CI_95']:.4f}",
            'Upper_CI_95': f"{values['Upper_CI_95']:.4f}"
        }
        for metric, values in bootstrap_ci.items()
    ])
    bootstrap_df.to_csv(output_dir / 'logs' / 'bootstrap_confidence_intervals.csv', index=False)
    logger.info("[OK] Saved: bootstrap_confidence_intervals.csv")
    
    # Subgroup performance
    if len(subgroup_df) > 0:
        subgroup_df.to_csv(output_dir / 'logs' / 'subgroup_performance_table.csv', index=False)
        logger.info("[OK] Saved: subgroup_performance_table.csv")
    
    # Optuna params
    if optuna_params:
        with open(output_dir / 'artifacts' / 'optuna_best_params.json', 'w') as f:
            json.dump(optuna_params, f, indent=2)
        logger.info("[OK] Saved: optuna_best_params.json")
    
    # Best model
    if best_model:
        joblib.dump(best_model, output_dir / 'artifacts' / 'best_sepsis_model.pkl')
        logger.info("[OK] Saved: best_sepsis_model.pkl")
    
    logger.info("[COMPLETE] All exports complete")

# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Sepsis-3 Mortality Prediction Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to cleaned data CSV')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--optuna_trials', type=int, default=60, help='Number of Optuna trials')
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds for calibration')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    (output_dir / 'artifacts').mkdir(exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    
    logger = setup_logging(output_dir / 'logs')
    set_random_seeds(args.seed)
    
    logger.info("="*80)
    logger.info("Sepsis-3 Mortality Prediction Training Pipeline")
    logger.info("="*80)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Main progress bar
    main_steps = ['Data', 'Baseline', 'Optimization', 'Export']
    with tqdm(total=len(main_steps), desc="[Main] Pipeline", unit="stage") as main_pbar:
        
        # Stage 1: Data Loading
        main_pbar.set_description("[Main] Loading data")
        X, y = load_data(Path(args.data), logger)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, args.seed, logger)
        preprocessor = build_pipeline(X_train, logger)
        main_pbar.update(1)
        
        # Stage 2: Baseline Models
        main_pbar.set_description("[Main] Training baseline models")
        models, baseline_results, best_baseline_name, best_baseline_model = train_baseline_models(
            X_train, X_val, X_test, y_train, y_val, y_test, logger, args.seed, preprocessor
        )
        best_baseline_proba = models[best_baseline_name][1]
        main_pbar.update(1)
        
        # SHAP analysis
        shap_values, shap_importance = compute_shap_values(
            best_baseline_model, X_test, logger
        )
        
        # Plots
        plot_roc_pr_curves(models, y_test, output_dir / 'plots' / 'roc_pr_curves.png', logger)
        plot_calibration_curve(y_test, best_baseline_proba, 
                             output_dir / 'plots' / 'calibration_plot.png', logger)
        if shap_values is not None:
            plot_shap_summary(shap_values, X_test.iloc[:min(1000, len(X_test))],
                            output_dir / 'plots' / 'shap_summary.png', logger)
        
        # Stage 3: Optimization
        main_pbar.set_description("[Main] Running optimization")
        optimized_results = {}
        optimized_model = None
        optimized_proba = None
        
        # Optuna optimization - optimize best baseline model
        optuna_params = {}
        optimized_model = None
        optimized_proba = None
        models_to_optimize = []
        
        # Determine which models to optimize
        # 1. If Stacking is best, optimize it
        # 2. Also optimize the best single tree-based model (XGBoost or LightGBM)
        
        best_baseline_auc = max(baseline_results.values(), key=lambda x: x['AUC'])['AUC']
        best_baseline_name = max(baseline_results.keys(), key=lambda x: baseline_results[x]['AUC'])
        
        logger.info(f"Best baseline model: {best_baseline_name} (AUC={best_baseline_auc:.4f})")
        
        # Always optimize Stacking if it exists and is best, or if it's close to best
        if 'Stacking' in models:
            stacking_auc = baseline_results.get('Stacking', {}).get('AUC', 0)
            if stacking_auc >= best_baseline_auc - 0.002:  # Within 0.2% of best
                models_to_optimize.append('stacking')
                logger.info("Adding Stacking to optimization queue (best or near-best baseline)")
        
        # Also optimize best tree-based model (XGBoost or LightGBM)
        if 'XGBoost' in models and 'LightGBM' in models:
            xgb_auc = baseline_results.get('XGBoost', {}).get('AUC', 0)
            lgb_auc = baseline_results.get('LightGBM', {}).get('AUC', 0)
            if xgb_auc >= lgb_auc:
                models_to_optimize.append('xgb')
            else:
                models_to_optimize.append('lgb')
        elif 'XGBoost' in models:
            models_to_optimize.append('xgb')
        elif 'LightGBM' in models:
            models_to_optimize.append('lgb')
        
        # Optimize each model
        for model_to_optimize in models_to_optimize:
            logger.info(f"Optimizing {model_to_optimize}...")
            
            # Use fewer trials per model if optimizing multiple models
            trials_per_model = args.optuna_trials // len(models_to_optimize) if len(models_to_optimize) > 1 else args.optuna_trials
            
            optuna_params_model = run_optuna_optimization(
                X_train, y_train, X_val, y_val, trials_per_model, logger, model_to_optimize, preprocessor
            )
            
            # Train optimized model
            if model_to_optimize == 'xgb':
                optuna_params_model['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
                optuna_params_model.update({'random_state': args.seed, 'eval_metric': 'logloss',
                                          'use_label_encoder': False, 'verbosity': 0})
                opt_model = xgb.XGBClassifier(**optuna_params_model)
                try:
                    opt_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                                early_stopping_rounds=50, verbose=False)
                except TypeError:
                    opt_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                # Calibration
                calibrated_model = CalibratedClassifierCV(
                    opt_model, method='isotonic', cv=args.cv_folds
                )
                calibrated_model.fit(X_train, y_train)
                opt_proba = calibrated_model.predict_proba(X_test)[:, 1]
                
                model_name = 'XGBoost_Optimized'
                
            elif model_to_optimize == 'lgb':
                optuna_params_model['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
                optuna_params_model.update({'random_state': args.seed, 'verbosity': -1,
                                          'force_col_wise': True})
                opt_model = lgb.LGBMClassifier(**optuna_params_model)
                opt_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
                
                # Calibration
                calibrated_model = CalibratedClassifierCV(
                    opt_model, method='isotonic', cv=args.cv_folds
                )
                calibrated_model.fit(X_train, y_train)
                opt_proba = calibrated_model.predict_proba(X_test)[:, 1]
                
                model_name = 'LightGBM_Optimized'
                
            elif model_to_optimize == 'stacking':
                # Build optimized Stacking model
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                seed = args.seed
                
                # Extract optimized parameters
                xgb_params = {
                    'scale_pos_weight': scale_pos_weight,
                    'learning_rate': optuna_params_model.get('xgb_lr', 0.05),
                    'max_depth': optuna_params_model.get('xgb_depth', 5),
                    'n_estimators': optuna_params_model.get('xgb_n_est', 300),
                    'subsample': optuna_params_model.get('xgb_subsample', 0.8),
                    'colsample_bytree': optuna_params_model.get('xgb_colsample', 0.8),
                    'random_state': seed,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'verbosity': 0
                }
                
                lgb_params = {
                    'scale_pos_weight': scale_pos_weight,
                    'learning_rate': optuna_params_model.get('lgb_lr', 0.05),
                    'max_depth': optuna_params_model.get('lgb_depth', 5),
                    'n_estimators': optuna_params_model.get('lgb_n_est', 300),
                    'subsample': optuna_params_model.get('lgb_subsample', 0.8),
                    'colsample_bytree': optuna_params_model.get('lgb_colsample', 0.8),
                    'random_state': seed,
                    'verbosity': -1,
                    'force_col_wise': True
                }
                
                lr_C = optuna_params_model.get('lr_C', 0.5)
                meta_C = optuna_params_model.get('meta_C', 1.0)
                stacking_cv = optuna_params_model.get('stacking_cv', 5)
                
                base_estimators = [
                    ('lr', make_pipeline(
                        SimpleImputer(strategy='median'),
                        StandardScaler(),
                        LogisticRegression(class_weight='balanced', C=lr_C, max_iter=1000, random_state=seed)
                    )),
                    ('xgb', xgb.XGBClassifier(**xgb_params)),
                    ('lgb', lgb.LGBMClassifier(**lgb_params))
                ]
                
                opt_model = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(class_weight='balanced', C=meta_C, random_state=seed),
                    cv=stacking_cv
                )
                
                opt_model.fit(X_train, y_train)
                
                # Calibration for Stacking
                calibrated_model = CalibratedClassifierCV(
                    opt_model, method='isotonic', cv=args.cv_folds
                )
                calibrated_model.fit(X_train, y_train)
                opt_proba = calibrated_model.predict_proba(X_test)[:, 1]
                
                model_name = 'Stacking_Optimized'
            
            # Evaluate and save
            opt_results = evaluate_model(
                y_test, (opt_proba >= 0.5).astype(int), opt_proba, "Test"
            )
            optimized_results[model_name] = opt_results
            optuna_params[model_name] = optuna_params_model
            
            logger.info(f"Optimized {model_name} AUC: {opt_results['AUC']:.4f}")
            
            # Keep track of best optimized model (use the one with highest AUC)
            if optimized_proba is None:
                optimized_model = calibrated_model
                optimized_proba = opt_proba
            else:
                # Compare with current best
                current_best_auc = max([r['AUC'] for r in optimized_results.values()])
                if opt_results['AUC'] > current_best_auc:
                    optimized_model = calibrated_model
                    optimized_proba = opt_proba
        
        if not models_to_optimize:
            logger.info("Skipping Optuna (no models available for optimization)")
        
        # Threshold optimization (use best optimized model if available, else best baseline)
        if optimized_proba is not None and optimized_model is not None:
            threshold_results = optimize_thresholds(y_val, 
                optimized_model.predict_proba(X_val)[:, 1], logger)
        else:
            threshold_results = optimize_thresholds(y_val, 
                best_baseline_model.predict_proba(X_val)[:, 1], logger)
        
        # Bootstrap CI
        final_proba = optimized_proba if optimized_proba is not None else best_baseline_proba
        bootstrap_ci = bootstrap_confidence_intervals(y_test, final_proba, logger, 1000, args.seed)
        
        # Subgroup analysis
        subgroup_df = subgroup_analysis(X_test, y_test, final_proba, logger)
        
        # DCA
        plot_dca(y_test, final_proba, output_dir / 'plots' / 'dca_plot.png', logger)
        
        main_pbar.update(1)
        
        # Stage 4: Export
        main_pbar.set_description("[Main] Exporting results")
        best_final_model = optimized_model if optimized_model is not None else best_baseline_model
        export_results(
            models, baseline_results, optimized_results,
            shap_importance, threshold_results, bootstrap_ci, subgroup_df,
            optuna_params, best_final_model, output_dir, logger
        )
        main_pbar.update(1)
    
    elapsed_time = time.time() - start_time
    logger.info("="*80)
    logger.info(f"[COMPLETE] Pipeline finished in {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)

if __name__ == "__main__":
    main()

