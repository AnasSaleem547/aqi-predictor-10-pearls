#!/usr/bin/env python3
"""
AQI LGBM Hyperparameter Optimization with Optuna
===============================================

Optuna-based hyperparameter optimization for LightGBM AQI prediction model.
This script performs Bayesian optimization to find the best hyperparameters.

Author: Anas Saleem
Institution: FAST NUCES
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import data fetching functions from the main pipeline
from aqi_lgbm_training_pipeline import (
    fetch_data_from_hopsworks, preprocess_data, ALL_FEATURES, TARGET_VARIABLE
)

def load_data_for_optuna():
    """Load and preprocess data specifically for Optuna optimization."""
    print("📥 Loading data for Optuna optimization...")
    
    # Fetch data from Hopsworks
    df = fetch_data_from_hopsworks()
    if df is None or df.empty:
        raise RuntimeError("❌ No data retrieved from Hopsworks")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare features and target
    X = df_processed[ALL_FEATURES].values
    y = df_processed[TARGET_VARIABLE].values
    
    print(f"📊 Dataset shape for optimization: {X.shape}")
    
    return X, y

def objective(trial, X, y):
    """Objective function for Optuna optimization."""
    
    # Define hyperparameter search space
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
        'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000),
        'verbose': -1,
        'random_state': 42
    }
    
    # Time series cross-validation
    n_splits = 3
    test_size = max(1, int(len(X) * 0.2))  # 20% for testing, minimum 1 sample
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    rmse_scores = []
    
    for train_index, test_index in tscv.split(X):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # Train model with early stopping
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(False)
            ]
        )
        
        # Make predictions and calculate RMSE
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_scores.append(rmse)
    
    # Return average RMSE across folds
    return np.mean(rmse_scores)

def optimize_hyperparameters(n_trials=100):
    """Main function to optimize hyperparameters using Optuna."""
    print("🚀 Starting Hyperparameter Optimization with Optuna")
    print("=" * 60)
    
    # Load data
    X, y = load_data_for_optuna()
    
    # Create Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner()
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n🏆 Optimization Results:")
    print("=" * 40)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'l2',
        'boosting_type': 'gbdt',
        'verbose': -1,
        'random_state': 42
    })
    
    # Save to JSON file
    import json
    with open('best_lgbm_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\n💾 Best parameters saved to 'best_lgbm_params.json'")
    
    # Plot optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html('optuna_optimization_history.html')
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html('optuna_param_importances.html')
        
        print("📊 Optimization visualizations saved")
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    return best_params

def test_optimized_parameters():
    """Test the optimized parameters on the full dataset."""
    print("\n🧪 Testing optimized parameters...")
    
    # Load best parameters
    try:
        import json
        with open('best_lgbm_params.json', 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        print("❌ No optimized parameters found. Run optimization first.")
        return
    
    # Load data
    X, y = load_data_for_optuna()
    
    # Final evaluation with time series split
    test_size = max(1, int(len(X) * 0.2))  # 20% for testing, minimum 1 sample
    tscv = TimeSeriesSplit(n_splits=3, test_size=test_size)
    
    all_metrics = []
    
    for train_index, test_index in tscv.split(X):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        # Train model with optimized parameters
        model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        all_metrics.append(rmse)
        
        print(f"Fold RMSE: {rmse:.4f}")
    
    print(f"\n📊 Average RMSE with optimized parameters: {np.mean(all_metrics):.4f}")
    print(f"📈 Standard deviation: {np.std(all_metrics):.4f}")

def main():
    """Main function for hyperparameter optimization."""
    
    # Number of optimization trials
    n_trials = 100
    
    # Run optimization
    best_params = optimize_hyperparameters(n_trials)
    
    # Test optimized parameters
    test_optimized_parameters()
    
    print("\n🎉 Hyperparameter optimization completed!")
    print("💡 Use the best parameters in your main training pipeline.")

if __name__ == "__main__":
    main()