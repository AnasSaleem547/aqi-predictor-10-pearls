#!/usr/bin/env python3
"""
AQI Random Forest Hyperparameter Optimization with Optuna
===========================================================

Optuna-based hyperparameter optimization for Random Forest AQI prediction model.
This script performs Bayesian optimization to find the best hyperparameters.

Author: Anas Saleem
Institution: FAST NUCES
"""

import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import data fetching functions from the main pipeline
from aqi_randomforest_training_pipeline import (
    fetch_data_from_hopsworks, preprocess_data, ALL_FEATURES, TARGET_VARIABLE
)

def objective(trial, X, y):
    """
    Optuna objective function for Random Forest hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target vector
        
    Returns:
        Average RMSE across time series splits
    """
    # Define hyperparameter search space
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20, step=2),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10, step=1),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': bootstrap,
        'oob_score': bootstrap,  # Only use OOB score if bootstrap is True
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
    
    rmse_scores = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Further split training data for validation
        val_size = int(len(train_index) * 0.2)
        if val_size < 10:
            val_size = min(10, len(train_index) - 1)
            
        val_index = train_index[-val_size:]
        train_index = train_index[:-val_size]
        
        X_train_final = X[train_index]
        y_train_final = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train Random Forest model
        model = RandomForestRegressor(**params)
        model.fit(X_train_scaled, y_train_final)
        
        # Validate on validation set (for early stopping equivalent)
        val_pred = model.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Report intermediate value to Optuna for pruning
        trial.report(val_rmse, fold)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Test on test set
        test_pred = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        rmse_scores.append(test_rmse)
    
    # Return average RMSE across folds
    return np.mean(rmse_scores)

def run_optimization(n_trials=50, timeout=3600):
    """
    Run Optuna hyperparameter optimization for Random Forest.
    
    Args:
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds for optimization
        
    Returns:
        Dictionary with best parameters
    """
    print("🚀 Starting Random Forest Hyperparameter Optimization with Optuna...")
    print("=" * 70)
    
    try:
        # Fetch and preprocess data
        print("📊 Fetching and preprocessing data...")
        df = fetch_data_from_hopsworks()
        
        if df is None or len(df) == 0:
            raise ValueError("❌ No data available for optimization")
        
        # Save raw data to CSV for later use by other models
        import zoneinfo
        from datetime import datetime
        pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
        current_time_pkt = datetime.now(pkt_tz)
        timestamp = current_time_pkt.strftime('%m_%d_%Y_%H%M_pkt')
        csv_filename = f"retrieved_karachi_aqi_features_randomforest_{timestamp}.csv"
        
        print(f"💾 Saving fetched data to CSV: {csv_filename}")
        df.to_csv(csv_filename, index=False)
        print(f"✅ Raw data saved to {csv_filename}")
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        if df_processed is None or len(df_processed) == 0:
            raise ValueError("❌ No valid data after preprocessing")
        
        # Prepare features and target
        X = df_processed[ALL_FEATURES].values
        y = df_processed[TARGET_VARIABLE].values
        
        print(f"📈 Dataset shape: {X.shape}")
        print(f"🎯 Target range: {y.min():.1f} - {y.max():.1f}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',  # We want to minimize RMSE
            sampler=TPESampler(seed=42),
            pruner=HyperbandPruner()
        )
        
        # Run optimization
        print(f"\n🔍 Running {n_trials} trials with {timeout}s timeout...")
        
        study.optimize(
            lambda trial: objective(trial, X, y),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Print optimization results
        print("\n✅ Optimization completed!")
        print("=" * 70)
        print(f"📊 Best trial: #{study.best_trial.number}")
        print(f"🎯 Best RMSE: {study.best_value:.4f}")
        print("\n🔧 Best hyperparameters:")
        
        best_params = study.best_params.copy()
        best_params['oob_score'] = True  # Always include OOB score
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
        for param, value in best_params.items():
            print(f"  {param:25}: {value}")
        
        # Save best parameters to JSON file
        output_file = 'best_randomforest_params.json'
        with open(output_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\n💾 Best parameters saved to: {output_file}")
        
        # Display optimization history
        print("\n📈 Optimization History:")
        print("-" * 40)
        
        trials_df = study.trials_dataframe()
        if len(trials_df) > 0:
            print(f"Total trials: {len(trials_df)}")
            print(f"Completed trials: {len(trials_df[trials_df['state'] == 'COMPLETE'])}")
            print(f"Pruned trials: {len(trials_df[trials_df['state'] == 'PRUNED'])}")
            
            # Show top 5 trials
            print("\n🏆 Top 5 Trials:")
            top_trials = trials_df.nsmallest(5, 'value')[['number', 'value', 'datetime_complete']]
            for _, trial in top_trials.iterrows():
                print(f"  Trial #{int(trial['number'])}: RMSE = {trial['value']:.4f}")
        
        return best_params
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_best_params(best_params, X, y):
    """
    Validate the best parameters using cross-validation.
    
    Args:
        best_params: Dictionary with best hyperparameters
        X: Feature matrix
        y: Target vector
        
    Returns:
        Dictionary with validation metrics
    """
    print("\n🔍 Validating best parameters with cross-validation...")
    
    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.2))
    
    all_metrics = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Further split training data for validation
        val_size = int(len(train_index) * 0.2)
        if val_size < 10:
            val_size = min(10, len(train_index) - 1)
            
        val_index = train_index[-val_size:]
        train_index = train_index[:-val_size]
        
        X_train_final = X[train_index]
        y_train_final = y[train_index]
        X_val = X[val_index]
        y_val = y[val_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with best parameters
        model = RandomForestRegressor(**best_params)
        model.fit(X_train_scaled, y_train_final)
        
        # Evaluate on test set
        test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
        
        metrics = {
            'MAE': mean_absolute_error(y_test, test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
            'R2': r2_score(y_test, test_pred),
            'MAPE': mean_absolute_percentage_error(y_test, test_pred)
        }
        
        all_metrics.append(metrics)
        
        print(f"Fold {fold + 1} RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {
        metric: np.mean([fold_metrics[metric] for fold_metrics in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    std_metrics = {
        f"{metric}_std": np.std([fold_metrics[metric] for fold_metrics in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    print("\n📊 Cross-Validation Results:")
    print("-" * 40)
    for metric, value in avg_metrics.items():
        std_value = std_metrics[f"{metric}_std"]
        print(f"{metric:10}: {value:.4f} ± {std_value:.4f}")
    
    return {**avg_metrics, **std_metrics}

if __name__ == "__main__":
    # Configuration
    N_TRIALS = 50  # Number of optimization trials
    TIMEOUT = 3600  # 1 hour timeout in seconds
    
    print("🌲 Random Forest Hyperparameter Optimization for AQI Prediction")
    print("=" * 70)
    
    # Run optimization
    best_params = run_optimization(n_trials=N_TRIALS, timeout=TIMEOUT)
    
    if best_params:
        # Fetch data for validation
        print("\n📊 Fetching data for final validation...")
        df = fetch_data_from_hopsworks()
        
        if df is not None and len(df) > 0:
            df_processed = preprocess_data(df)
            
            if df_processed is not None and len(df_processed) > 0:
                X = df_processed[ALL_FEATURES].values
                y = df_processed[TARGET_VARIABLE].values
                
                # Validate best parameters
                validation_results = validate_best_params(best_params, X, y)
                
                print("\n✅ Optimization and validation completed successfully!")
                print("🚀 You can now run the Random Forest training pipeline with optimized parameters!")
            else:
                print("\n⚠️ Could not validate parameters due to data preprocessing issues")
        else:
            print("\n⚠️ Could not validate parameters due to data fetching issues")
    else:
        print("\n❌ Optimization failed. Check the error messages above.")