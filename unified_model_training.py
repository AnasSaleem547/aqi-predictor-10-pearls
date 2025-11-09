#!/usr/bin/env python3
"""
Unified Model Training Script
============================

This script runs all three AQI prediction models (RandomForest, LightGBM, XGBoost) 
sequentially for automated daily retraining.

Author: Anas Saleem
Institution: FAST NUCES
"""

import os
import sys
import warnings
from datetime import datetime
import zoneinfo

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_current_timestamp():
    """Get current timestamp in Pakistan timezone."""
    pkt_tz = zoneinfo.ZoneInfo('Asia/Karachi')
    return datetime.now(pkt_tz).strftime('%Y-%m-%d %H:%M:%S PKT')

def run_model_training(model_name, training_function):
    """Run a single model training with error handling and logging."""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name} Model")
    print(f"⏰ Started at: {get_current_timestamp()}")
    print(f"{'='*60}")
    
    try:
        # Run the training pipeline
        result = training_function()
        
        print(f"\n✅ {model_name} training completed successfully!")
        print(f"   Result: {result}")
        return True
        
    except Exception as e:
        print(f"\n❌ {model_name} training failed!")
        print(f"   Error: {str(e)}")
        return False

def main():
    """Main function to run all model trainings sequentially."""
    print("🎯 Unified AQI Model Training Pipeline")
    print(f"🕐 Pipeline started at: {get_current_timestamp()}")
    print("="*70)
    
    # Import training functions
    try:
        from aqi_randomforest_training_pipeline import main as run_randomforest
        from aqi_lgbm_training_pipeline import main as run_lgbm
        from aqi_xgboost_training_pipeline import main as run_xgboost
    except ImportError as e:
        print(f"❌ Failed to import training pipelines: {e}")
        sys.exit(1)
    
    # Define models to train
    models = [
        ("RandomForest", run_randomforest),
        ("LightGBM", run_lgbm),
        ("XGBoost", run_xgboost)
    ]
    
    # Track results
    training_results = {}
    
    # Train each model
    for model_name, training_function in models:
        success = run_model_training(model_name, training_function)
        training_results[model_name] = success
        
        if not success:
            print(f"\n⚠️  Continuing with next model despite {model_name} failure...")
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"⏰ Completed at: {get_current_timestamp()}")
    
    successful_models = sum(training_results.values())
    total_models = len(training_results)
    
    print(f"✅ Successful: {successful_models}/{total_models} models")
    
    for model_name, success in training_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {model_name}")
    
    # Exit with appropriate code
    if successful_models == total_models:
        print("\n🎉 All models trained successfully!")
        sys.exit(0)
    elif successful_models > 0:
        print(f"\n⚠️  {total_models - successful_models} models failed, but {successful_models} succeeded")
        sys.exit(0)  # Consider this a success since some models worked
    else:
        print("\n❌ All model trainings failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()