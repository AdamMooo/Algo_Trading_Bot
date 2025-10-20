#!/usr/bin/env python3
"""
Optimized ML Training Pipeline for Trading Bot
Enhanced Random Forest with hyperparameter tuning and comprehensive stock coverage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_engineering import FeatureEngineer
from ml.models import MLTradingModels
from ml.data_processor import MLDataProcessor
from datetime import datetime, timedelta
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

# Import stock symbols from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_SYMBOLS as ALL_STOCKS, TRAINING_DAYS, TRAINING_FORWARD_PERIODS, TRAINING_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description='Optimized ML Training Pipeline for Trading Bot')
    parser.add_argument('--symbols', nargs='+', default=ALL_STOCKS,
                       help='List of symbols to train on (default: all stocks)')
    parser.add_argument('--days', type=int, default=TRAINING_DAYS,
                       help='Number of days of historical data')
    parser.add_argument('--forward-periods', type=int, default=TRAINING_FORWARD_PERIODS,
                       help='How many periods to look ahead for labels')
    parser.add_argument('--threshold', type=float, default=TRAINING_THRESHOLD,
                       help='Minimum price move threshold for labels')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='Enable hyperparameter optimization (slower but better performance)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2 = 20%)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("OPTIMIZED ML TRADING TRAINING PIPELINE")
    print("=" * 80)
    print(f"Training on {len(args.symbols)} stocks")
    print(f"Training period: {args.days} days")
    print(f"Forward periods: {args.forward_periods} (look ahead)")
    print(f"Threshold: {args.threshold:.4f} ({args.threshold*100:.2f}% min move)")
    print(f"Hyperparameter optimization: {'ENABLED' if args.optimize_hyperparams else 'DISABLED'}")
    print(f"Test set size: {args.test_size:.1%}")
    print("=" * 80)
    print()
    
    # Initialize components
    feature_engineer = FeatureEngineer()
    ml_models = MLTradingModels()
    data_processor = MLDataProcessor()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print()
    
    # Step 1: Collect historical data with progress tracking
    print("Step 1: Collecting historical data...")
    print(f"Fetching {args.days} days of data for {len(args.symbols)} symbols...")
    
    start_time = time.time()
    data_dict = data_processor.collect_historical_data(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        save_to_file=False
    )
    collection_time = time.time() - start_time
    
    if not data_dict:
        print("ERROR: No data collected. Check your API keys and internet connection.")
        return 1
    
    print(f"Data collection completed in {collection_time:.1f} seconds")
    print(f"Successfully collected data for {len(data_dict)}/{len(args.symbols)} symbols")
    
    # Step 2: Create features and labels with enhanced processing
    print(f"\nStep 2: Creating features and labels...")
    all_features = []
    all_labels = []
    symbol_stats = {}
    
    feature_start = time.time()
    
    for symbol, df in data_dict.items():
        if df.empty or len(df) < 100:
            print(f"  Skipping {symbol} - insufficient data ({len(df)} rows)")
            continue
        
        try:
            # Create features
            features_df = feature_engineer.create_all_features(df)
            
            if features_df.empty:
                print(f"  Skipping {symbol} - feature creation failed")
                continue
            
            # Create labels
            labeled_df = feature_engineer.create_training_labels(
                features_df, 
                forward_periods=args.forward_periods,
                threshold=args.threshold
            )
            
            if labeled_df.empty:
                print(f"  Skipping {symbol} - labeling failed")
                continue
            
            # Get feature columns
            feature_cols = feature_engineer.get_feature_columns()
            
            X = labeled_df[feature_cols].values
            y = labeled_df['target'].values
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:  # Need minimum samples
                print(f"  Skipping {symbol} - insufficient valid samples ({len(X)})")
                continue
            
            all_features.append(X)
            all_labels.append(y)
            
            # Store statistics
            symbol_stats[symbol] = {
                'samples': len(X),
                'positive_rate': np.mean(y)
            }
            
            print(f"  {symbol}: {len(X)} valid samples (positive rate: {np.mean(y):.3f})")
            
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
            continue
    
    feature_time = time.time() - feature_start
    
    if not all_features:
        print("Error: No features created.")
        return 1
    
    # Combine all data
    X_all = np.vstack(all_features)
    y_all = np.hstack(all_labels)
    
    print(f"Features created: {len(X_all):,} samples, {X_all.shape[1]} features ({feature_time:.1f}s)")
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, 
        test_size=args.test_size, 
        random_state=42, 
        stratify=y_all
    )
    
    # Step 4: Train models
    print("Training model...")
    ml_models.create_models()
    
    training_start = time.time()
    results = ml_models.train_models(X_train, y_train, X_test, y_test)
    training_time = time.time() - training_start
    
    # Optional hyperparameter optimization
    if args.optimize_hyperparams:
        print("Optimizing hyperparameters...")
        from sklearn.ensemble import RandomForestClassifier
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save optimized model
        import joblib
        joblib.dump(grid_search.best_estimator_, "ml/models/random_forest_optimized.joblib")
    
    print(f"Training complete ({training_time:.1f}s)")
    
    # Cross-validation
    print("Cross-validation...")
    cv_results = ml_models.cross_validate_models(X_all, y_all, cv=5)
    
    # Model evaluation
    if 'random_forest' in ml_models.models:
        best_model = ml_models.models['random_forest']
        y_pred = best_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
    else:
        print("Model not available for evaluation")
    
    # Feature importance
    importance = ml_models.get_feature_importance('random_forest')
    if importance is not None:
        feature_names = feature_engineer.get_feature_columns()
        importance_sorted = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        
        print("Top 10 features:")
        for i, (name, imp) in enumerate(importance_sorted[:10], 1):
            print(f"  {i:2d}. {name}: {imp:.4f}")
        
        # Save feature importance
        importance_df = pd.DataFrame(importance_sorted, columns=['feature', 'importance'])
        importance_df.to_csv('ml/models/feature_importance.csv', index=False)
    
    # Save models and metadata
    
    # Save feature names
    feature_names = feature_engineer.get_feature_columns()
    with open("ml/models/feature_names.txt", 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    # Save metadata
    symbol_stats_df = pd.DataFrame.from_dict(symbol_stats, orient='index')
    symbol_stats_df.to_csv('ml/models/symbol_statistics.csv')
    
    total_time = collection_time + feature_time + training_time
    print(f"\nTraining complete: {len(X_all):,} samples, {len(feature_names)} features ({total_time:.1f}s)")
    print(f"Symbols: {len(data_dict)}/{len(args.symbols)} successful")
    print("Model saved: ml/models/random_forest.joblib")
    
    return 0

if __name__ == "__main__":
    exit(main())
