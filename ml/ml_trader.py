import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime, timedelta
import os
from .feature_engineering import FeatureEngineer
from .models import MLTradingModels
from .data_processor import MLDataProcessor

class MLTrader:
    """
    SIMPLIFIED ML trading system.
    Uses basic Random Forest on 15 key features.
    """
    
    def __init__(self, model_dir: str = "ml/models", data_dir: str = "ml/data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.ml_models = MLTradingModels(model_dir)
        self.data_processor = MLDataProcessor(data_dir)
        
        # Trading parameters - AGGRESSIVE
        self.signal_threshold = 0.50  # Trade when >50% confident 
        # Lower = more trades, higher = fewer but more confident trades
        
        # Model state
        self.current_model = 'random_forest'
        self.model_loaded = False
        
        # Try to load existing model
        self._load_model()
    
    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Generate ML-based trading signal for a symbol.
        
        Args:
            symbol: Stock symbol
            df: Historical price data (OHLCV)
            
        Returns:
            Dictionary with signal, confidence, and reason
        """
        if not self.model_loaded:
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': 'No trained model available'
            }
        
        if df.empty or len(df) < 50:
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        try:
            # Create features
            features_df = self.feature_engineer.create_all_features(df)
            
            if features_df.empty:
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'reason': 'Feature engineering failed'
                }
            
            # Get feature columns (exclude OHLCV)
            feature_cols = self.feature_engineer.get_feature_columns()
            
            if not feature_cols:
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'reason': 'No feature columns found'
                }
            
            # Get latest features
            latest_features = features_df[feature_cols].iloc[-1:].values.astype(np.float64)
            
            # Make prediction
            prediction, probabilities = self.ml_models.predict_signals(
                latest_features, 
                self.current_model, 
                return_proba=True
            )
            
            # Extract confidence (probability of class 1 - up move)
            confidence = probabilities[0][1] if probabilities is not None else 0.5
            
            # Generate signal based on confidence
            if confidence >= self.signal_threshold:
                signal = 1  # BUY
                reason = f"ML predicts UP with {confidence:.1%} confidence"
            elif confidence <= (1 - self.signal_threshold):
                signal = -1  # SELL
                reason = f"ML predicts DOWN with {1-confidence:.1%} confidence"
            else:
                signal = 0  # HOLD
                reason = f"Low confidence: {confidence:.1%}"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'prediction': int(prediction[0]),
                'reason': reason,
                'model': self.current_model
            }
            
        except Exception as e:
            return {
                'signal': 0,
                'confidence': 0.0,
                'reason': f'Error: {str(e)}'
            }
    
    def _load_model(self):
        """Load the trained Random Forest model"""
        model = self.ml_models.load_model('random_forest')
        if model is not None:
            self.ml_models.models['random_forest'] = model
            self.model_loaded = True
            print(f"ML model loaded successfully")
        else:
            self.model_loaded = False
            print("Warning: No trained ML model found.")
            print("To train a model, run: python ml/training_pipeline.py")
