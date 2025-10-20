import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class MLTradingModels:
    """
    Random Forest model.
    Simple, interpretable, and works well for trading signals.
    """
    
    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_scores = {}
        
        # Create model directory
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create a simple Random Forest model.
        
        Returns:
            Dictionary with the model
        """
        model = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=10,          # Prevent overfitting
            min_samples_split=20,  # Need 20 samples to split
            min_samples_leaf=10,   # Need 10 samples in leaf
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        self.models['random_forest'] = model
        return self.models
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        if not self.models:
            self.create_models()
        
        results = {}
        model = self.models['random_forest']
        
        print("Training Random Forest...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"Training Accuracy: {train_accuracy:.3f}")
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, average='binary', zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, average='binary', zero_division=0)
            
            print(f"Validation Accuracy: {val_accuracy:.3f}")
            print(f"Validation Precision: {val_precision:.3f}")
            print(f"Validation Recall: {val_recall:.3f}")
            print(f"Validation F1: {val_f1:.3f}")
            
            results['random_forest'] = {
                'train_accuracy': train_accuracy,
                'val_metrics': {
                    'accuracy': val_accuracy,
                    'precision': val_precision,
                    'recall': val_recall,
                    'f1': val_f1
                }
            }
        else:
            results['random_forest'] = {
                'train_accuracy': train_accuracy,
                'val_metrics': None
            }
        
        # Save model
        self.save_model(model, 'random_forest')
        
        # Save results
        import json
        with open(f"{self.model_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.model_scores = results
        return results
    
    def cross_validate_models(self, X: np.ndarray, y: np.ndarray, cv: int = 3) -> Dict:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with CV scores
        """
        if not self.models:
            self.create_models()
        
        model = self.models['random_forest']
        
        print(f"Running {cv}-fold cross-validation...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        cv_results = {
            'random_forest': {
                'cv_scores': scores.tolist(),
                'cv_mean': scores.mean(),
                'cv_std': scores.std()
            }
        }
        
        print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return cv_results
    
    def predict_signals(self, X: np.ndarray, model_name: str = 'random_forest',
                       return_proba: bool = False):
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of model to use
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions and optionally probabilities
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Loading from file...")
            model = self.load_model(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not available")
            self.models[model_name] = model
        
        model = self.models[model_name]
        
        predictions = model.predict(X)
        
        if return_proba:
            probabilities = model.predict_proba(X)
            return predictions, probabilities
        
        return predictions
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Optional[np.ndarray]:
        """Get feature importance from Random Forest"""
        if model_name not in self.models:
            model = self.load_model(model_name)
            if model is None:
                return None
            self.models[model_name] = model
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        
        return None
    
    def save_model(self, model: Any, model_name: str):
        """Save model to disk"""
        filepath = f"{self.model_dir}/{model_name}.joblib"
        joblib.dump(model, filepath)
        print(f"Model saved: {filepath}")
    
    def load_model(self, model_name: str) -> Optional[Any]:
        """Load model from disk"""
        filepath = f"{self.model_dir}/{model_name}.joblib"
        
        if not os.path.exists(filepath):
            print(f"Model file not found: {filepath}")
            return None
        
        try:
            model = joblib.load(filepath)
            print(f"Model loaded: {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
