import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import os
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

class MLDataProcessor:
    """
    Data processing pipeline for ML trading models.
    Handles data collection, preprocessing, and preparation for training.
    """
    
    def __init__(self, data_dir: str = "ml/data"):
        self.data_dir = data_dir
        self.data_cache = {}
        
        # Load API keys
        load_dotenv()
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("APCA_API_SECRET_KEY")
        
        # Initialize Alpaca clients (both stock and crypto)
        self.stock_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        self.crypto_client = CryptoHistoricalDataClient(self.api_key, self.api_secret)
        
        # Create data directory
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a crypto pair"""
        # Crypto symbols have format: BTC/USD, ETH/USD, etc.
        return '/' in symbol
    
    def collect_historical_data(self, symbols: List[str], 
                              start_date: datetime,
                              end_date: datetime = None,
                              timeframe: TimeFrame = TimeFrame.Minute,
                              save_to_file: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection
            end_date: End date (default: now)
            timeframe: Data timeframe
            save_to_file: Whether to save data to files
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        if end_date is None:
            end_date = datetime.now()
        
        all_data = {}
        
        for symbol in symbols:
            print(f"Collecting data for {symbol}...")
            
            try:
                # Check if data already exists
                cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                if cache_key in self.data_cache:
                    all_data[symbol] = self.data_cache[cache_key]
                    continue
                
                # Determine if crypto or stock and use appropriate client
                is_crypto = self.is_crypto_symbol(symbol)
                
                if is_crypto:
                    # Fetch crypto data
                    request_params = CryptoBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=timeframe,
                        start=start_date,
                        end=end_date
                    )
                    bars = self.crypto_client.get_crypto_bars(request_params).df
                else:
                    # Fetch stock data
                    request_params = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=timeframe,
                        start=start_date,
                        end=end_date
                    )
                    bars = self.stock_client.get_stock_bars(request_params).df
                
                if not bars.empty:
                    # Reset index to make timestamp a column
                    bars = bars.reset_index()
                    
                    # Drop symbol column if it exists (from multi-index)
                    if 'symbol' in bars.columns:
                        bars = bars.drop('symbol', axis=1)
                    
                    # Ensure we have required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in bars.columns for col in required_cols):
                        all_data[symbol] = bars
                        self.data_cache[cache_key] = bars
                        
                        # Save to file if requested
                        if save_to_file:
                            filename = f"{self.data_dir}/{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                            bars.to_csv(filename, index=False)
                            print(f"Saved data for {symbol} to {filename}")
                    else:
                        print(f"Missing required columns for {symbol}: {bars.columns.tolist()}")
                else:
                    print(f"No data returned for {symbol}")
                    
            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")
                continue
        
        return all_data
    
    def load_cached_data(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Load cached data from file.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame if found, None otherwise
        """
        filename = f"{self.data_dir}/{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                print(f"Error loading cached data for {symbol}: {e}")
        
        return None
    
    def prepare_training_data(self, data_dict: Dict[str, pd.DataFrame],
                            min_data_points: int = 500,
                            target_symbols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for ML training by combining multiple symbols.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            min_data_points: Minimum number of data points required
            target_symbols: Specific symbols to use (None = use all)
            
        Returns:
            Tuple of (features, targets, feature_names)
        """
        if target_symbols is None:
            target_symbols = list(data_dict.keys())
        
        # Filter symbols with sufficient data
        valid_symbols = []
        for symbol in target_symbols:
            if symbol in data_dict and len(data_dict[symbol]) >= min_data_points:
                valid_symbols.append(symbol)
        
        if not valid_symbols:
            print("No symbols with sufficient data found")
            return np.array([]), np.array([]), []
        
        print(f"Preparing training data for {len(valid_symbols)} symbols...")
        
        # Combine data from all symbols
        combined_features = []
        combined_targets = []
        feature_names = None
        
        for symbol in valid_symbols:
            df = data_dict[symbol].copy()
            
            # Add symbol identifier as feature
            df['symbol'] = symbol
            
            # Create features using feature engineering
            from .feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            
            # Create features
            features_df = fe.create_all_features(df)
            
            if features_df.empty:
                continue
            
            # Create targets
            features_df = fe.create_target_variables(features_df, target_periods=[1])
            
            # Remove rows with NaN targets
            features_df = features_df.dropna(subset=['direction_1d'])
            
            if features_df.empty:
                continue
            
            # Prepare features and targets
            feature_cols = [col for col in features_df.columns 
                           if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'symbol']
                           and not col.startswith('future_') and not col.startswith('direction_')]
            
            if feature_names is None:
                feature_names = feature_cols
            
            # Ensure consistent feature columns
            if len(feature_cols) == len(feature_names):
                X_symbol = features_df[feature_cols].values
                y_symbol = features_df['direction_1d'].values
                
                # Remove NaN values
                mask = ~(np.isnan(X_symbol).any(axis=1) | np.isnan(y_symbol))
                X_symbol, y_symbol = X_symbol[mask], y_symbol[mask]
                
                if len(X_symbol) > 0:
                    combined_features.append(X_symbol)
                    combined_targets.append(y_symbol)
                    print(f"Added {len(X_symbol)} samples for {symbol}")
        
        if not combined_features:
            print("No valid data found for training")
            return np.array([]), np.array([]), []
        
        # Combine all data
        X = np.vstack(combined_features)
        y = np.hstack(combined_targets)
        
        print(f"Combined training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, feature_names
    
    def create_sliding_window_data(self, df: pd.DataFrame, 
                                 window_size: int = 60,
                                 step_size: int = 1) -> List[pd.DataFrame]:
        """
        Create sliding window datasets for time series ML.
        
        Args:
            df: Input DataFrame
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            List of window DataFrames
        """
        windows = []
        
        for i in range(0, len(df) - window_size + 1, step_size):
            window = df.iloc[i:i + window_size].copy()
            windows.append(window)
        
        return windows
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray, 
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset to handle class imbalance.
        
        Args:
            X: Features
            y: Targets
            method: Balancing method ('undersample', 'oversample', 'smote')
            
        Returns:
            Balanced (X, y) tuple
        """
        from collections import Counter
        
        class_counts = Counter(y)
        print(f"Original class distribution: {class_counts}")
        
        if method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
            
        elif method == 'oversample':
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_balanced, y_balanced = ros.fit_resample(X, y)
            
        elif method == 'smote':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
        else:
            print(f"Unknown balancing method: {method}")
            return X, y
        
        balanced_counts = Counter(y_balanced)
        print(f"Balanced class distribution: {balanced_counts}")
        
        return X_balanced, y_balanced
    
    def create_time_series_splits(self, X: np.ndarray, y: np.ndarray,
                                n_splits: int = 5) -> List[Tuple]:
        """
        Create time series splits for cross-validation.
        
        Args:
            X: Features
            y: Targets
            n_splits: Number of splits
            
        Returns:
            List of (train_idx, val_idx) tuples
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(X):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def save_processed_data(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str],
                          filename: str = None) -> str:
        """
        Save processed data to file.
        
        Args:
            X: Features
            y: Targets
            feature_names: List of feature names
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.data_dir}/processed_data_{timestamp}.npz"
        
        np.savez_compressed(
            filename,
            X=X,
            y=y,
            feature_names=feature_names
        )
        
        print(f"Saved processed data to {filename}")
        return filename
    
    def load_processed_data(self, filename: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load processed data from file.
        
        Args:
            filename: Path to data file
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            data = np.load(filename, allow_pickle=True)
            X = data['X']
            y = data['y']
            feature_names = data['feature_names'].tolist()
            
            print(f"Loaded processed data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return np.array([]), np.array([]), []
    
    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get summary statistics for collected data.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            
        Returns:
            DataFrame with data summary
        """
        summary_data = []
        
        for symbol, df in data_dict.items():
            if df.empty:
                continue
            
            summary_data.append({
                'symbol': symbol,
                'start_date': df['timestamp'].min(),
                'end_date': df['timestamp'].max(),
                'total_points': len(df),
                'avg_price': df['close'].mean(),
                'price_volatility': df['close'].std(),
                'avg_volume': df['volume'].mean(),
                'missing_values': df.isnull().sum().sum()
            })
        
        return pd.DataFrame(summary_data)
