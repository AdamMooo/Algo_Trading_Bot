import pandas as pd
import numpy as np
import talib
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    SIMPLIFIED feature engineering - only essential technical indicators.
    Focus on proven features that won't overfit.
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create simplified feature set with only essential indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with 15 key features
        """
        if df.empty or len(df) < 50:
            return pd.DataFrame()
            
        # Make a copy and keep only OHLCV columns (drop symbol, timestamp if present)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame()
        
        # Create features_df with only numeric columns we need
        features_df = df[required_cols].copy()
        
        try:
            # Convert to numpy arrays for TA-Lib
            open_p = features_df['open'].values
            high = features_df['high'].values
            low = features_df['low'].values
            close = features_df['close'].values
            volume = features_df['volume'].values
            
            # === 15 ESSENTIAL FEATURES ===
            
            # 1-2. Simple Moving Averages (trend)
            features_df['sma_20'] = talib.SMA(close, timeperiod=20)
            features_df['sma_50'] = talib.SMA(close, timeperiod=50)
            features_df['sma_ratio'] = features_df['sma_20'] / features_df['sma_50']
            
            # 3. RSI (momentum)
            features_df['rsi_14'] = talib.RSI(close, timeperiod=14)
            
            # 4-5. MACD (momentum)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            
            # 6-8. Bollinger Bands (volatility)
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            features_df['bb_upper'] = upper
            features_df['bb_lower'] = lower
            features_df['bb_width'] = (upper - lower) / middle
            
            # 9. ATR (volatility)
            features_df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            
            # 10. Volume ratio (volume)
            features_df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            features_df['volume_ratio'] = volume / features_df['volume_sma']
            
            # 11-12. Price momentum
            features_df['price_change_1d'] = close / np.roll(close, 1) - 1
            features_df['price_change_5d'] = close / np.roll(close, 5) - 1
            
            # 13. Distance from SMA (trend strength)
            features_df['price_vs_sma20'] = (close - features_df['sma_20']) / features_df['sma_20']
            
            # 14. High-Low range (volatility)
            features_df['hl_ratio'] = (high - low) / close
            
            # 15. Close position in range (momentum)
            features_df['close_position'] = (close - low) / (high - low + 0.0001)  # Avoid div by 0
            
            # Drop NaN rows
            features_df = features_df.dropna()
            
            # Store feature names (exclude OHLCV columns and any string columns)
            # Only keep numeric feature columns
            exclude_cols = required_cols + ['timestamp', 'trade_count', 'vwap', 'symbol']
            self.feature_names = [col for col in features_df.columns 
                                 if col not in exclude_cols 
                                 and features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            
            return features_df
            
        except Exception as e:
            print(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def create_training_labels(self, df: pd.DataFrame, forward_periods: int = 5, 
                               threshold: float = 0.002) -> pd.DataFrame:
        """
        Create binary labels for training: 1 if price goes up, 0 if down.
        
        Args:
            df: DataFrame with features
            forward_periods: How many periods to look ahead
            threshold: Minimum % move to consider (filters noise)
            
        Returns:
            DataFrame with 'target' column
        """
        if 'close' not in df.columns:
            return df
        
        # Calculate future return
        df['future_return'] = (df['close'].shift(-forward_periods) - df['close']) / df['close']
        
        # Binary classification: 1 = up, 0 = down
        # Only label strong moves above threshold
        df['target'] = 0
        df.loc[df['future_return'] > threshold, 'target'] = 1
        df.loc[df['future_return'] < -threshold, 'target'] = 0
        
        # Drop rows without future data
        df = df.dropna(subset=['future_return', 'target'])
        
        # Drop the helper column
        df = df.drop('future_return', axis=1)
        
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names"""
        return self.feature_names
