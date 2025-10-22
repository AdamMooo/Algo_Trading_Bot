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
            
            # === ENHANCED FEATURE SET ===
            
            # Trend Features
            features_df['sma_20'] = talib.SMA(close, timeperiod=20)
            features_df['sma_50'] = talib.SMA(close, timeperiod=50)
            features_df['sma_200'] = talib.SMA(close, timeperiod=200)  # Long-term trend
            features_df['sma_ratio'] = features_df['sma_20'] / features_df['sma_50']
            features_df['lt_trend'] = features_df['sma_50'] / features_df['sma_200']  # Long-term trend strength
            
            # Momentum Features
            features_df['rsi_14'] = talib.RSI(close, timeperiod=14)
            features_df['rsi_2'] = talib.RSI(close, timeperiod=2)  # Short-term momentum
            
            # MACD Features
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_hist'] = macd_hist
            
            # Volatility Features
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            features_df['bb_upper'] = upper
            features_df['bb_lower'] = lower
            features_df['bb_width'] = (upper - lower) / middle
            features_df['bb_position'] = (close - lower) / (upper - lower)  # Price position within BB
            
            # Enhanced Volatility Metrics
            features_df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features_df['volatility_regime'] = features_df['atr_14'] / close
            features_df['trend_volatility'] = abs(features_df['sma_20'] - features_df['sma_50']) / features_df['atr_14']
            
            # Convert numpy arrays to pandas Series for volume analysis
            volume_series = pd.Series(volume)
            
            # Enhanced Volume Analysis
            volume_sma = talib.SMA(volume, timeperiod=20)
            features_df['volume_sma'] = volume_sma
            features_df['volume_ratio'] = volume / volume_sma
            
            # Volume Trend Analysis
            features_df['volume_trend'] = talib.SMA(volume, timeperiod=5) / talib.SMA(volume, timeperiod=20)
            features_df['volume_force'] = (close - np.roll(close, 1)) * volume  # Volume Force Index
            
            # Money Flow Analysis
            features_df['money_flow'] = ((close - low) - (high - close)) / (high - low + 0.0001) * volume
            features_df['money_flow_sma'] = talib.SMA(features_df['money_flow'].values, timeperiod=14)
            
            # Volume Price Confirmation
            features_df['price_volume_trend'] = talib.ROC(close, timeperiod=1) * talib.ROC(volume, timeperiod=1)
            features_df['volume_breakout'] = volume_series / pd.Series(volume_sma).rolling(window=20).max()
            
            # Multi-timeframe Price Action
            features_df['price_change_1d'] = close / np.roll(close, 1) - 1
            features_df['price_change_5d'] = close / np.roll(close, 5) - 1
            features_df['price_change_20d'] = close / np.roll(close, 20) - 1
            
            # Range and Support/Resistance
            features_df['high_low_range'] = (high - low) / close
            features_df['close_to_high'] = (high - close) / (high - low + 0.0001)
            features_df['close_to_low'] = (close - low) / (high - low + 0.0001)
            
            # Trend Strength and Momentum
            features_df['price_vs_sma20'] = (close - features_df['sma_20']) / features_df['sma_20']
            features_df['price_vs_sma50'] = (close - features_df['sma_50']) / features_df['sma_50']
            features_df['momentum_1d'] = close - np.roll(close, 1)
            features_df['momentum_5d'] = close - np.roll(close, 5)
            
            # Volatility and Range Patterns
            features_df['range_expansion'] = features_df['high_low_range'] / features_df['high_low_range'].rolling(window=10).mean()
            features_df['volatility_adjusted_return'] = features_df['price_change_1d'] / features_df['volatility_regime']
            features_df['rsi_volatility_ratio'] = features_df['rsi_14'] / features_df['volatility_regime']
            
            # Convert numpy arrays to pandas Series for rolling operations
            close_series = pd.Series(close)
            
            # Pattern Duration Features
            for window in [3, 5, 10, 20]:
                # Streak detection
                features_df[f'up_streak_{window}'] = pd.Series(features_df['price_change_1d'] > 0).rolling(window=window).sum()
                features_df[f'down_streak_{window}'] = pd.Series(features_df['price_change_1d'] < 0).rolling(window=window).sum()
                
                # Range analysis
                features_df[f'high_{window}d'] = close_series / close_series.rolling(window=window).max() - 1
                features_df[f'low_{window}d'] = close_series / close_series.rolling(window=window).min() - 1
                
                # Volatility structure
                features_df[f'vol_regime_{window}d'] = pd.Series(features_df['volatility_regime']).rolling(window=window).std()
                
            # Consolidation/Breakout Detection
            features_df['price_channel_width'] = (close_series.rolling(20).max() - close_series.rolling(20).min()) / close_series.rolling(20).mean()
            features_df['breakout_strength'] = (close_series - close_series.rolling(20).mean()) / (close_series.rolling(20).std() + 0.0001)
            
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
                               base_threshold: float = 0.002) -> pd.DataFrame:
        """
        Create binary labels for training with dynamic thresholds and risk/reward filtering.
        
        Args:
            df: DataFrame with features
            forward_periods: How many periods to look ahead
            base_threshold: Base threshold for minimum move (will be adjusted by volatility)
            
        Returns:
            DataFrame with 'target' column
        """
        if 'close' not in df.columns or 'atr_14' not in df.columns:
            return df
            
        # Calculate market regime metrics
        volatility = df['atr_14'] / df['close']
        vol_percentile = volatility.rolling(window=100, min_periods=20).rank(pct=True)
        trend_strength = abs(df['sma_20'] - df['sma_50']) / df['atr_14']
        
        # Adaptive thresholds based on regime
        base_vol_threshold = base_threshold * (volatility / volatility.rolling(window=100, min_periods=20).mean())
        trend_adjustment = np.where(trend_strength > trend_strength.rolling(100, min_periods=20).median(),
                                  1.2,  # Higher threshold in strong trends
                                  0.8)  # Lower threshold in ranges
        
        dynamic_threshold = base_vol_threshold * trend_adjustment
        
        # Calculate returns and risk metrics
        df['future_return'] = (df['close'].shift(-forward_periods) - df['close']) / df['close']
        df['max_adverse_excursion'] = df['close'].rolling(window=forward_periods).min().shift(-forward_periods)
        df['max_adverse_excursion'] = (df['max_adverse_excursion'] - df['close']) / df['close']
        df['avg_volume'] = df['volume'].rolling(20).mean()
        
        # Initialize target
        df['target'] = 0
        
        # Adaptive reward/risk ratio based on volatility regime
        min_reward_ratio = np.where(vol_percentile > 0.7,
                                  2.0,  # Higher RR needed in high volatility
                                  1.3)  # Lower RR acceptable in low volatility
        
        # Enhanced signal filtering
        volume_filter = df['volume'] > df['avg_volume'] * 0.8  # Require decent volume
        trend_filter = df['close'] > df['sma_20']  # Basic trend filter
        
        # Long signals with regime-aware filtering
        # More balanced signal criteria
        min_reward_ratio = 2.0  # Back to 2.0 for more signals
        
        # Volume confirmation (relaxed)
        volume_confirmed = (
            (df['volume'] > df['volume'].rolling(window=20).mean() * 1.1) &  # 10% above average volume
            (df['volume_trend'] > 1.0)  # Neutral or rising volume
        )
        
        # Trend confirmation (relaxed but still meaningful)
        trend_confirmed = (
            (df['sma_20'] > df['sma_50']) &  # Basic uptrend
            (df['close'] > df['sma_20']) &  # Price above short MA
            (df['rsi_14'] > 30) &  # Not oversold
            (df['rsi_14'] < 75)  # Not too overbought
        )
        
        # Price momentum (relaxed)
        momentum_confirmed = (
            (df['macd'] > df['macd_signal']) |  # MACD bullish
            (df['price_change_5d'] > 0)  # Or positive 5-day return
        )
        
        # Volatility check (simplified)
        volatility_confirmed = (
            df['volatility_regime'] < df['volatility_regime'].rolling(window=20).mean() * 1.2  # Not extremely volatile
        )
        
        # Enhanced signal criteria (more balanced)
        long_mask = (
            (df['future_return'] > dynamic_threshold) &  # Standard threshold
            (df['future_return'] >= abs(df['max_adverse_excursion']) * min_reward_ratio) &
            volume_confirmed &
            trend_confirmed &
            (momentum_confirmed | volatility_confirmed)  # Need either momentum OR good volatility
        )
        
        df.loc[long_mask, 'target'] = 1
        
        # Drop rows without future data
        df = df.dropna(subset=['future_return', 'target'])
        
        # Drop the helper column
        df = df.drop('future_return', axis=1)
        
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names"""
        return self.feature_names
