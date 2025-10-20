import pandas as pd

def simple_sma_strategy(df, fast=5, slow=20, rsi_period=14,
                        vol_period=10, vol_threshold=0.03, trend_period=200):
    """
    SMA + RSI + Volatility + Trend strategy with dynamic position sizing.
    df: DataFrame with 'close' column
    fast: short-term SMA period
    slow: long-term SMA period
    rsi_period: period for RSI calculation
    vol_period: rolling period to calculate volatility
    vol_threshold: max allowed daily return volatility to take trades
    trend_period: period for long-term trend filter
    Returns df with columns: close, SMA_fast, SMA_slow, SMA_trend, RSI, volatility, Signal, PositionSizeFactor
    """
    df = df.copy()

    # --- Short & long SMA ---
    df['SMA_fast'] = df['close'].rolling(fast).mean()
    df['SMA_slow'] = df['close'].rolling(slow).mean()
    df['SMA_trend'] = df['close'].rolling(trend_period).mean()

    # --- SMA crossover signal ---
    df['SMA_diff'] = df['SMA_fast'] - df['SMA_slow']
    df['Signal'] = 0
    df.loc[df['SMA_diff'] > 0, 'Signal'] = 1   # buy
    df.loc[df['SMA_diff'] < 0, 'Signal'] = -1  # sell

    # --- RSI calculation ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- RSI filter ---
    df.loc[df['RSI'] > 70, 'Signal'] = -1  # overbought → sell
    df.loc[df['RSI'] < 30, 'Signal'] = 1   # oversold → buy

    # --- Volatility filter ---
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(vol_period).std()
    df.loc[df['volatility'] > vol_threshold, 'Signal'] = 0  # skip high-volatility trades

    # --- Trend filter ---
    df.loc[df['close'] < df['SMA_trend'], 'Signal'] = -1  # below trend → sell
    df.loc[df['close'] > df['SMA_trend'], 'Signal'] = 1   # above trend → buy

    # --- Dynamic position sizing ---
    # Scale between 0.1–1 based on volatility
    df['PositionSizeFactor'] = df['volatility'].apply(lambda v: max(0.1, min(1, vol_threshold / v)) if v > 0 else 1)

    # Keep only relevant columns
    df = df[['close', 'SMA_fast', 'SMA_slow', 'SMA_trend', 'RSI', 'volatility', 'Signal', 'PositionSizeFactor']]

    return df