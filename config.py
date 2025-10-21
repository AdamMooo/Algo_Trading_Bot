"""
Configuration file for Algo Trading Bot
Centralized configuration for all trading parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Trading Configuration
CASH_PER_TRADE = 5000
MAX_PROFIT = 500
MAX_LOSS = -200

# ML Configuration
ML_CONFIDENCE_MULTIPLIER = 1.5
LEVERAGE_CONFIDENCE_THRESHOLD = 0.70
LEVERAGE_MULTIPLIER = 2.0
MAX_LEVERAGE_PER_TRADE = 3.0

# Market Hours (ET)
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

# Trading Settings
TRADE_CRYPTO_AFTER_HOURS = True

# Stock Universe
STOCK_SYMBOLS = [
    # Major Index ETFs
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO",
    
    # Large Cap Tech
    "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "TSLA", "META", "NFLX",
    
    # Additional Tech Giants
    "ORCL", "INTC", "CSCO", "IBM", "NOW", "SNOW", "PLTR", "ZM",
    
    # Growth & Cloud Stocks
    "AMZN", "CRM", "ADBE", "SHOP", "SQ", "PYPL", "ROKU", "SPOT",
    
    # Financials
    "JPM", "BAC", "GS", "V", "MA", "AXP", "WFC", "C", "BLK",
    
    # Consumer & Retail
    "WMT", "COST", "HD", "TGT", "LOW", "MCD", "SBUX", "NKE", "DIS",
    
    # Healthcare & Biotech
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "LLY",
    
    # Energy & Materials
    "XOM", "CVX", "COP", "EOG", "SLB", "FCX", "NEM", "LIN",
    
    # Industrials
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "RTX",
    
    # Telecom & Utilities
    "VZ", "T", "TMUS", "NEE", "SO", "DUK", "AEP", "EXC",
    
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR",
    
    # International Exposure
    "ASML", "TSM", "BABA", "PDD", "NIO", "XPEV", "LI",
]

# Crypto Universe
CRYPTO_SYMBOLS = [
    # Top 5 - Largest Market Cap
    "BTC/USD",    # Bitcoin - #1 by market cap
    "ETH/USD",    # Ethereum - #2
    "SOL/USD",    # Solana - #5
    "XRP/USD",    # Ripple - #4
    
    # Large Cap Altcoins
    "AVAX/USD",   # Avalanche
    "DOT/USD",    # Polkadot

    # DeFi Blue Chips
    "LINK/USD",   # Chainlink
    "UNI/USD",    # Uniswap
    "AAVE/USD",   # Aave
    "LTC/USD",    # Litecoin
]

# Model Training Configuration
TRAINING_DAYS = 90 # Use past 90 days of data for training
TRAINING_FORWARD_PERIODS = 3 # Predict 3 periods ahead
TRAINING_THRESHOLD = 0.0015 # 0.15% threshold for labeling
MODEL_RETRAIN_HOURS = 72 # Retrain model every 72 hours

# File Paths
MODEL_PATH = "ml/models/random_forest.joblib"
FEATURE_NAMES_PATH = "ml/models/feature_names.txt"
LOGS_DIR = "logs"
