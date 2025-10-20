# Algo Trading Bot

Professional algorithmic trading bot with ML-powered decision making and leverage capabilities.

## Features

- **Hybrid Trading**: Stocks during market hours, crypto 24/7
- **ML-Powered**: Random Forest model with confidence-based position sizing
- **Leverage Trading**: 2x leverage for ultra-high confidence trades (>70%)
- **Risk Management**: Stop loss and take profit automation
- **News Monitoring**: Sentiment analysis for trade filtering
- **85+ Stocks**: Comprehensive coverage across 12 sectors

## Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys**:
   Create `.env` file with your Alpaca credentials:
   ```
   APCA_API_KEY_ID=your_key
   APCA_API_SECRET_KEY=your_secret
   ```

3. **Train Model** (optional - auto-retrains every 3 days):
   ```bash
   python ml/training_pipeline.py
   ```

4. **Run Trading Bot**:
   ```bash
   python main.py
   ```

## Project Structure

```
algo_trading_bot/
├── main.py                 # Main trading bot
├── config.py              # Centralized configuration
├── execution/             # Trade execution
├── risk/                  # Position sizing
├── strategy/              # Trading strategies
├── ml/                    # Machine learning
│   ├── training_pipeline.py  # Model training
│   ├── ml_trader.py
│   └── models/            # Trained models
├── monitor/               # Trade & news monitoring
└── logs/                  # Trading logs
```

## Configuration

All trading parameters are centralized in `config.py`:

- **Position Sizing**: SMA=$3,000, ML=$15,000, Leverage=$30,000
- **Risk Management**: Stop loss=$250, Take profit=$600
- **Leverage**: 2x for >70% ML confidence
- **Market Hours**: 9:30 AM - 4:00 PM ET

## Trading Logic

1. **Signal Generation**: SMA + ML signals with confidence scoring
2. **Position Sizing**: Based on signal type and ML confidence
3. **Risk Management**: Automated stop loss and take profit
4. **Market Hours**: Automatic switching between stocks and crypto

## Model Training

The ML model is automatically retrained every 24 hours or can be manually trained:

```bash
python ml/training_pipeline.py --optimize-hyperparams
```

## Monitoring

- Real-time position monitoring
- Performance statistics
- News sentiment analysis
- Trade logging and analysis

## Requirements

- Python 3.8+
- Alpaca API credentials
- Internet connection for data and news feeds