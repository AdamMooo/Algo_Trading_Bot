from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from execution.paper_trader import execute_trade, execute_trade_and_wait_for_fill
from risk.position_sizing import calculate_qty
from strategy.simple_sma import simple_sma_strategy
from monitor.trade_monitor import TradeMonitor
from monitor.news_monitor import NewsMonitor
from ml.ml_trader import MLTrader
from datetime import datetime, timedelta, time
import time as t_module
import numpy as np
from alpaca.trading.client import TradingClient
import subprocess
import sys
from config import *

# Connect to Alpaca (both stock and crypto clients)
stock_data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
crypto_data_client = CryptoHistoricalDataClient(API_KEY, API_SECRET)

def check_and_retrain_model():
    """Check if ML model needs retraining."""
    needs_training = False
    
    if not os.path.exists(MODEL_PATH):
        print("ML Model: Not found - training new model...")
        needs_training = True
    else:
        model_age = datetime.now().timestamp() - os.path.getmtime(MODEL_PATH)
        hours_old = model_age / 3600
        
        if hours_old > MODEL_RETRAIN_HOURS:
            print(f"ML Model: {hours_old:.1f}h old - retraining...")
            needs_training = True
        else:
            print(f"ML Model: Fresh ({hours_old:.1f}h old)")
    
    if needs_training:
        training_cmd = [sys.executable, "ml/training_pipeline.py"]
        
        try:
            result = subprocess.run(training_cmd, check=True, capture_output=True)
            if result.returncode == 0:
                print("ML Model: Training complete")
            else:
                print("ML Model: Training completed with warnings")
        except Exception as e:
            print(f"ML Model: Training error - {e}")
            print("Continuing with existing model...")
    
    print()

def is_crypto_symbol(symbol):
    """Check if symbol is a crypto pair"""
    return '/' in symbol  # Crypto pairs have format: BTC/USD, ETH/USD, etc.

# Initialize monitors
trade_monitor = TradeMonitor()
news_monitor = NewsMonitor()
ml_trader = MLTrader()

# Dynamic symbol selection based on market hours
SECTOR_STOCKS = STOCK_SYMBOLS  # Will be updated automatically

# Market hours (ET)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)



def fetch_positions():
    """
    Fetches all open positions from Alpaca and returns a dict:
    {symbol: {'entry_price': float, 'qty': int, 'value': float}}
    """
    positions = {}
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    try:
        alpaca_positions = trading_client.get_all_positions()
        for pos in alpaca_positions:
            symbol = pos.symbol
            
            # Normalize crypto symbols: BTCUSD -> BTC/USD for consistency
            if len(symbol) > 3 and symbol.endswith('USD') and '/' not in symbol:
                # This is a crypto symbol without slash (BTCUSD)
                # Convert to slash format (BTC/USD)
                base = symbol[:-3]  # Remove 'USD'
                symbol = f"{base}/USD"
            
            qty = float(pos.qty)  # Keep as float for crypto
            entry_price = float(pos.avg_entry_price)
            market_value = float(pos.market_value)
            positions[symbol] = {
                'entry_price': entry_price, 
                'qty': qty,
                'value': market_value
            }
    except Exception as e:
        print(f"Error fetching positions: {e}")
    return positions

def get_portfolio_value():
    """Get total portfolio value from Alpaca."""
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    try:
        account = trading_client.get_account()
        return float(account.portfolio_value)
    except Exception as e:
        print(f"Error fetching portfolio value: {e}")
        return 100000  # Default fallback

def is_market_open():
    """Check if stock market is open"""
    now = datetime.now()
    return now.weekday() < 5 and MARKET_OPEN <= now.time() <= MARKET_CLOSE

def run_strategy():
    """Main strategy execution function"""
    global SECTOR_STOCKS
    now = datetime.now()
    
    # Switch between stocks and crypto based on market hours
    if is_market_open():
        SECTOR_STOCKS = STOCK_SYMBOLS + CRYPTO_SYMBOLS
        print(f"Market Open: Trading {len(SECTOR_STOCKS)} symbols ({now.strftime('%H:%M')})")
    elif TRADE_CRYPTO_AFTER_HOURS:
        SECTOR_STOCKS = CRYPTO_SYMBOLS
        print(f"After Hours: Trading {len(CRYPTO_SYMBOLS)} crypto ({now.strftime('%H:%M')})")
    else:
        print("Market closed - waiting...")
        return

    # Always get latest positions from Alpaca
    positions = fetch_positions()
    portfolio_value = get_portfolio_value()
    
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for sym, pos in positions.items():
            direction = "LONG" if pos['qty'] > 0 else "SHORT"
            
            # Get current price
            try:
                if is_crypto_symbol(sym):
                    current_bars = crypto_data_client.get_crypto_bars(
                        CryptoBarsRequest(
                            symbol_or_symbols=sym,
                            timeframe=TimeFrame.Minute,
                            start=now - timedelta(minutes=5),
                            end=now
                        )
                    ).df
                else:
                    current_bars = stock_data_client.get_stock_bars(
                        StockBarsRequest(
                            symbol_or_symbols=sym,
                            timeframe=TimeFrame.Minute,
                            start=now - timedelta(minutes=5),
                            end=now
                        )
                    ).df
                
                if not current_bars.empty:
                    current_price = current_bars.iloc[-1]['close']
                    trade_monitor.update_position_prices(sym, current_price)
                    
                    # Calculate P&L
                    pnl = (current_price - pos['entry_price']) * pos['qty']
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    
                    # Status indicators
                    status = ""
                    if pnl <= MAX_LOSS:
                        status = " [STOP LOSS]"
                    elif pnl >= MAX_PROFIT:
                        status = " [TAKE PROFIT]"
                    elif pnl < MAX_LOSS * 0.8:
                        status = " [NEAR STOP]"
                    elif pnl > MAX_PROFIT * 0.8:
                        status = " [NEAR PROFIT]"
                    
                    print(f"  {sym} ({direction}): ${pnl:.2f} ({pnl_pct:+.1f}%){status}")
                        
            except Exception as e:
                print(f"  {sym}: Error getting price")
    else:
        print("No open positions")
    
    print(f"Portfolio: ${portfolio_value:,.0f}")
    
    # Trading stats
    stats = trade_monitor.get_trade_stats()
    if stats:
        print(f"Stats: {stats['total_trades']} trades, {stats['win_rate']:.1%} win rate, ${stats['total_pnl']:.0f} PnL")

    for symbol in SECTOR_STOCKS:
        start = now - timedelta(days=4)  # 4-day lookback for balance between noise filtering and responsiveness
        end = now
        
        # Fetch data with try/except - use appropriate client
        try:
            if is_crypto_symbol(symbol):
                request_params = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end
                )
                bars = crypto_data_client.get_crypto_bars(request_params).df
            else:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol, 
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end
                )
                bars = stock_data_client.get_stock_bars(request_params).df
        except Exception as e:
            print(f"Data fetch error for {symbol}: {e}")
            continue
        
        if bars.empty:
            print(f"No data for {symbol}")
            continue
        
        # Run SMA strategy
        signals = simple_sma_strategy(bars)
        sma_signal = signals.iloc[-1]['Signal']
        latest_price = signals.iloc[-1]['close']
        
        # Get ML signal - only use if HIGHLY confident (>0.6)
        ml_signal_info = ml_trader.generate_signal(symbol, bars)
        ml_signal = ml_signal_info['signal']
        ml_confidence = ml_signal_info['confidence']
        
        # SIMPLIFIED SIGNAL LOGIC - HIGH CONFIDENCE ONLY:
        # If ML is confident (>0.55), use ML signal
        # Otherwise, use SMA signal
        # Higher threshold for larger positions = better quality trades
        if ml_confidence > 0.55 and ml_signal != 0:
            final_signal = ml_signal
            signal_source = f"ML ({ml_confidence:.2f})"
        else:
            final_signal = sma_signal
            signal_source = "SMA"
        
        # Calculate volatility
        bars['return'] = bars['close'].pct_change()
        vol = bars['return'].std()

        # Check news sentiment
        news_monitor.print_latest_news(symbol)
        should_skip, pos_factor = news_monitor.should_skip_trade(symbol, final_signal)
        if should_skip:
            print(f"  {symbol}: Skipping due to news sentiment")
            continue
            
        print(f"  {symbol}: SMA={sma_signal}, ML={ml_signal} ({ml_confidence:.0%}), Using={signal_source}")
            
        # Get existing position
        position = positions.get(symbol)
            
        # POSITION SIZING - Different sizes for ML vs SMA with leverage options
        base_cash = CASH_PER_TRADE
        leverage_applied = 1.0  # Track leverage for monitoring
        
        # Position sizing based on signal type and confidence
        if signal_source == "SMA":
            base_cash = CASH_PER_TRADE * 0.20
        else:
            if ml_confidence >= LEVERAGE_CONFIDENCE_THRESHOLD:  # >70%
                leverage_applied = min(LEVERAGE_MULTIPLIER, MAX_LEVERAGE_PER_TRADE)
                base_cash = base_cash * leverage_applied
            elif ml_confidence > 0.60:  # 60-70%
                base_cash = base_cash * ML_CONFIDENCE_MULTIPLIER
        
        # Check if trading crypto (needs fractional quantities)
        is_crypto = is_crypto_symbol(symbol)
        base_qty = calculate_qty(base_cash, latest_price, volatility=vol, is_crypto=is_crypto)
        
        # Adjust for news sentiment (0.5 to 1.5x)
        if is_crypto:
            qty = base_qty * pos_factor  # Keep fractional for crypto
            qty = max(0.001, qty)  # At least 0.001 crypto units
        else:
            qty = int(base_qty * pos_factor)  # Integer for stocks
            qty = max(1, qty)  # At least 1 share


        # --- Exit logic ---
        pnl = 0
        if position:
            entry_price = position['entry_price']
            position_qty = position['qty']
            # Calculate PnL for open position (works for both long and short)
            pnl = (latest_price - entry_price) * position_qty
            
            # Check exit conditions - MORE RESPONSIVE
            exit_reason = None
            if pnl >= MAX_PROFIT:
                exit_reason = f"TAKE PROFIT (${pnl:.2f} >= ${MAX_PROFIT})"
            elif pnl <= MAX_LOSS:
                exit_reason = f"STOP LOSS (${pnl:.2f} <= ${MAX_LOSS})"
            elif final_signal == 0:
                exit_reason = f"SIGNAL NEUTRAL (was holding, now exit)"
            elif (final_signal > 0 and position_qty < 0) or (final_signal < 0 and position_qty > 0):
                exit_reason = f"SIGNAL REVERSAL (holding {'LONG' if position_qty > 0 else 'SHORT'}, signal says {'LONG' if final_signal > 0 else 'SHORT'})"
            
            if exit_reason:
                print(f"  {symbol}: EXIT - {exit_reason} (P&L: ${pnl:.2f})")
                execute_trade(symbol, 0, qty=abs(position_qty), latest_price=latest_price)
                trade_monitor.record_trade(
                    symbol=symbol,
                    side="SELL" if position_qty > 0 else "BUY",
                    qty=abs(position_qty),
                    price=latest_price,
                    trade_type="exit"
                )
                continue
            else:
                print(f"  {symbol}: Holding (P&L: ${pnl:.2f})")

        # --- Entry/Reverse logic ---
        if final_signal != 0:
            # CRYPTO RESTRICTION: Can't short crypto on Alpaca
            # WORKAROUND: When we get a short signal, exit any long positions
            if is_crypto_symbol(symbol) and final_signal < 0:
                if position and position['qty'] > 0:
                    print(f"  {symbol}: Crypto short signal - exiting long position")
                    execute_trade(symbol, 0, qty=abs(position['qty']), latest_price=latest_price)
                    trade_monitor.record_trade(
                        symbol=symbol,
                        side="SELL",
                        qty=abs(position['qty']),
                        price=latest_price,
                        trade_type="exit"
                    )
                else:
                    print(f"  {symbol}: Crypto short signal - no position to exit")
                continue
            
            if symbol not in positions:
                action = "BUY" if final_signal > 0 else "SELL"
                leverage_info = f" ({leverage_applied:.1f}x leverage)" if leverage_applied > 1.0 else ""
                print(f"  {symbol}: {action} {qty:.2f} @ ${latest_price:.2f} = ${qty * latest_price:.0f}{leverage_info}")
                
                execute_trade(symbol, final_signal, qty=qty, latest_price=latest_price)
                trade_monitor.record_trade(
                    symbol=symbol,
                    side="BUY" if final_signal > 0 else "SELL",
                    qty=qty * (1 if final_signal > 0 else -1),
                    price=latest_price,
                    trade_type="entry"
                )
            else:
                # Reverse position if signal direction changes
                position_qty = positions[symbol]['qty']
                if (final_signal > 0 and position_qty < 0) or (final_signal < 0 and position_qty > 0):
                    print(f"  {symbol}: Reversing position (Signal: {final_signal})")
                    close_order = execute_trade_and_wait_for_fill(symbol, 0, qty=abs(position_qty), latest_price=latest_price)
                    if close_order:
                        trade_monitor.record_trade(
                            symbol=symbol,
                            side="BUY" if position_qty < 0 else "SELL",
                            qty=abs(position_qty),
                            price=latest_price,
                            trade_type="reversal"
                        )
                    execute_trade(symbol, final_signal, qty=qty, latest_price=latest_price)
                    trade_monitor.record_trade(
                        symbol=symbol,
                        side="BUY" if final_signal > 0 else "SELL",
                        qty=qty * (1 if final_signal > 0 else -1),
                        price=latest_price,
                        trade_type="entry"
                    )

# Main loop
print("Trading Bot Starting...")
check_and_retrain_model()

print(f"Market Hours: {len(STOCK_SYMBOLS)} stocks + {len(CRYPTO_SYMBOLS)} crypto")
print(f"After Hours: {len(CRYPTO_SYMBOLS)} crypto only")
print(f"Position Sizing: SMA=${int(CASH_PER_TRADE * 0.20):,}, ML=${CASH_PER_TRADE:,}, Leverage=${int(CASH_PER_TRADE * LEVERAGE_MULTIPLIER):,}")
print()

while True:
    if is_market_open():
        run_strategy()
        print("Sleeping 10 minutes...")
        t_module.sleep(600)
    elif TRADE_CRYPTO_AFTER_HOURS:
        run_strategy()
        print("Sleeping 10 minutes...")
        t_module.sleep(600)
    else:
        print("Market closed - waiting...")
        t_module.sleep(1800)

