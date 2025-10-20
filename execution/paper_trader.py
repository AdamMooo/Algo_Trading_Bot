from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os
from dotenv import load_dotenv
import time as t_module

# Load API keys
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# Connect to Alpaca paper trading
trading_client = TradingClient(API_KEY, API_SECRET, paper=True)

def execute_trade(symbol, signal, qty=2, max_position_value=100000, latest_price=None):
    """
    symbol: stock ticker (e.g., "AAPL") or crypto pair (e.g., "BTC/USD")
    signal: 1 = buy, -1 = sell, 0 = exit/close position
    qty: number of shares/units (can be fractional for crypto)
    max_position_value: max $ exposure per symbol
    latest_price: latest stock price (required for max position check)
    """
    # List of symbols that cannot be sold short
    NON_SHORTABLE = {'VXX', 'UVXY'}  # Add more as needed
    
    # Check if symbol is crypto
    is_crypto = '/' in symbol
    
    # Convert crypto symbol format: BTC/USD -> BTCUSD for Alpaca API
    alpaca_symbol = symbol.replace('/', '') if is_crypto else symbol
    
    # Determine time in force based on asset type
    if is_crypto:
        tif = TimeInForce.GTC  # Good Till Cancelled for crypto (24/7)
    else:
        tif = TimeInForce.DAY  # Day order for stocks
    
    # Check if trying to short a non-shortable security
    if signal == -1 and symbol in NON_SHORTABLE:
        print(f"Warning: {symbol} cannot be sold short, skipping trade")
        return None
    if latest_price is None:
        raise ValueError("latest_price is required for max position check.")

    # If signal is 0, we want to close the position (sell all shares)
    if signal == 0:
        # Try to get current position to determine side and qty
        try:
            pos = trading_client.get_open_position(alpaca_symbol)
            position_qty = float(pos.qty)  # Use float for crypto
            if position_qty == 0:
                print(f"No position to close for {symbol}.")
                return
            side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=abs(position_qty),
                side=side,
                time_in_force=tif
            )
            order = trading_client.submit_order(order_data)
            print(f"Closed position for {symbol}, qty={position_qty}")
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
        return

    # Check current position
    current_position_value = 0
    try:
        pos = trading_client.get_position(alpaca_symbol)
        current_position_value = float(pos.market_value)
    except Exception:
        current_position_value = 0  # no current position

    # Check max exposure
    if current_position_value + qty * latest_price > max_position_value:
        print(f"Trade skipped: would exceed max position of ${max_position_value}")
        return

    # Determine buy/sell
    side = OrderSide.BUY if signal == 1 else OrderSide.SELL

    try:
        order_data = MarketOrderRequest(
            symbol=alpaca_symbol,
            qty=qty,
            side=side,
            time_in_force=tif  # Use appropriate TIF for asset type
        )
        order = trading_client.submit_order(order_data)
        print(f"{side} order submitted for {symbol}, qty={qty}")
    except Exception as e:
        print(f"Error executing trade: {e}")

def execute_trade_and_wait_for_fill(symbol, signal, qty=2, max_position_value=100000, latest_price=None):
    """
    Like execute_trade, but if signal==0 (close), waits for the order to fill before returning order id.
    Returns the order object if submitted, else None.
    """
    if latest_price is None:
        raise ValueError("latest_price is required for max position check.")

    # Check if symbol is crypto
    is_crypto = '/' in symbol
    
    # Convert crypto symbol format: BTC/USD -> BTCUSD for Alpaca API
    alpaca_symbol = symbol.replace('/', '') if is_crypto else symbol
    
    # Determine time in force based on asset type
    if is_crypto:
        tif = TimeInForce.GTC  # Good Till Cancelled for crypto
    else:
        tif = TimeInForce.DAY  # Day order for stocks

    if signal == 0:
        try:
            pos = trading_client.get_open_position(alpaca_symbol)
            position_qty = float(pos.qty)  # Use float for crypto
            if position_qty == 0:
                print(f"No position to close for {symbol}.")
                return None
            side = OrderSide.SELL if position_qty > 0 else OrderSide.BUY
            order_data = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=abs(position_qty),
                side=side,
                time_in_force=tif
            )
            order = trading_client.submit_order(order_data)
            print(f"Closed position for {symbol}, qty={position_qty}, waiting for fill...")
            # Wait for fill
            order_id = order.id
            for _ in range(30):  # wait up to ~30*2=60s
                o = trading_client.get_order_by_id(order_id)
                if o.status == 'filled':
                    print(f"Order {order_id} filled for {symbol}.")
                    return order
                t_module.sleep(2)
            print(f"Warning: Order {order_id} for {symbol} not filled after 60s.")
            return order
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")
        return None
    else:
        # fallback to normal execute_trade logic for open/scale
        return execute_trade(symbol, signal, qty, max_position_value, latest_price)

