from typing import Dict, List, Tuple
from alpaca.trading.client import TradingClient
import numpy as np
from datetime import datetime

class PortfolioManager:
    def __init__(self, trading_client: TradingClient, max_portfolio_value: float = 100000,
                 max_position_size: float = 0.2):
        self.trading_client = trading_client
        self.max_portfolio_value = max_portfolio_value
        self.max_position_size = max_position_size  # Maximum size of any single position (20% default)
        self.positions = {}
        # pending_signals: list of tuples (symbol, signal_strength, price, qty, timestamp)
        self.pending_signals = []
        
    def refresh_account_info(self) -> Tuple[float, float]:
        """Get current buying power and portfolio value"""
        account = self.trading_client.get_account()
        return float(account.buying_power), float(account.portfolio_value)
    
    def update_positions(self) -> Dict:
        """Update current positions"""
        try:
            positions = self.trading_client.get_all_positions()
            self.positions = {
                p.symbol: {
                    'qty': float(p.qty),
                    'market_value': float(p.market_value),
                    'entry_price': float(p.avg_entry_price)
                } for p in positions
            }
            return self.positions
        except Exception as e:
            print(f"Error updating positions: {e}")
            return {}
    
    def can_take_trade(self, symbol: str, proposed_value: float) -> bool:
        """Check if we can take a new trade"""
        buying_power, portfolio_value = self.refresh_account_info()
        
        # Check if we have enough buying power
        if proposed_value > buying_power:
            return False
            
        # Check if this would exceed max position size
        if proposed_value > (portfolio_value * self.max_position_size):
            return False
            
        return True
    
    def add_trade_signal(self, symbol: str, signal_strength: float, price: float, qty: int):
        """Add a trade signal to be prioritized. Stores a timestamp for stable sorting.

        Signals are stored as (symbol, strength, price, qty, timestamp).
        Sorting priority: strength (desc), notional size (desc), timestamp (asc).
        """
        proposed_value = abs(qty * price)
        strength = float(signal_strength)
        ts = datetime.now().timestamp()

        print(f"\nAnalyzing signal for {symbol}:")
        print(f"- Signal Strength: {strength:.3f}")
        print(f"- Proposed Value: ${proposed_value:,.2f}")

        buying_power, portfolio_value = self.refresh_account_info()
        print(f"- Current Buying Power: ${buying_power:,.2f}")
        print(f"- Max Position Size: ${(portfolio_value * self.max_position_size):,.2f}")

        # Append with timestamp
        self.pending_signals.append((symbol, strength, price, qty, ts))

        # Stable sort: primary by strength desc, secondary by notional desc, tertiary by timestamp asc
        self.pending_signals.sort(key=lambda x: (x[1], abs(x[2] * x[3]), -x[4]), reverse=True)

        print(f"- Signal added to queue (Position {len(self.pending_signals)} in line)")
    
    def get_position_to_close(self) -> str:
        """Find weakest position to potentially close"""
        if not self.positions:
            return None
            
        # Get current signals for existing positions
        position_signals = []
        for symbol in self.positions.keys():
            # You'll need to implement get_current_signal() in your strategy
            signal = self.get_current_signal(symbol)
            position_signals.append((symbol, abs(signal)))
            
        # Return symbol with weakest signal
        if position_signals:
            weakest_position = min(position_signals, key=lambda x: x[1])
            return weakest_position[0]
        return None
    
    def process_pending_signals(self):
        """Process pending trade signals in priority order"""
        if not self.pending_signals:
            print("\nNo pending signals to process")
            return
            
        print("\n=== Processing Pending Signals ===")
        print(f"Total signals in queue: {len(self.pending_signals)}")
        
        buying_power, portfolio_value = self.refresh_account_info()
        print(f"Current Buying Power: ${buying_power:,.2f}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        
        # Ensure queue is sorted by priority right before processing (in case external changes occurred)
        self.pending_signals.sort(key=lambda x: (x[1], abs(x[2] * x[3]), -x[4]), reverse=True)

        # Print queue snapshot
        print("\nSignal queue (priority ->):")
        for idx, item in enumerate(self.pending_signals, 1):
            s_sym, s_strength, s_price, s_qty, s_ts = item
            print(f" {idx}. {s_sym} | strength={s_strength:.3f} | value=${abs(s_price * s_qty):,.2f}")

        for i, (symbol, strength, price, qty, ts) in enumerate(self.pending_signals, 1):
            proposed_value = abs(price * qty)
            print(f"\nProcessing Signal #{i} - {symbol}:")
            print(f"- Signal Strength: {strength:.3f}")
            print(f"- Required Capital: ${proposed_value:,.2f}")
            
            if self.can_take_trade(symbol, proposed_value):
                print(f"✓ Trade approved for {symbol}")
                yield (symbol, qty, price)
            else:
                print(f"⚠ Insufficient capital for {symbol}")
                # Try to free up capital by closing weakest position
                weak_pos = self.get_position_to_close()
                if weak_pos and self.positions[weak_pos]['market_value'] >= proposed_value:
                    print(f"- Closing weak position in {weak_pos} to free up capital")
                    yield (weak_pos, 0, None)  # Signal to close position
                    # Then try the new trade
                    if self.can_take_trade(symbol, proposed_value):
                        print(f"✓ Trade now possible for {symbol} after freeing capital")
                        yield (symbol, qty, price)
                    else:
                        print(f"✗ Still insufficient capital for {symbol} after freeing position")
                else:
                    print(f"✗ Could not free up enough capital for {symbol}")
        
        print("\n=== Signal Processing Complete ===")
        # Clear pending signals
        self.pending_signals = []
    
    def get_current_signal(self, symbol: str) -> float:
        """Get current signal strength for a symbol"""
        # This should be implemented to match your strategy's signal calculation
        # For now, returning 0 as placeholder
        return 0.0