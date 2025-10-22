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
        # pending_signals: list of tuples (symbol, signal_strength, price, qty, source, timestamp)
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
    
    def add_trade_signal(self, symbol: str, signal_strength: float, price: float, qty: int, source: str = "SMA"):
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

        # Append with timestamp and source
        self.pending_signals.append((symbol, strength, price, qty, source, ts))

        # Stable sort: prioritize ML signals, then strength desc, notional desc, older first
        def sort_key(item):
            s_sym, s_strength, s_price, s_qty, s_source, s_ts = item
            source_priority = 1 if s_source.upper().startswith('ML') else 0
            return (source_priority, s_strength, abs(s_price * s_qty), -s_ts)

        self.pending_signals.sort(key=sort_key, reverse=True)

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
        
        # Use the same sort key as get_queue_snapshot
        def sort_key(item):
            s_sym, s_strength, s_price, s_qty, s_source, s_ts = item
            source_priority = 1 if s_source.upper().startswith('ML') else 0
            return (source_priority, s_strength, abs(s_price * s_qty), -float(s_ts))

        # Sort using the correct key function
        self.pending_signals.sort(key=sort_key, reverse=True)

        # Print queue snapshot
        print("\nSignal queue (priority ->):")
        for idx, item in enumerate(self.pending_signals, 1):
            s_sym, s_strength, s_price, s_qty, s_source, s_ts = item
            print(f" {idx}. {s_sym} | src={s_source} | strength={s_strength:.3f} | value=${abs(s_price * s_qty):,.2f}")
        for i, (symbol, strength, price, qty, source, ts) in enumerate(self.pending_signals, 1):
            proposed_value = abs(price * qty)
            print(f"\nProcessing Signal #{i} - {symbol}:")
            print(f"- Signal Strength: {strength:.3f} (source={source})")
            print(f"- Required Capital: ${proposed_value:,.2f}")
            
            if self.can_take_trade(symbol, proposed_value):
                print(f"✓ Trade approved for {symbol}")
                yield (symbol, qty, price)
            else:
                print(f"Insufficient capital for {symbol}")
                # If this is an SMA-based (lower priority) signal, skip it instead of trying to close
                if source.upper().startswith('SMA'):
                    print(f"- Skipping SMA signal for {symbol} due to insufficient buying power")
                    continue

                # For ML signals: try to free capital by closing one or more weakest positions
                buying_power, portfolio_value = self.refresh_account_info()
                needed = max(0.0, proposed_value - buying_power)
                print(f"- Need to free approximately ${needed:,.2f} to take this trade")

                # Build list of candidate positions (symbol, weakness, market_value)
                candidates = []
                for pos_sym, pos in self.positions.items():
                    pos_signal = abs(self.get_current_signal(pos_sym))
                    pos_value = float(pos.get('market_value', 0.0))
                    candidates.append((pos_sym, pos_signal, pos_value))

                # Sort candidates by weakness (low signal first) and larger market_value first to free more capital
                candidates.sort(key=lambda x: (x[1], -x[2]))

                freed = 0.0
                to_close = []
                for pos_sym, pos_signal, pos_value in candidates:
                    if freed >= needed:
                        break
                    # Don't close the same symbol we're trying to open
                    if pos_sym == symbol:
                        continue
                    to_close.append(pos_sym)
                    freed += pos_value

                if not to_close:
                    print(f" Could not free up enough capital for {symbol} (no suitable positions to close)")
                    continue

                print(f"- Will attempt to close positions to free ${freed:,.2f}: {to_close}")
                # Yield closures for each candidate
                for close_sym in to_close:
                    print(f"- Scheduling close of {close_sym} to free capital")
                    yield (close_sym, 0, None)

                # After yielding closures, attempt the trade again (when generator resumes, account may have updated)
                buying_power_after, _ = self.refresh_account_info()
                if self.can_take_trade(symbol, proposed_value):
                    print(f" Trade now possible for {symbol} after freeing capital")
                    yield (symbol, qty, price)
                else:
                    print(f" Still insufficient capital for {symbol} after freeing positions (freed=${freed:,.2f})")
        
        print("\n=== Signal Processing Complete ===")
        # Clear pending signals
        self.pending_signals = []

    def get_queue_snapshot(self) -> List[Tuple]:
        """Return the current pending queue as a list of tuples in priority order."""
        # Ensure sorted
        def sort_key(item):
            s_sym, s_strength, s_price, s_qty, s_source, s_ts = item
            source_priority = 1 if s_source.upper().startswith('ML') else 0
            return (source_priority, s_strength, abs(s_price * s_qty), -s_ts)

        self.pending_signals.sort(key=sort_key, reverse=True)
        return [(s, strength, price, qty, source, ts) for (s, strength, price, qty, source, ts) in self.pending_signals]
    
    def get_current_signal(self, symbol: str) -> float:
        """Get current signal strength for a symbol"""
        # This should be implemented to match your strategy's signal calculation
        # For now, returning 0 as placeholder
        return 0.0