from datetime import datetime
import pandas as pd
import os
from typing import Dict, List

class TradeMonitor:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.positions: Dict[str, dict] = {}  # Current positions
        self.trades: List[dict] = []  # Historical trades
        self.daily_pnl = {}  # Daily PnL tracking
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Initialize CSV files if they don't exist
        self.trades_file = os.path.join(log_dir, "trades.csv")
        self.positions_file = os.path.join(log_dir, "positions.csv")
        self._initialize_files()

    def _initialize_files(self):
        """Initialize CSV files with headers if they don't exist"""
        if not os.path.exists(self.trades_file):
            pd.DataFrame(columns=[
                'timestamp', 'symbol', 'side', 'qty', 'price',
                'pnl', 'trade_type', 'duration'
            ]).to_csv(self.trades_file, index=False)
        
        if not os.path.exists(self.positions_file):
            pd.DataFrame(columns=[
                'timestamp', 'symbol', 'qty', 'entry_price',
                'current_price', 'unrealized_pnl'
            ]).to_csv(self.positions_file, index=False)

    def record_trade(self, symbol: str, side: str, qty: int, price: float, 
                    trade_type: str = "entry"):
        """Record a new trade"""
        timestamp = datetime.now()
        
        # Calculate PnL if it's an exit trade
        pnl = 0
        duration = None
        if trade_type in ["exit", "reversal"] and symbol in self.positions:
            pos = self.positions[symbol]
            pnl = (price - pos['entry_price']) * pos['qty'] if pos['qty'] > 0 else \
                  (pos['entry_price'] - price) * abs(pos['qty'])
            duration = (timestamp - pos['entry_time']).total_seconds() / 3600  # hours
        
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'price': price,
            'pnl': pnl,
            'trade_type': trade_type,
            'duration': duration
        }
        self.trades.append(trade)
        
        # Update positions
        if trade_type == "entry" or trade_type == "reversal":
            self.positions[symbol] = {
                'qty': qty,
                'entry_price': price,
                'entry_time': timestamp
            }
        elif trade_type == "exit":
            if symbol in self.positions:
                del self.positions[symbol]
        
        # Log to CSV
        pd.DataFrame([trade]).to_csv(self.trades_file, mode='a', header=False, index=False)
        self._update_position_log()

    def update_position_prices(self, symbol: str, current_price: float):
        """Update current price and unrealized PnL for a position"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            qty = pos['qty']
            entry_price = pos['entry_price']
            unrealized_pnl = (current_price - entry_price) * qty
            
            position_data = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'qty': qty,
                'entry_price': entry_price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl
            }
            
            # Update positions log
            pd.DataFrame([position_data]).to_csv(
                self.positions_file, mode='a', header=False, index=False
            )

    def _update_position_log(self):
        """Update the positions log file with current positions"""
        if self.positions:
            positions_data = []
            for symbol, pos in self.positions.items():
                positions_data.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'qty': pos['qty'],
                    'entry_price': pos['entry_price'],
                    'current_price': None,  # Will be updated by update_position_prices
                    'unrealized_pnl': None
                })
            pd.DataFrame(positions_data).to_csv(
                self.positions_file, mode='w', index=False
            )

    def get_position_summary(self) -> pd.DataFrame:
        """Get a summary of current positions"""
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, pos in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'qty': pos['qty'],
                'entry_price': pos['entry_price'],
                'holding_time': (datetime.now() - pos['entry_time']).total_seconds() / 3600
            })
        return pd.DataFrame(positions_data)

    def get_trade_stats(self) -> dict:
        """Calculate trading statistics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        stats = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'total_pnl': df['pnl'].sum(),
            'avg_trade_duration': df['duration'].mean(),
            'win_rate': len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0
        }
        return stats