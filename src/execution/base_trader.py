import json
import os
from datetime import datetime

class BaseTrader:
    def __init__(self, initial_balance=10000, wallet_file='wallet.json'):
        self.wallet_file = wallet_file
        self.initial_balance = initial_balance
        self.state = self._load_wallet()
        
        # Shared access to history
        self.trades = self.state.get('history', [])

    def _load_wallet(self):
        """Loads the wallet file or creates a new one if missing."""
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: {self.wallet_file} was corrupted. Resetting.")
                
        # Return default structure (Child classes can extend this)
        return {
            "balance": self.initial_balance,
            "history": [],
            "positions": {}
        }

    def _save_wallet(self):
        """Saves current state to JSON."""
        with open(self.wallet_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def log_trade(self, trade_info):
        """Standardized logging for all trades."""
        # Add timestamp if missing
        if 'time' not in trade_info:
            trade_info['time'] = str(datetime.now())
            
        self.trades.append(trade_info)
        self.state['history'] = self.trades
        self._save_wallet()