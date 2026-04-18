import json
import os

class BaseTrader:
    def __init__(self, initial_balance=10000, wallet_file='wallet.json'):
        # 1. Force the file to be inside 'assets/'
        self.assets_dir = 'assets'
        os.makedirs(self.assets_dir, exist_ok=True) # Create folder if missing
        
        # 2. Update path to be inside assets
        self.wallet_file = os.path.join(self.assets_dir, wallet_file)
        
        self.starting_balance = initial_balance
        self.state = self._load_wallet()

    @property
    def trades(self):
        """
        Dynamically fetch trade history from state. 
        This prevents 'disconnected list' errors.
        """
        return self.state.get('history', [])

    def _load_wallet(self):
        """Loads the wallet file or creates a new one if missing."""
        if os.path.exists(self.wallet_file):
            try:
                with open(self.wallet_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: {self.wallet_file} was corrupted. Resetting.")
                
        # Return default structure
        return {
            "usd_balance": self.starting_balance, 
            "history": [],
            "positions": {}
        }

    def _save_wallet(self):
        """Saves current state to JSON."""
        try:
            with open(self.wallet_file, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"❌ Error saving wallet: {e}")

    # --- 🛡️ POLYMORPHISM PLACEHOLDERS (CRITICAL) ---
    # These methods allow the Engine to treat any trader (Spot or Futures) exactly the same.

    def get_total_equity(self, current_price):
        """Default equity is just the cash balance."""
        return self.state.get('usd_balance', 0.0)

    def execute_strategy(self, decision, current_price, timestamp, size):
        pass

    def _log_trade(self, *args, **kwargs):
        pass

    def get_open_positions(self):
        """
        Returns empty list by default.
        Prevents Engine crash if the HUD asks for positions 
        from a trader that doesn't use them (like a basic Spot trader).
        """
        return []