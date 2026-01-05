import time
import pandas as pd
from src.market_data import MarketDataHandler
from src.features import FeatureEngineer
from src.ai_model import CryptoModel

class Trader:
    def __init__(self, symbol='BTC/USDT', dry_run=True):
        self.symbol = symbol
        self.dry_run = dry_run
        self.handler = MarketDataHandler('binance', symbol, timeframe='1h')
        self.model = CryptoModel()
        self.model.load_model("production_model.pkl")
        
        # Keep track of state
        self.position = 0 # 0 = No position, 1 = Long
        self.entry_price = 0
        
        print(f"--- TRADER INITIALIZED (Dry Run: {self.dry_run}) ---")

    def execute_trade(self):
        """
        The Main Loop: Fetches data -> Predicts -> Orders
        """
        print(f"\n[{pd.Timestamp.now()}] Checking market...")
        
        # 1. GET DATA (Need enough for indicators)
        df = self.handler.fetch_data(limit=100)
        
        # 2. PREPARE FEATURES
        engineer = FeatureEngineer(df)
        current_state = engineer.add_indicators()
        latest_row = current_state.tail(1)
        
        current_price = latest_row['close'].values[0]
        timestamp = latest_row['timestamp'].values[0]
        
        # 3. ASK THE BRAIN
        signal = self.model.predict_signal(latest_row, threshold=0.65)
        
        # 4. DECISION LOGIC
        if self.position == 0:
            if signal == 1:
                self.buy(current_price, timestamp)
            else:
                print(f"WAIT: Model sees no opportunity (Price: {current_price})")
                
        elif self.position > 0:
            # Check exit conditions
            # Exit if Signal is 0 (AI says sell) OR Stop Loss hit (-2%)
            pct_change = (current_price - self.entry_price) / self.entry_price
            
            if signal == 0:
                self.sell(current_price, timestamp, reason="AI Signal")
            elif pct_change < -0.02:
                self.sell(current_price, timestamp, reason="Stop Loss")
            else:
                print(f"HOLD: Profit currently {pct_change:.2%}")

    def buy(self, price, time):
        print(f">>> 🟢 OPENING BUY ORDER @ {price}")
        if not self.dry_run:
            # REAL MONEY CODE
            # order = self.handler.exchange.create_market_buy_order(self.symbol, amount)
            pass
        else:
            self.position = 1
            self.entry_price = price
            print("    (Dry Run: Order Simulated)")

    def sell(self, price, time, reason):
        print(f">>> 🔴 CLOSING POSITION @ {price} ({reason})")
        if not self.dry_run:
            # REAL MONEY CODE
            # order = self.handler.exchange.create_market_sell_order(self.symbol, amount)
            pass
        else:
            # Calculate simulated profit
            profit = (price - self.entry_price) / self.entry_price
            print(f"    (Dry Run: Trade Closed. Profit: {profit:.2%})")
            self.position = 0
            self.entry_price = 0

if __name__ == "__main__":
    # Simple loop to test the trader
    bot = Trader(symbol='BTC/USDT', dry_run=True)
    
    # Run once to test
    bot.execute_trade()