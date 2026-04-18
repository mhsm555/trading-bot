import json
import os
from datetime import datetime, timezone
from src.execution.base_trader import BaseTrader
from src.database import get_session, init_db
from src.models import Trade

class SpotPaperTrader(BaseTrader):
    def __init__(self, initial_balance=10000, wallet_file='spot_wallet.json'):
        # Initialize the Parent
        super().__init__(initial_balance, wallet_file)
        
        # Initialize DB
        init_db()
        
        # Set specific Spot variables
        if 'btc_balance' not in self.state:
            self.state['btc_balance'] = 0.0
            self.state['usd_balance'] = initial_balance
            self.state['in_position'] = False
            self.state['entry_price'] = 0.0

    def get_total_equity(self, current_price):
        usd = self.state['usd_balance']
        btc_value = self.state['btc_balance'] * current_price
        return usd + btc_value

    # ⚡ NEW: VISUALIZE SPOT TRADES IN HUD
    def get_open_positions(self):
        """
        Constructs a 'Virtual Position' so the HUD can display 
        your Spot holding just like a Futures position.
        """
        if self.state['in_position'] and self.state['btc_balance'] > 0:
            # We don't have real-time price here, so PnL is calculated 
            # by the Engine loop when it calls this, or estimated here.
            # For simplicity, we return the static data.
            return [{
                'symbol': 'BTC/USDT',
                'side': 'SPOT_LONG',
                'entry_price': self.state['entry_price'],
                'size': self.state['btc_balance'],
                # PnL will be calculated dynamically by the Engine if needed, 
                # or we can leave it 0 here.
                'unrealizedProfit': 0.0 
            }]
        return []

    def execute_strategy(self, decision, current_price, timestamp, size):
        usd = self.state['usd_balance']
        btc = self.state['btc_balance']
        
        # --- BUY LOGIC ---
        if decision == 1 and not self.state['in_position']:
            cost_usd = size * current_price
            
            # Check funds
            if cost_usd > usd:
                print(f"⚠️ Insufficient funds. Adjusting size...")
                size = (usd * 0.99) / current_price 
                cost_usd = size * current_price

            fee = cost_usd * 0.001 # 0.1% Spot Fee
            
            self.state['btc_balance'] = size
            self.state['usd_balance'] -= (cost_usd + fee)
            self.state['in_position'] = True
            self.state['entry_price'] = current_price
            
            self._log_trade("BUY", current_price, size, timestamp, -fee)
            print(f"💰 SPOT BUY: {size:.4f} BTC at ${current_price:.2f}")
            
        # --- SELL LOGIC ---
        elif decision == 0 and self.state['in_position']:
            gross_value = btc * current_price
            fee = gross_value * 0.001
            net_usd = gross_value - fee
            
            # PnL calc
            cost_basis = btc * self.state['entry_price']
            pnl = net_usd - cost_basis
            
            self.state['usd_balance'] += net_usd
            self.state['btc_balance'] = 0.0
            self.state['in_position'] = False
            self.state['entry_price'] = 0.0
            
            self._log_trade("SELL", current_price, btc, timestamp, pnl)
            
            color = "\033[92m" if pnl > 0 else "\033[91m"
            reset = "\033[0m"
            print(f"{color}💰 SPOT SELL at ${current_price:.2f} | PnL: ${pnl:.2f}{reset}")

        self._save_wallet()

    def _log_trade(self, action, price, amount, time, pnl=0):
        # 1. JSON History
        record = {
            "time": str(time),
            "action": action,
            "price": price,
            "amount": amount,
            "pnl": pnl,
            "total_equity": self.get_total_equity(price)
        }
        if 'history' not in self.state: self.state['history'] = []
        self.state['history'].append(record)

        # 2. ⚡ DATABASE SYNC (Matches Futures Trader)
        try:
            status = "CLOSED" if action == "SELL" else "OPEN"
            side = "LONG" # Spot is always Long

            new_trade = Trade(
                symbol="BTC/USDT",
                side=side,
                size=amount,
                entry_price=price,
                pnl=pnl if status == "CLOSED" else 0.0,
                status=status,
                strategy="SPOT",
                timestamp=datetime.now(timezone.utc)
            )

            with get_session() as session:
                session.add(new_trade)
                session.commit()
                
        except Exception as e:
            print(f"⚠️ DB Save Error: {e}")