import json
import os
from datetime import datetime

class PaperTrader:
    def __init__(self, starting_balance=10000, wallet_file='wallet.json'):
        self.wallet_file = wallet_file
        self.starting_balance = starting_balance
        
        # Load existing wallet or create a new one
        self.state = self._load_wallet()

    def _load_wallet(self):
        if os.path.exists(self.wallet_file):
            with open(self.wallet_file, 'r') as f:
                return json.load(f)
        else:
            # Fresh Start
            return {
                "usd_balance": self.starting_balance,
                "btc_balance": 0.0,
                "history": [],
                "in_position": False,
                "entry_price": 0.0
            }

    def _save_wallet(self):
        with open(self.wallet_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def execute_strategy(self, decision, current_price, timestamp):
        """
        decision: 1 (Buy Signal) or 0 (Wait/Sell Signal)
        """
        usd = self.state['usd_balance']
        btc = self.state['btc_balance']
        
        # LOGIC:
        # If Signal is BUY (1) AND we have USD -> BUY BTC
        # If Signal is WAIT (0) AND we have BTC -> SELL BTC (Take Profit/Stop Loss)
        
        # --- BUY LOGIC ---
        if decision == 1 and not self.state['in_position']:
            # Buy as much as possible
            amount_to_buy = usd / current_price
            
            # Fee (0.1% simulation)
            fee = usd * 0.001
            amount_to_buy = (usd - fee) / current_price
            
            self.state['btc_balance'] = amount_to_buy
            self.state['usd_balance'] = 0.0
            self.state['in_position'] = True
            self.state['entry_price'] = current_price
            
            self._log_trade("BUY", current_price, amount_to_buy, timestamp)
            print(f"💰 PAPER TRADE: BOUGHT {amount_to_buy:.5f} BTC at ${current_price}")
            
        # --- SELL LOGIC ---
        elif decision == 0 and self.state['in_position']:
            # We treat 'Wait' as an exit signal (The trend is over)
            
            # Calculate value
            gross_value = btc * current_price
            fee = gross_value * 0.001 # 0.1% fee
            net_usd = gross_value - fee
            
            # Calculate Profit/Loss
            pnl = net_usd - (btc * self.state['entry_price'])
            
            self.state['usd_balance'] = net_usd
            self.state['btc_balance'] = 0.0
            self.state['in_position'] = False
            self.state['entry_price'] = 0.0
            
            self._log_trade("SELL", current_price, btc, timestamp, pnl)
            
            color = "🟢" if pnl > 0 else "🔴"
            print(f"{color} PAPER TRADE: SOLD at ${current_price}. PnL: ${pnl:.2f}")

        else:
            # Hold
            if self.state['in_position']:
                print(f"✋ HOLDING BTC (Entry: ${self.state['entry_price']})")
            else:
                print(f"💤 WAITING in USD (Balance: ${usd:.2f})")
                
        self._save_wallet()

    def _log_trade(self, action, price, amount, time, pnl=0):
        record = {
            "time": str(time),
            "action": action,
            "price": price,
            "amount": amount,
            "pnl": pnl,
            "total_equity": self.get_total_equity(price)
        }
        self.state['history'].append(record)

    def get_total_equity(self, current_price):
        usd = self.state['usd_balance']
        btc_value = self.state['btc_balance'] * current_price
        return usd + btc_value