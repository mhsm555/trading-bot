import os
import json
from datetime import datetime, timezone
from src.execution.base_trader import BaseTrader
from src.database import get_session, init_db
from src.models import Trade

class FuturesPaperTrader(BaseTrader):
    def __init__(self, initial_balance=10000, leverage=10, wallet_file='futures_wallet.json'):
        super().__init__(initial_balance, wallet_file)
        self.leverage = leverage
        self.fees = {'maker': 0.0002, 'taker': 0.0004} 
        
        # --- ⚙️ RISK SETTINGS (Matches Backtest) ---
        self.stop_loss_pct = 0.02    # 2% SL
        self.take_profit_pct = 0.04  # 4% TP
        
        init_db()

        if 'positions' not in self.state:
            self.state['positions'] = {'BTC': None}
        
        self.positions = self.state['positions']

    def get_open_positions(self):
        """Returns active position for the HUD."""
        pos = self.positions.get('BTC')
        if pos and float(pos.get('size', 0)) != 0:
            return [pos]
        return []

    def get_total_equity(self, current_price):
        """Calculates Equity = Cash + Unrealized PnL."""
        equity = self.state['usd_balance']
        pos = self.positions.get('BTC')
        
        if pos:
            entry = float(pos['entry_price'])
            size = float(pos['size'])
            
            if pos['side'] == 'LONG':
                pnl = (current_price - entry) * size
            else: # SHORT
                pnl = (entry - current_price) * size
            
            pos['unrealizedProfit'] = pnl
            equity += pnl
            
        return equity

    def check_risk_management(self, current_price, timestamp):
        """
        ⚡ NEW: Checks if Stop Loss or Take Profit was hit.
        Runs automatically before every strategy step.
        """
        pos = self.positions.get('BTC')
        if not pos: return

        sl_price = pos.get('sl_price')
        tp_price = pos.get('tp_price')
        side = pos['side']
        
        # --- LONG RISK CHECK ---
        if side == 'LONG':
            if sl_price and current_price <= sl_price:
                print(f"🛑 STOP LOSS HIT (LONG) at ${current_price:.2f}")
                self.close_position(current_price, timestamp, reason="STOP_LOSS")
            elif tp_price and current_price >= tp_price:
                print(f"🎯 TAKE PROFIT HIT (LONG) at ${current_price:.2f}")
                self.close_position(current_price, timestamp, reason="TAKE_PROFIT")

        # --- SHORT RISK CHECK ---
        elif side == 'SHORT':
            if sl_price and current_price >= sl_price:
                print(f"🛑 STOP LOSS HIT (SHORT) at ${current_price:.2f}")
                self.close_position(current_price, timestamp, reason="STOP_LOSS")
            elif tp_price and current_price <= tp_price:
                print(f"🎯 TAKE PROFIT HIT (SHORT) at ${current_price:.2f}")
                self.close_position(current_price, timestamp, reason="TAKE_PROFIT")

    def execute_strategy(self, decision, current_price, timestamp, size):
        """
        Executes orders based on signal.
        decision: 1 (LONG), -1 (SHORT), 0 (EXIT)
        """
        # 1. First, check if we got stopped out since last check
        self.check_risk_management(current_price, timestamp)
        
        current_pos = self.positions.get('BTC')

        # --- SIGNAL: OPEN LONG (1) ---
        if decision == 1:
            # If we are Short, Close it first (Flip)
            if current_pos and current_pos['side'] == 'SHORT':
                self.close_position(current_price, timestamp, reason="FLIP_TO_LONG")
                current_pos = None

            if current_pos is None:
                self._enter_trade('LONG', current_price, size, timestamp)

        # --- SIGNAL: OPEN SHORT (-1) ---
        elif decision == -1:
            # If we are Long, Close it first (Flip)
            if current_pos and current_pos['side'] == 'LONG':
                self.close_position(current_price, timestamp, reason="FLIP_TO_SHORT")
                current_pos = None

            if current_pos is None:
                self._enter_trade('SHORT', current_price, size, timestamp)

        # --- SIGNAL: EXIT (0) ---
        elif decision == 0:
            if current_pos is not None:
                self.close_position(current_price, timestamp, reason="SIGNAL_EXIT")

        # Update equity & save
        self.get_total_equity(current_price)
        self._save_wallet()

    def _enter_trade(self, side, price, size_usd, timestamp):
        """Helper to handle margin checks and entry logic."""
        required_margin = (size_usd * price) / self.leverage
        
        if required_margin > self.state['usd_balance']:
            required_margin = self.state['usd_balance'] * 0.98 
        
        self.open_position(side, price, required_margin, timestamp)

    def open_position(self, side, price, amount_usd, timestamp):
        buying_power = amount_usd * self.leverage
        size_in_coins = buying_power / price
        
        # Deduct Fee
        fee = buying_power * self.fees['taker']
        
        # 🛑 FIX: Deduct Collateral (Margin) AND Fee from Balance
        self.state['usd_balance'] -= (amount_usd + fee)

        # --- 📐 CALCULATE SL/TP PRICES ---
        if side == 'LONG':
            sl_price = price * (1 - self.stop_loss_pct)
            tp_price = price * (1 + self.take_profit_pct)
        else: # SHORT
            sl_price = price * (1 + self.stop_loss_pct)
            tp_price = price * (1 - self.take_profit_pct)

        self.positions['BTC'] = {
            'symbol': 'BTC/USDT',
            'side': side,
            'entry_price': price,
            'size': size_in_coins,
            'collateral': amount_usd,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'unrealizedProfit': 0.0
        }
        
        self._log_trade(f"OPEN_{side}", price, size_in_coins, timestamp, -fee)
        print(f"🚀 OPEN {side} {self.leverage}x at ${price:.2f} | SL: ${sl_price:.2f}")

    def close_position(self, price, timestamp, reason="signal"):
        pos = self.positions.get('BTC')
        if not pos: return

        # Calculate PnL
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * pos['size']
        else: # SHORT
            pnl = (pos['entry_price'] - price) * pos['size']

        position_value = pos['size'] * price
        fee = position_value * self.fees['taker']
        
        # Return Money to Balance
        self.state['usd_balance'] += (pos['collateral'] + pnl - fee)
        
        self._log_trade(f"CLOSE_{pos['side']}", price, pos['size'], timestamp, pnl - fee)
        
        color = "\033[92m" if pnl > 0 else "\033[91m"
        reset = "\033[0m"
        print(f"💰 CLOSE {pos['side']} (${reason}) | PnL: {color}${pnl:.2f}{reset}")
        
        self.positions['BTC'] = None
    
    def _log_trade(self, action, price, amount, time, pnl=0):
        # 1. LEGACY JSON
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

        # 2. POSTGRES DB
        try:
            if "OPEN" in action:
                status = "OPEN"
                side = action.replace("OPEN_", "")
            else:
                status = "CLOSED"
                side = action.replace("CLOSE_", "")

            new_trade = Trade(
                symbol="BTC/USDT",
                side=side,
                size=amount,
                entry_price=price,
                pnl=pnl if status == "CLOSED" else 0.0,
                status=status,
                strategy="FUTURES",
                timestamp=datetime.now(timezone.utc)
            )

            with get_session() as session:
                session.add(new_trade)
                session.commit()

        except Exception as e:
            print(f"⚠️ DB Save Error: {e}")