from src.execution.base_trader import BaseTrader

class FuturesPaperTrader(BaseTrader):
    def __init__(self, initial_balance=10000, leverage=10, wallet_file='futures_wallet.json'):
        # 1. Initialize Parent (Loads JSON, sets up 'self.trades')
        super().__init__(initial_balance, wallet_file)

        self.leverage = leverage
        self.fees = {'maker': 0.0002, 'taker': 0.0004} # Binance Futures Fees

        # 2. Set default state if new wallet
        if 'positions' not in self.state:
            self.state['positions'] = {'BTC': None}
        
        # Helper shortcut (pointer to the dict inside state)
        self.positions = self.state['positions']

    def get_equity(self, current_price):
        """Calculates Net Worth = Balance + Unrealized PnL"""
        # Use state['balance'] instead of self.balance to ensure it saves
        equity = self.state['balance']
        
        pos = self.positions.get('BTC')
        if pos:
            # PnL Calculation differs for Long vs Short
            if pos['side'] == 'LONG':
                pnl = (current_price - pos['entry_price']) * pos['size']
            else: # SHORT
                pnl = (pos['entry_price'] - current_price) * pos['size']
            
            equity += pnl
        return equity

    def open_position(self, side, price, amount_usd):
        # 1. Check if we already have a position
        if self.positions.get('BTC'):
            print("⚠️ Position already open. Close it first.")
            return

        # 2. Calculate Size
        buying_power = amount_usd * self.leverage
        size_in_coins = buying_power / price
        
        # 3. Pay Fee
        fee = buying_power * self.fees['taker']
        self.state['balance'] -= fee

        # 4. Record Position
        self.positions['BTC'] = {
            'side': side,
            'entry_price': price,
            'size': size_in_coins,
            'collateral': amount_usd
        }
        
        # 5. Log & Save
        self.log_trade({
            'type': f'OPEN_{side}', 
            'price': price, 
            'size': size_in_coins, 
            'fee': fee, 
            'leverage': self.leverage,
            'equity': self.get_equity(price)
        })
        print(f"🚀 OPEN {side} {self.leverage}x at ${price:.2f}")

    def close_position(self, price):
        pos = self.positions.get('BTC')
        if not pos:
            return

        # 1. Calculate PnL
        if pos['side'] == 'LONG':
            pnl = (price - pos['entry_price']) * pos['size']
        else: # SHORT
            pnl = (pos['entry_price'] - price) * pos['size']

        # 2. Pay Fee
        position_value = pos['size'] * price
        fee = position_value * self.fees['taker']
        
        # 3. Update Balance
        self.state['balance'] += (pnl - fee)
        
        # 4. Log & Save
        self.log_trade({
            'type': f'CLOSE_{pos["side"]}', 
            'price': price, 
            'pnl': pnl, 
            'fee': fee, 
            'final_balance': self.state['balance']
        })
        print(f"💰 CLOSE {pos['side']} at ${price:.2f} | PnL: ${pnl:.2f}")
        
        # 5. Reset
        self.positions['BTC'] = None
        self._save_wallet() # Force save