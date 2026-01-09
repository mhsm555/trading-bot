# src/money_management.py

def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss):
    """
    Calculates the position size (in coins) based on risk percentage.
    
    Formula:
    Risk Amount ($) = Account Balance * Risk Percentage
    Risk Per Coin ($) = |Entry Price - Stop Loss|
    Position Size = Risk Amount / Risk Per Coin
    """
    
    # 1. Calculate how much money we are willing to lose
    risk_amount = account_balance * risk_per_trade
    
    # 2. Calculate the price difference (Risk per single unit of asset)
    # We use abs() so it works for both Long and Short positions
    risk_per_coin = abs(entry_price - stop_loss)
    
    # Avoid division by zero crash
    if risk_per_coin == 0:
        return 0.0
        
    # 3. Calculate final size
    position_size = risk_amount / risk_per_coin
    
    return position_size