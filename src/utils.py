# src/utils.py

def calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss):
    """
    Calculates the position size (in coins) based on risk percentage.
    Includes Safety Checks for Min Value and Precision.
    """
    # 1. Prevent Division by Zero
    risk_per_coin = abs(entry_price - stop_loss)
    if risk_per_coin == 0:
        return 0.0

    # 2. Basic Risk Math
    risk_amount = account_balance * risk_per_trade
    position_size = risk_amount / risk_per_coin

    # 3. Safety Check: MINIMUM SIZE ($10 minimum to be safe for Binance/Bybit)
    min_trade_value_usd = 10.0 
    trade_value_usd = position_size * entry_price
    
    if trade_value_usd < min_trade_value_usd:
        # Optional: Print warning only if you are debugging
        # print(f"⚠️ Position value (${trade_value_usd:.2f}) < Min (${min_trade_value_usd}). Skipping.")
        return 0.0

    # 4. Safety Check: PRECISION
    # BTC usually supports 3 decimals, but 4 is safe for internal calculation.
    # If trading altcoins (like DOGE), you might need round(..., 0) or round(..., 1).
    position_size = round(position_size, 4)

    return position_size