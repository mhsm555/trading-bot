import pytest
from src.money_management import calculate_position_size

def test_position_sizing_logic():
    balance = 10000
    risk_per_trade = 0.02 # 2%
    entry_price = 50000
    stop_loss = 49000 # $1000 risk per coin
    
    # Expected: Risking $200 total. $200 / $1000 per coin = 0.2 BTC
    expected_size = 0.2
    
    actual_size = calculate_position_size(balance, risk_per_trade, entry_price, stop_loss)
    
    assert actual_size == expected_size