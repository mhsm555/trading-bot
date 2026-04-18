import sys
import os

# --- FIX: Add Project Root to Path ---
# Get the folder where this script lives (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root (TradingBot_BTC/)
project_root = os.path.dirname(current_dir)
# Add it to Python's search path
sys.path.append(project_root)
# -------------------------------------

# Now this import will work!
from src.execution.futures_paper import FuturesPaperTrader

# --- CONFIG ---
TEST_WALLET = "test_wallet_temp.json"
START_BALANCE = 10000
LEVERAGE = 10

def run_manual_test():
    print(f"\n🧪 STARTING PAPER TRADER TEST")
    print("========================================")

    # 1. CLEANUP (Delete old test wallet if exists)
    wallet_path = f"assets/{TEST_WALLET}"
    if os.path.exists(wallet_path):
        os.remove(wallet_path)
        print("🧹 Cleaned up old test wallet.")

    # 2. INITIALIZE
    # We use a temporary wallet file so we don't mess up your real demo account
    trader = FuturesPaperTrader(initial_balance=START_BALANCE, leverage=LEVERAGE, wallet_file=TEST_WALLET)
    print(f"✅ Trader Initialized. Balance: ${trader.get_total_equity(0):,.2f}")

    # 3. OPEN LONG POSITION
    # Scenario: Price is $50,000. We want to buy 0.1 BTC ($5,000 value).
    # Since leverage is 10x, this should cost us $500 margin.
    entry_price = 50000.0
    position_size = 0.1 # BTC
    timestamp = "2024-01-01 12:00:00"

    print(f"\n--- 🚀 ACTION: OPEN LONG at ${entry_price:,.2f} ---")
    # Decision 1 = LONG
    trader.execute_strategy(decision=1, current_price=entry_price, timestamp=timestamp, size=position_size)

    # CHECK: Did it work?
    positions = trader.get_open_positions()
    if not positions:
        print("❌ FAILED: No position opened!")
        return
    
    pos = positions[0]
    print(f"   👉 Position: {pos['side']} {pos['size']} BTC")
    print(f"   👉 Margin Used: ${pos['collateral']:.2f} (Should be ~$500)")
    print(f"   👉 Cash Balance: ${trader.state['usd_balance']:.2f}")

    # 4. SIMULATE PRICE INCREASE (PROFIT)
    # Price goes up to $51,000 (+2%)
    # Profit calculation: ($51,000 - $50,000) * 0.1 BTC = $100 Profit
    new_price = 51000.0
    
    print(f"\n--- 📈 SCENARIO: Price Pumps to ${new_price:,.2f} ---")
    # We call get_total_equity to force the PnL update
    equity = trader.get_total_equity(new_price)
    
    # Read the live PnL from the position state
    live_pnl = trader.positions['BTC']['unrealizedProfit']
    print(f"   👉 Unrealized PnL: ${live_pnl:.2f} (Expected: $100.00)")
    print(f"   👉 Total Equity:   ${equity:.2f} (Expected: $10,100.00)")

    if abs(live_pnl - 100.0) < 1.0:
        print("   ✅ MATH CHECK PASSED!")
    else:
        print("   ❌ MATH CHECK FAILED!")

    # 5. TEST STOP LOSS (LOGIC CHECK)
    # We just run check_risk_management manually to see if it triggers
    # Our SL is set to 2% (in trader.py). Entry 50k -> SL 49k.
    # Let's crash the price to 48k.
    crash_price = 48000.0
    print(f"\n--- 🛑 SCENARIO: Price Crashes to ${crash_price:,.2f} ---")
    print("   Checking Risk Management...")
    
    # This should trigger the close
    trader.check_risk_management(crash_price, "2024-01-01 13:00:00")
    
    # CHECK: Position should be gone
    if trader.get_open_positions():
        print("   ❌ FAILED: Stop Loss didn't trigger!")
    else:
        print("   ✅ SUCCESS: Stop Loss triggered and closed position.")
        
        # Check final balance logic
        # Loss should be roughly: (49,000 (SL Price) - 50,000) * 0.1 = -$100
        # (Note: In paper trading simulation, it might close at 48k or 49k depending on logic, 
        # but your code sets exit at current_price in check_risk_management).
        final_equity = trader.get_total_equity(0)
        print(f"   👉 Final Wallet Balance: ${final_equity:.2f}")

    print("\n========================================")
    print("🏁 TEST COMPLETE")

    # Cleanup
    if os.path.exists(wallet_path):
        os.remove(wallet_path)

if __name__ == "__main__":
    run_manual_test()