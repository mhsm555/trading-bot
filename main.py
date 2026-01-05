import os
import time
import pandas as pd
from dotenv import load_dotenv

# Ensure these match your folder structure
from src.market_data import MarketDataHandler
from src.features import FeatureEngineer
from src.ai_model import CryptoModel
from src.paper_trader import PaperTrader

load_dotenv()

# --- 🧠 BRAIN SELECTION ---
# Change this to: 'model_xgb', 'model_rf', 'model_lgbm', 'model_ensemble', or 'model_btc_lstm'
CURRENT_MODEL_NAME = "model_xgb" 

# Detect model type automatically from the name for initialization
if 'lstm' in CURRENT_MODEL_NAME:
    MODEL_TYPE = 'lstm'
else:
    MODEL_TYPE = CURRENT_MODEL_NAME.replace('model_', '') # e.g. 'xgb'

def get_seconds_to_next_hour():
    now = time.time()
    next_hour = (int(now / 3600) + 1) * 3600
    return (next_hour - now) + 10

def run_bot_loop():
    print(f"--- 🤖 STARTING PAPER TRADER with {CURRENT_MODEL_NAME.upper()} ---")
    
    # 1. Initialize the correct Brain
    bot_brain = CryptoModel(model_type=MODEL_TYPE)
    
    try:
        # This loads the weights (and the scaler!)
        bot_brain.load_model(CURRENT_MODEL_NAME)
        print(f"✅ Loaded {CURRENT_MODEL_NAME} successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Did you run 'python -m src.train_ml' first?")
        return

    # 2. Initialize Wallet
    paper_wallet = PaperTrader(starting_balance=10000)
    print(f"💰 Paper Wallet Equity: ${paper_wallet.get_total_equity(0):.2f}")

    # 3. Infinite Loop
    while True:
        try:
            print("\n--- ⏳ Checking Market ... ---")
            
            # A. Fetch Data
            handler = MarketDataHandler('binance', 'BTC/USDT', timeframe='1h')
            # Fetch 100 candles. 
            # Note: For LSTM we might want more context, but 100 is safe for all ML models.
            raw_df = handler.fetch_data(limit=100)
            
            # B. Engineering
            engineer = FeatureEngineer(raw_df)
            market_state = engineer.add_indicators()
            
            # C. AI Prediction
            decision, confidence = bot_brain.predict_signal(market_state, threshold=0.7)
            
            current_price = market_state.iloc[-1]['close']
            current_time = market_state.iloc[-1]['timestamp']
            
            print(f"Time: {current_time} | Price: ${current_price:.2f}")
            print(f"🧠 Model: {MODEL_TYPE.upper()} | Confidence: {confidence:.2%}")
            
            if decision == 1:
                print(f">>> 🟢 SIGNAL: BUY")
            else:
                print(f">>> 🔴 SIGNAL: WAIT")

            # D. Execute Trade
            paper_wallet.execute_strategy(decision, current_price, current_time)
            
            # E. Report
            total_equity = paper_wallet.get_total_equity(current_price)
            print(f"📊 Total Account Value: ${total_equity:.2f}")
            
            # F. Sleep
            seconds_to_wait = get_seconds_to_next_hour()
            print(f"zzZ Sleeping {seconds_to_wait/60:.1f} min...")
            time.sleep(seconds_to_wait) 
            
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped.")
            break
        except Exception as e:
            print(f"⚠️ Unexpected Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot_loop()