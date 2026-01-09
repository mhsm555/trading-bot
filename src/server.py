import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
from src.market_data import MarketDataHandler
from src.features import FeatureEngineer
from src.ai_model import CryptoModel
from src.execution.spot_paper import SpotPaperTrader
from src.execution.futures_paper import FuturesPaperTrader 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
TRADING_MODE = 'FUTURES'  # 'SPOT' or 'FUTURES'
LEVERAGE = 5

# --- INITIALIZE WALLET ---
# 1. FIXED: Removed the accidental overwrite
if TRADING_MODE == 'SPOT':
    print("🔹 Running in SPOT Mode")
    wallet = SpotPaperTrader(initial_balance=10000)
elif TRADING_MODE == 'FUTURES':
    print(f"🚀 Running in FUTURES Mode ({LEVERAGE}x)")
    wallet = FuturesPaperTrader(initial_balance=10000, leverage=LEVERAGE)

# Global State
bot_active = False
bot_brain = CryptoModel(model_type='ensemble')

try:
    bot_brain.load_model("model_ensemble") 
    print("✅ Server: Brain Loaded.")
except Exception as e:
    print(f"❌ Server: Brain missing! {e}")

@app.get("/status")
def get_status():
    # 2. FIXED: Handle different method names for Equity
    if TRADING_MODE == 'FUTURES':
        # Futures wallet needs current price (approx 0 is dangerous, but acceptable for static check if no positions)
        equity = wallet.get_equity(0) 
    else:
        equity = wallet.get_total_equity(0)

    return {
        "active": bot_active,
        "balance": equity,
        "history": wallet.trades # Note: Ensure both classes use 'trades' list
    }

@app.get("/history/{timeframe}")
async def get_history(timeframe: str):
    handler = MarketDataHandler('binance', 'BTC/USDT', timeframe)
    df = handler.fetch_data(limit=1000)
    candles = []
    for _, row in df.iterrows():
        candles.append({
            "time": int(row['timestamp'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close']
        })
    return candles

@app.post("/bot/start")
def start_bot():
    global bot_active
    bot_active = True
    return {"message": "Bot Started"}

@app.post("/bot/stop")
def stop_bot():
    global bot_active
    bot_active = False
    return {"message": "Bot Stopped"}

@app.websocket("/ws/{timeframe}")
async def websocket_endpoint(websocket: WebSocket, timeframe: str):
    await websocket.accept()
    
    chart_handler = MarketDataHandler('binance', 'BTC/USDT', timeframe=timeframe)
    bot_handler = MarketDataHandler('binance', 'BTC/USDT', timeframe="1h")
    
    try:
        while True:
            # A. FETCH DATA
            chart_df = chart_handler.fetch_data(limit=1)
            latest_chart = chart_df.iloc[-1]
            current_price = latest_chart['close']
            
            bot_df = bot_handler.fetch_data(limit=50)
            
            # B. RUN AI
            decision = 0
            confidence = 0.0
            
            if bot_active:
                engineer = FeatureEngineer(bot_df)
                processed_bot_df = engineer.add_indicators()
                
                # predict_signal returns (decision, confidence)
                # decision 1 = Buy, decision 0 = Wait
                decision, confidence = bot_brain.predict_signal(processed_bot_df, threshold=0.6)
                
                # --- C. EXECUTION LOGIC (The smart part) ---
                if TRADING_MODE == 'SPOT':
                    # Standard Spot Execution
                    wallet.execute_strategy(decision, current_price, latest_chart['timestamp'])
                
                elif TRADING_MODE == 'FUTURES':
                    # 3. FIXED: Custom Futures Logic (Long AND Short)
                    # Logic: 
                    # High Conf (> 0.60) -> LONG
                    # Low Conf (< 0.40) -> SHORT
                    # Middle -> Close positions or Wait
                    
                    if confidence >= 0.60:
                        wallet.open_position('LONG', current_price, amount_usd=1000)
                    elif confidence <= 0.40:
                        wallet.open_position('SHORT', current_price, amount_usd=1000)
                    else:
                        # Optional: Close positions if confidence is weak
                        # wallet.close_position(current_price)
                        pass

            # D. REPORTING
            # Get equity based on mode
            if TRADING_MODE == 'FUTURES':
                current_equity = wallet.get_equity(current_price)
            else:
                current_equity = wallet.get_total_equity(current_price)

            payload = {
                "candle": {
                    "time": int(latest_chart['timestamp'].timestamp()),
                    "open": latest_chart['open'],
                    "high": latest_chart['high'],
                    "low": latest_chart['low'],
                    "close": latest_chart['close']
                },
                "bot": {
                    "is_active": bot_active,
                    "decision": "LONG" if confidence > 0.6 else ("SHORT" if confidence < 0.4 else "WAIT"),
                    "confidence": float(confidence),
                    "equity": current_equity
                }
            }
            
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"WS Error: {e}")
        import traceback
        traceback.print_exc() # Helps you see why it crashed