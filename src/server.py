import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
from src.market_data import MarketDataHandler
from src.features import FeatureEngineer
from src.ai_model import CryptoModel
from src.paper_trader import PaperTrader

app = FastAPI()

# Allow the React frontend to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... inside src/server.py ...

# Global State
bot_active = False
current_timeframe = "1h"
wallet = PaperTrader()

# Initialize Brain
bot_brain = CryptoModel(model_type='ensemble')
try:
    bot_brain.load_model("model_ensemble") # <--- Automatically looks in data/model_ensemble.pkl
    print("✅ Server: Brain Loaded.")
except Exception as e:
    print(f"❌ Server: Brain missing! {e}")

@app.get("/status")
def get_status():
    """Returns wallet balance and bot status."""
    return {
        "active": bot_active,
        "balance": wallet.get_total_equity(0), # Pass 0 if we just want raw balance
        "history": wallet.state['history']
    }

@app.get("/history/{timeframe}")
async def get_history(timeframe: str):
    """Fetches historical candles for the chart initialization."""
    handler = MarketDataHandler('binance', 'BTC/USDT', timeframe)
    df = handler.fetch_data(limit=1000)
    
    # Format for Lightweight Charts (time must be UNIX timestamp in seconds)
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

# In src/server.py

@app.websocket("/ws/{timeframe}")
async def websocket_endpoint(websocket: WebSocket, timeframe: str):
    await websocket.accept()
    
    # 1. Handler for the CHART (Dynamic)
    chart_handler = MarketDataHandler('binance', 'BTC/USDT', timeframe=timeframe)
    
    # 2. Handler for the BOT (Always 1h)
    bot_handler = MarketDataHandler('binance', 'BTC/USDT', timeframe="1h")
    
    try:
        while True:
            # --- A. FETCH CHART DATA (What you see) ---
            chart_df = chart_handler.fetch_data(limit=1)
            latest_chart = chart_df.iloc[-1]
            
            # --- B. FETCH BOT DATA (What the AI needs) ---
            # We fetch this SEPARATELY so the bot is never confused by 15m/4h data
            bot_df = bot_handler.fetch_data(limit=50)
            
            # --- C. RUN BOT LOGIC (Always on 1h data) ---
            decision = 0
            confidence = 0.0
            
            if bot_active:
                # 1. Engineer Features on the 1h data
                engineer = FeatureEngineer(bot_df)
                processed_bot_df = engineer.add_indicators()
                
                # 2. Ask the Brain
                decision, confidence = bot_brain.predict_signal(processed_bot_df, threshold=0.6)
                
                # 3. Execute Trade (using the CURRENT price, which is fine)
                # Note: We use the chart's current close price for execution simulation
                # so it matches what you see on screen.
                wallet.execute_strategy(decision, latest_chart['close'], latest_chart['timestamp'])

            # --- D. SEND PAYLOAD ---
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
                    "decision": "BUY" if decision == 1 else "WAIT",
                    "confidence": float(confidence),
                    "equity": wallet.get_total_equity(latest_chart['close'])
                }
            }
            
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"WS Error: {e}")