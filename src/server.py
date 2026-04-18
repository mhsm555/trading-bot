from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# --- IMPORTS ---
from src.engine import TradingEngine 
from src.market_data import MarketDataHandler
from src.config_manager import get_config, update_config # <--- NEW: Link to DB
from src.models import BotConfig 
from sqlmodel import select
from src.models import Trade # Import the Trade model
from src.database import get_session
# -----------------
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- GLOBAL ENGINE INSTANCE ---
# We create ONE engine for the whole app
bot_engine = TradingEngine(mode="FUTURES", leverage=5)

# --- WEBSOCKET MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try: await connection.send_json(message)
            except: pass

manager = ConnectionManager()

# --- BACKGROUND LISTENER ---
async def queue_reader():
    """Reads data from Engine -> Sends to React"""
    while True:
        if not bot_engine.message_queue.empty():
            data = bot_engine.message_queue.get()
            
            # --- FIX: DYNAMIC TIMEFRAME ---
            # Instead of hardcoding '1h', we check what the user selected in DB
            config = get_config()
            current_tf = config.selected_timeframe if config else '1h'
            
            # Fetch Chart Data for visualization
            try:
                handler = MarketDataHandler('binance', 'BTC/USDT', current_tf)
                df = handler.fetch_data(limit=1)
                latest = df.iloc[-1]
                
                full_payload = {
                    "candle": {
                        "time": int(latest['timestamp'].timestamp()),
                        "open": latest['open'], "high": latest['high'],
                        "low": latest['low'], "close": latest['close']
                    },
                    "bot": data
                }
                await manager.broadcast(full_payload)
            except: pass
            
        await asyncio.sleep(0.5)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_reader())

# --- NEW API ENDPOINTS (The Control Panel) ---

@app.get("/config")
def read_config():
    """Frontend asks: 'What are the current settings?'"""
    return get_config()

@app.post("/config")
async def write_config(request: Request):
    """Frontend says: 'Change the settings!'"""
    data = await request.json()
    updated_config = update_config(data)
    return updated_config

# --- EXISTING ROUTES ---

@app.get("/history/{timeframe}")
def get_history(timeframe: str):
    handler = MarketDataHandler('binance', 'BTC/USDT', timeframe)
    df = handler.fetch_data(limit=500)
    
    chart_data = []
    for _, row in df.iterrows():
        chart_data.append({
            "time": int(row['timestamp'].timestamp()),
            "open": row['open'],
            "high": row['high'],
            "low": row['low'],
            "close": row['close']
        })
    return chart_data

@app.post("/bot/start")
def start_bot():
    # Update DB status so it remembers across restarts
    update_config({"status": "RUNNING"})
    bot_engine.start()
    return {"status": "started"}

@app.post("/bot/stop")
def stop_bot():
    update_config({"status": "STOPPED"})
    bot_engine.stop()
    return {"status": "stopped"}

@app.get("/trades")
def get_trade_history():
    """Fetches all executed trades from the Database."""
    with get_session() as session:
        # Get trades, sorted by newest first
        statement = select(Trade).order_by(Trade.timestamp.desc())
        results = session.exec(statement).all()
        return results

@app.websocket("/ws/{timeframe}")
async def websocket_endpoint(websocket: WebSocket, timeframe: str):
    await manager.connect(websocket)
    try:
        while True: await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)