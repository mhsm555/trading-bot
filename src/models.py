from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone

class Trade(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str
    side: str           # "LONG" or "SHORT" (Futures) / "BUY" or "SELL" (Spot)
    size: float         # Size in Coins (e.g., 0.5 BTC)
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"  # "OPEN" or "CLOSED"
    
    # ⚠️ FIX: Python 3.13 compatible UTC timestamp
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Extra field to differentiate Spot vs Futures
    strategy: str = "FUTURES" # "SPOT" or "FUTURES"

class BotConfig(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)  # Singleton Row (ID=1)
    
    # --- 🎛️ STEERING WHEEL ---
    selected_timeframe: str = "1h"   # 1m, 5m, 15m, 1h
    selected_model: str = "xgb"      # xgb, rf, ensemble, lgbm
    trading_mode: str = "PAPER"      # PAPER, LIVE
    status: str = "STOPPED"          # RUNNING, STOPPED

    # --- 🛡️ RISK MANAGEMENT ---
    leverage: int = 5
    risk_per_trade: float = 0.01     # 1% per trade
    stop_loss_pct: float = 0.02      # 2% hard stop
    take_profit_pct: float = 0.04    # 4% target
    
    # --- 🌊 TREND BIAS (The Filter) ---
    trend_bias: str = "NEUTRAL"      # LONG_ONLY, SHORT_ONLY, NEUTRAL
    confidence_threshold: float = Field(default=0.60) # Only trade if AI is 60% sure
    
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))