from src.database import get_session, init_db
from src.models import BotConfig
from sqlmodel import select

def get_config():
    """Reads the current settings from DB. If none, creates defaults."""
    with get_session() as session:
        config = session.get(BotConfig, 1)
        if not config:
            # Create Default
            config = BotConfig(
                id=1,
                selected_timeframe="1h",
                selected_model="xgb",
                status="STOPPED",
                trend_bias="NEUTRAL"
            )
            session.add(config)
            session.commit()
            session.refresh(config)
        return config

def update_config(updates: dict):
    """Updates specific settings (e.g., changing Trend Bias)."""
    with get_session() as session:
        config = session.get(BotConfig, 1)
        if not config:
            get_config() # Create if missing
            config = session.get(BotConfig, 1)
            
        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        session.add(config)
        session.commit()
        session.refresh(config)
        return config