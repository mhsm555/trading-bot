import time
import threading
import queue
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --- IMPORTS ---
from src.config_manager import get_config
from src.market_data import MarketDataHandler
from src.features import FeatureEngineer
from src.ai_model import CryptoModel
from src.execution.spot_paper import SpotPaperTrader
from src.execution.futures_paper import FuturesPaperTrader
from src.utils import calculate_position_size  # Ensure this exists in src/utils.py

# Optional RL Import
try:
    from stable_baselines3 import PPO
    HAS_RL = True
except ImportError:
    HAS_RL = False

class TradingEngine:
    def __init__(self, mode="FUTURES", leverage=5):
        self.mode = mode
        self.leverage = leverage
        self.running = False
        self.stop_event = threading.Event()
        self.message_queue = queue.Queue()
        
        # --- BRAIN STATE ---
        self.analyst = None  # XGBoost/LSTM (The Predictor)
        self.trader_bot = None   # PPO Agent (The Executor)
        self.current_model_signature = None 
        
        # --- INITIALIZE EXECUTION ---
        print(f"⚙️ Initializing Engine [{mode}]...")
        if mode == "SPOT":
            self.trader = SpotPaperTrader(initial_balance=10000, wallet_file='spot_wallet.json')
        else:
            self.trader = FuturesPaperTrader(initial_balance=10000, leverage=leverage, wallet_file='futures_wallet.json')

    def start(self):
        if self.running: return
        print("🟢 Bot Engine Started.")
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop)
        self.thread.start()

    def stop(self):
        print("🔴 Bot Engine Stopping...")
        self.stop_event.set()
        self.running = False
        if hasattr(self, 'thread'): self.thread.join()
        print("🛑 Bot Stopped.")

    def _reload_brains(self, config):
        """Loads both the Analyst (ML) and the Trader (RL)."""
        needed_signature = f"{config.selected_model}_{config.selected_timeframe}"
        
        if self.current_model_signature != needed_signature:
            print(f"🔄 Loading Brains for [{needed_signature}]...")
            
            # 1. Load Analyst (ML)
            try:
                new_analyst = CryptoModel(model_type=config.selected_model)
                new_analyst.load_model(f"model_{needed_signature}")
                self.analyst = new_analyst
                print(f"   ✅ Analyst Loaded (Accuracy Mode)")
            except Exception as e:
                print(f"   ❌ Analyst Load Failed: {e}")
                return

            # 2. Load Trader (RL) - Optional
            rl_path = "data/agent_execution_ppo"
            if HAS_RL and os.path.exists(f"{rl_path}.zip"):
                try:
                    self.trader_bot = PPO.load(rl_path)
                    print(f"   ✅ Trader Agent Loaded (Hybrid Execution Mode)")
                except Exception as e:
                    print(f"   ⚠️ RL Agent Error: {e}")
            else:
                self.trader_bot = None
                print("   ℹ️ No RL Agent found. Running in Standard ML Mode.")

            self.current_model_signature = needed_signature

    def _get_rl_observation(self, ml_prob, row):
        """Constructs the vector for the RL Agent."""
        # Calculate PnL %
        pnl_pct = 0.0
        positions = self.trader.get_open_positions()
        if positions:
            pos = positions[0]
            entry = float(pos['entry_price'])
            current = float(row['close'])
            if pos['side'] == 'LONG' or pos['side'] == 'SPOT_LONG':
                pnl_pct = (current - entry) / entry
            else:
                pnl_pct = (entry - current) / entry

        # Mock Sentiment (0.0) until you integrate FinBERT
        sentiment = 0.0

        # [ML_Prob, Sentiment, RSI, ATR, PnL]
        return np.array([
            ml_prob,
            sentiment,
            row.get('RSI', 50),
            row.get('ATR', 0),
            pnl_pct
        ], dtype=np.float32)

    def _run_loop(self):
        while not self.stop_event.is_set():
            try:
                config = get_config()
                self._reload_brains(config)
                self._reload_trader_if_needed(config)

                # --- STOPPED STATE HEARTBEAT ---
                if config.status != "RUNNING":
                    self._send_heartbeat(active=False)
                    time.sleep(1)
                    continue

                if not self.analyst:
                    print("⚠️ Waiting for Analyst Model...")
                    time.sleep(5)
                    continue

                # --- STEP 1: FETCH & PROCESS DATA ---
                handler = MarketDataHandler('binance', 'BTC/USDT', config.selected_timeframe)
                raw_df = handler.fetch_data(limit=500)
                
                if raw_df.empty:
                    time.sleep(2)
                    continue

                engineer = FeatureEngineer(raw_df)
                market_state = engineer.add_indicators()
                current_row = market_state.iloc[-1]
                current_price = current_row['close']
                current_time = str(current_row['timestamp'])

                # --- STEP 2: ANALYST PREDICTION ---
                # Get the raw probability from XGBoost
                # Note: predict_signal returns (decision, probability)
                _, ml_prob = self.analyst.predict_signal(market_state, threshold=0.5)

                # --- STEP 3: DECISION LOGIC (Hybrid vs Standard) ---
                final_decision = 0  # Default WAIT
                
                if self.trader_bot:
                    # 🤖 MODE A: HYBRID (RL DECISION)
                    obs = self._get_rl_observation(ml_prob, current_row)
                    action, _ = self.trader_bot.predict(obs, deterministic=True)
                    
                    # Map RL Action: 0=HOLD, 1=LONG, 2=SHORT, 3=CLOSE
                    if action == 1: final_decision = 1
                    elif action == 2: final_decision = -1
                    elif action == 3: final_decision = 0 # Signal to check exit logic
                    
                    # Force Close Logic needs special handling in execution
                    if action == 3 and self.trader.get_open_positions():
                        # We simulate a "Close" by sending the opposite signal or custom logic
                        # For simplicity here: 0 usually means "Exit" in your trader logic
                        final_decision = 0 
                    
                else:
                    # 🧠 MODE B: STANDARD (THRESHOLD RULES)
                    threshold = config.confidence_threshold
                    if ml_prob >= threshold: final_decision = 1
                    elif ml_prob <= (1 - threshold): final_decision = -1
                    else: final_decision = 0

                # --- STEP 4: APPLY TREND BIAS FILTER ---
                if config.trend_bias == "LONG_ONLY" and final_decision == -1:
                    final_decision = 0
                elif config.trend_bias == "SHORT_ONLY" and final_decision == 1:
                    final_decision = 0

                # --- STEP 5: RISK CALCULATION ---
                stop_loss_price = 0.0
                if final_decision == 1:
                    stop_loss_price = current_price * (1 - config.stop_loss_pct)
                elif final_decision == -1:
                    stop_loss_price = current_price * (1 + config.stop_loss_pct)

                size = 0.0
                if final_decision != 0:
                    size = calculate_position_size(
                        self.trader.get_total_equity(current_price),
                        config.risk_per_trade,
                        current_price,
                        stop_loss_price
                    )

                # --- STEP 6: EXECUTE ---
                # Special Case: If RL says CLOSE (Action 3), we need to ensure we are in a position
                # For this standard execute_strategy, 0 usually means "Check Exits" or "Do Nothing"
                # If you want 0 to mean FORCE CLOSE, you update execute_strategy to handle decision=99 or similar
                # For now, we rely on the trader's internal SL/TP logic + Signal flips.
                
                self.trader.execute_strategy(final_decision, current_price, current_time, size)

                # --- STEP 7: UPDATE UI ---
                self._send_heartbeat(active=True, decision=final_decision, conf=ml_prob, price=current_price)

                # Sleep based on timeframe
                sleep_s = 5 if config.selected_timeframe == '1m' else 10
                for _ in range(sleep_s):
                    if self.stop_event.is_set(): break
                    time.sleep(1)

            except Exception as e:
                print(f"❌ Engine Loop Error: {e}")
                time.sleep(5)

    def _send_heartbeat(self, active=False, decision=0, conf=0.0, price=0.0):
        """Helper to package UI data."""
        decision_map = {1: "LONG", -1: "SHORT", 0: "WAIT"}
        
        # Calculate PnL
        pnl = 0.0
        pos = None
        open_pos = self.trader.get_open_positions()
        if open_pos:
            pos = open_pos[0]
            pnl = float(pos.get('unrealizedProfit', 0))

        payload = {
            "is_active": active,
            "equity": self.trader.get_total_equity(price if price else 0),
            "decision": decision_map.get(decision, "WAIT"),
            "confidence": float(conf),
            "pnl": pnl,
            "positions": pos,
            "btc_balance": self.trader.state.get('btc_balance', 0),
            "active_model": self.current_model_signature,
            "trend_bias": get_config().trend_bias
        }
        self.message_queue.put(payload)

        # Add this inside TradingEngine class
    def _reload_trader_if_needed(self, config):
        """Switches between Spot and Futures if config changes."""
        # 1. Check if mode changed
        # We need to map DB string "PAPER" (which assumes Futures usually) or specific mode logic
        # For this logic, let's assume config.trading_mode decides the class, 
        # or you can add a new field 'market_type' to BotConfig.
        
        # Let's assume we want to support switching leverage dynamically too:
        if self.mode == "FUTURES" and self.leverage != config.leverage:
            print(f"⚙️ Updating Leverage: {self.leverage}x -> {config.leverage}x")
            self.leverage = config.leverage
            # Re-init Futures Trader with new leverage
            self.trader = FuturesPaperTrader(initial_balance=self.trader.state['usd_balance'], leverage=self.leverage)

        # (If you add a field for SPOT/FUTURES switching later, implement it here)