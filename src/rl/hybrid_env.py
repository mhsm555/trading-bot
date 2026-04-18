import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class HybridTradingEnv(gym.Env):
    """
    The 'Cockpit' for the RL Agent.
    It sees: [ML_Confidence, Sentiment, RSI, Volatility, PnL]
    It controls: Buy/Sell/Hold/Close
    """
    def __init__(self, df, ml_model=None, sentiment_engine=None, initial_balance=10000):
        super(HybridTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = None # {'entry': float, 'side': str}
        self.current_step = 0
        
        # --- ACTION SPACE ---
        # 0: HOLD, 1: LONG, 2: SHORT, 3: CLOSE
        self.action_space = spaces.Discrete(4)
        
        # --- OBSERVATION SPACE ---
        # 1. ML Signal (0.0 - 1.0)
        # 2. Sentiment (-1.0 to 1.0)
        # 3. RSI (Normalized 0.0 - 1.0)
        # 4. Volatility (ATR %)
        # 5. Current PnL %
        # Shape = (5,)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Resets the simulation to the beginning."""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = None
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        
        # 1. ML Prediction
        ml_prob = row.get('ml_signal', 0.5)
        
        # 2. Sentiment (Default to 0 if missing)
        news_score = row.get('news_score', 0.0)
        
        # 3. PnL Calculation
        pnl_pct = 0.0
        if self.position:
            current_price = row['close']
            entry = self.position['entry']
            if self.position['side'] == 'LONG':
                pnl_pct = (current_price - entry) / entry
            else:
                pnl_pct = (entry - current_price) / entry

        # 4. Construct Vector (Normalized)
        obs = np.array([
            ml_prob,
            news_score,
            row.get('RSI', 50) / 100.0,   # ✅ FIX: Normalize RSI to 0-1
            row.get('ATR_pct', 0.01),     # ✅ FIX: Use Percentage, not Raw $
            pnl_pct
        ], dtype=np.float32)
        
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        terminated = False
        truncated = False
        
        # Calculate PnL for Risk Check (Fast local calc)
        pnl_pct = 0.0
        if self.position:
            entry = self.position['entry']
            if self.position['side'] == 'LONG':
                pnl_pct = (current_price - entry) / entry
            else:
                pnl_pct = (entry - current_price) / entry

        # --- 🛡️ RISK LAYER (Hard Rules) ---
        # "Kill Switch": If down 2%, Close immediately.
        if self.position and pnl_pct < -0.02:
            action = 3     # Override Agent's decision
            reward -= 0.5  # Penalty for hitting stop loss

        # --- EXECUTION LAYER ---
        if action == 3: # CLOSE
            if self.position:
                # Reward = Actual PnL % (scaled up for faster learning)
                reward += (pnl_pct * 10.0) 
                
                # Bonus for closing in profit
                if pnl_pct > 0: reward += 0.1
                
                self.position = None
        
        elif action == 1: # LONG
            if not self.position:
                self.position = {'entry': current_price, 'side': 'LONG'}
                
        elif action == 2: # SHORT
            if not self.position:
                self.position = {'entry': current_price, 'side': 'SHORT'}

        # --- NEXT STEP ---
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
        return self._get_observation(), reward, terminated, truncated, {}