import pandas as pd
import numpy as np
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure we can find 'src'
sys.path.append(os.getcwd())

from src.rl.hybrid_env import HybridTradingEnv
from src.ai_model import CryptoModel

# --- CONFIGURATION ---
TIMEFRAME = '5m'
CUTOFF_DATE = '2024-01-01'  # 🛑 STOP Training here. Save the rest for Backtesting.
TOTAL_TIMESTEPS = 100_000

def train_hybrid_agent():
    print(f"🚀 STARTING HYBRID RL TRAINING ({TIMEFRAME})")
    
    # --- 1. LOAD DATA ---
    file_path = f'data/btc_{TIMEFRAME}_processed.csv'
    if not os.path.exists(file_path):
        print(f"❌ Data file not found: {file_path}")
        return

    print("📂 Loading Data...")
    df = pd.read_csv(file_path)
    
    # Parse & Sort Dates
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    # --- 2. LOAD ANALYST (XGBoost) ---
    print("🧠 Loading Analyst (XGBoost) to pre-compute signals...")
    try:
        analyst = CryptoModel('xgb')
        analyst.load_model(f'model_xgb_{TIMEFRAME}')
    except Exception as e:
        print(f"❌ Failed to load Analyst model: {e}")
        print("⚠️ Ensure you ran 'python src/train_ml_suite.py' first!")
        return

    # --- 3. PRE-COMPUTE SIGNALS ---
    # We run the XGBoost model ONCE over the whole dataset.
    # This is 1000x faster than running it inside the RL loop.
    print("⚡ Generating Analyst Opinions...")
    
    try:
        # Ensure we have the exact features the model expects
        feature_data = df[analyst.features]
        scaled_data = analyst.scaler.transform(feature_data)
        
        # Get probability of Class 1 (UP)
        probs = analyst.model.predict_proba(scaled_data)[:, 1]
        df['ml_signal'] = probs
        print("   ✅ Signals attached to DataFrame.")
    except KeyError as e:
        print(f"❌ Feature Mismatch: Missing column {e}")
        return
    except Exception as e:
        print(f"❌ Signal Generation Error: {e}")
        return

    # --- 4. TRAIN / VALIDATION SPLIT ---
    # ⚠️ CRITICAL: Only train on Past Data (Pre-2024)
    train_df = df[df['timestamp'] < CUTOFF_DATE].copy().reset_index(drop=True)
    
    if train_df.empty:
        print(f"❌ No training data found before {CUTOFF_DATE}!")
        return

    print(f"📚 Training on {len(train_df)} candles (2021-2023)...")

    # --- 5. INITIALIZE ENVIRONMENT ---
    # We wrap it in a Lambda + DummyVecEnv for SB3 compatibility
    env = DummyVecEnv([lambda: HybridTradingEnv(train_df, initial_balance=10000)])
    
    # --- 6. SETUP PPO AGENT ---
    # MlpPolicy = Multi-Layer Perceptron (Neural Network)
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64, 
        gamma=0.99
    )

    # --- 7. TRAIN ---
    print("🥊 Agent entering the Gym...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # --- 8. SAVE ---
    save_path = "data/agent_execution_ppo"
    model.save(save_path)
    print(f"✅ Full Stack Agent Saved to: {save_path}.zip")

if __name__ == "__main__":
    train_hybrid_agent()