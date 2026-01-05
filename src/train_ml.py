import pandas as pd
import os
import sys
from src.ai_model import CryptoModel

# --- CONFIGURATION ---
DATA_FILE = 'data/btc_hourly_2020_2026_processed.xlsx'
# 🛑 CRITICAL: We stop training here so we can test the future in backtest.py
TRAIN_CUTOFF_DATE = '2023-01-01' 

def train_specific_model(model_type):
    """
    Trains a single model type and saves it.
    model_type options: 'xgb', 'rf', 'lgbm', 'ensemble'
    """
    print(f"\n========================================")
    print(f"🏗️  TRAINING MODEL: {model_type.upper()}")
    print(f"========================================")

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    df = pd.read_excel(DATA_FILE)
    
    # 2. Verify Data
    if 'target' not in df.columns:
        print("❌ Error: 'target' column missing.")
        return

    # 3. ✂️ SPLIT DATA (Prevent Cheating)
    # We only train on the past, leaving 2023+ for the backtest
    train_df = df[df['timestamp'] < TRAIN_CUTOFF_DATE].copy()
    
    print(f"📅 Total Data Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"📚 Training Range:   {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"📉 Rows used: {len(train_df)} (Reserved {len(df) - len(train_df)} for backtesting)")

    # 4. Initialize & Train
    bot = CryptoModel(model_type=model_type)
    bot.train(train_df)

    # 5. Save
    filename = f"model_{model_type}"
    bot.save_model(filename)
    print(f"✅ Success! Saved {filename}.pkl (Trained only on Pre-2023 data)")

if __name__ == "__main__":
    # --- TRAIN ALL OF THEM ---
    for m in ['xgb', 'rf', 'lgbm', 'ensemble']:
        train_specific_model(m)