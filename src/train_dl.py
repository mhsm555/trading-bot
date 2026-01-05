from src.ai_model import CryptoModel
import pandas as pd
import os

# --- CONFIGURATION ---
DATA_FILE = 'data/btc_hourly_2020_2026_processed.xlsx'
TRAIN_CUTOFF_DATE = '2023-01-01'  # 🛑 Stop training here so we can test the future later

def train():
    print("--- 🧠 TRAINING DEEP LEARNING MODEL (LSTM) ---")

    # 1. Load Data
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found.")
        return

    print(f"--- 📂 Loading Data from {DATA_FILE} ---")
    df = pd.read_excel(DATA_FILE)
    
    if 'target' not in df.columns:
        print("❌ Error: 'target' column missing.")
        return

    # 2. ✂️ SPLIT DATA (Crucial Step)
    # We filter specifically for training data (The Past)
    train_df = df[df['timestamp'] < TRAIN_CUTOFF_DATE].copy()
    
    print(f"📅 Total Data: {len(df)} rows")
    print(f"📉 Training on: {len(train_df)} rows (Pre-{TRAIN_CUTOFF_DATE})")
    print(f"🙈 Hidden from AI: {len(df) - len(train_df)} rows (Reserved for Backtesting)")

    # 3. Train DL (LSTM)
    bot = CryptoModel(model_type='lstm')
    
    # LSTM training takes longer, so we add a print to be patient
    print("⏳ Starting LSTM training... (This might take a minute)")
    bot.train(train_df)

    # 4. Save
    # The updated ai_model.py will automatically save this into 'data/model_btc_lstm.keras'
    bot.save_model("model_btc_lstm")
    print("🎉 LSTM Training Complete! Saved to data/ folder.")

if __name__ == "__main__":
    train()