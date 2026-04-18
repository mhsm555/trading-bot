import pandas as pd
import os
import sys

# Ensure we can find 'src'
sys.path.append(os.getcwd())
try:
    from src.ai_model import CryptoModel
except ImportError:
    print("❌ Critical Error: Could not import src.ai_model.")
    sys.exit(1)

# --- CONFIGURATION ---
TARGETS = ['1h', '15m', '5m', '1m'] 

# 📅 THE WALL: The exact date where training MUST stop.
# Matches your ML Suite for fair comparison.
CUTOFF_DATE = '2024-01-01'

def train_dl_suite():
    print("🚀 STARTING DEEP LEARNING (LSTM) TRAINING SUITE...")
    print(f"   🎯 Timeframes: {TARGETS}")
    print(f"   📅 Training Cutoff: {CUTOFF_DATE}")

    for tf in TARGETS:
        # 1. Dynamic File Path
        data_file = f'data/btc_{tf}_processed.csv'
        
        if not os.path.exists(data_file):
            print(f"\n⚠️  Skipping {tf}: File {data_file} not found.")
            continue

        print(f"\n" + "="*60)
        print(f"🧠 TRAINING LSTM: {tf.upper()}")
        print(f"   📂 File: {data_file}")

        try:
            # 2. Load Data (CSV)
            df = pd.read_csv(data_file)
            
            # 3. Parse Dates (Crucial)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            else:
                print("   ❌ Error: 'timestamp' column missing.")
                continue

            # 4. THE DATE SPLIT (No Cheating!)
            # We strictly take data BEFORE the cutoff
            train_df = df[df['timestamp'] < CUTOFF_DATE].copy()
            
            if train_df.empty:
                print(f"   ❌ Error: No training data before {CUTOFF_DATE}.")
                continue

            print(f"   📚 Training on: {len(train_df):,} rows")
            print(f"   🕒 Data Range:  {train_df['timestamp'].iloc[0]} -> {train_df['timestamp'].iloc[-1]}")
            
            # 5. Train LSTM
            bot = CryptoModel(model_type='lstm')
            
            print("   ⏳ Training Neural Network... (This may take a moment)")
            bot.train(train_df)
            
            # 6. Save
            filename = f"model_lstm_{tf}"
            bot.save_model(filename)
            print(f"   ✅ Saved Brain: {filename}")

        except Exception as e:
            print(f"   ❌ Error training LSTM on {tf}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 DL SUITE COMPLETE. You now have LSTM brains for all timeframes.")

if __name__ == "__main__":
    train_dl_suite()