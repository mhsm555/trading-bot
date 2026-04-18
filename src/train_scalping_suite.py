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
TARGETS = ['1m', '5m', '15m', '1h'] 
MODEL_TYPES = ['xgb', 'rf', 'lgbm', 'ensemble']

# 📅 THE CUTOFF DATE
# Everything BEFORE this is for Learning.
# Everything AFTER this is for Testing.
CUTOFF_DATE = '2024-01-01'

def train_scalping_suite():
    print("🚀 STARTING SCALPING TRAINING SUITE")
    print(f"📅 Training Cutoff: {CUTOFF_DATE}")
    print("-" * 60)
    
    for tf in TARGETS:
        # 1. Load Data (CSV)
        data_file = f'data/btc_{tf}_processed.csv'
        
        if not os.path.exists(data_file):
            print(f"⚠️  Skipping {tf}: {data_file} not found.")
            continue
            
        print(f"\n📊 LOADING: {tf.upper()}")
        
        try:
            df = pd.read_csv(data_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            else:
                print("❌ Error: No timestamp column.")
                continue

            # --- 2. THE DATE SPLIT (CRITICAL STEP) ---
            # Instead of 80/20, we slice by Date
            train_df = df[df['timestamp'] < CUTOFF_DATE].copy()
            test_df  = df[df['timestamp'] >= CUTOFF_DATE].copy()
            
            if train_df.empty:
                print(f"❌ Error: No training data before {CUTOFF_DATE}!")
                continue
            if test_df.empty:
                print(f"⚠️ Warning: No test data after {CUTOFF_DATE} (Cannot evaluate accuracy).")
            
            print(f"   📚 Train Rows: {len(train_df):,} (Ends: {train_df['timestamp'].iloc[-1]})")
            print(f"   🧪 Test Rows:  {len(test_df):,} (Starts: {test_df['timestamp'].iloc[0] if not test_df.empty else 'N/A'})")
            
            # --- 3. Train Models ---
            for m_type in MODEL_TYPES:
                print(f"\n   ⚙️  Training {m_type.upper()}...")
                
                try:
                    bot = CryptoModel(model_type=m_type)
                    
                    # Library Check
                    if bot.model is None and m_type != 'lstm':
                        print(f"      ⚠️  Skipping {m_type}: Library missing.")
                        continue

                    # TRAIN
                    bot.train(train_df)
                    
                    # EVALUATE (Only if we have test data)
                    if not test_df.empty:
                        acc = 0.0
                        if hasattr(bot.model, 'score'):
                            # Standard ML Models
                            X_test = test_df[bot.features]
                            y_test = test_df['target']
                            # Remember: The model class handles scaling internally now?
                            # Check your src/ai_model.py. 
                            # If using the 'Corrected' version I gave you:
                            # You need to manually scale for scoring if not using predict_signal
                            # BUT bot.train() handles fitting the scaler.
                            
                            try:
                                X_test_scaled = bot.scaler.transform(X_test)
                                acc = bot.model.score(X_test_scaled, y_test)
                                print(f"      🏆 Accuracy (2024+): {acc:.2%}")
                            except:
                                pass

                    # SAVE
                    filename = f"model_{m_type}_{tf}"
                    bot.save_model(filename)
                    print(f"      💾 Saved: {filename}.pkl")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    # import traceback; traceback.print_exc()

        except Exception as e:
            print(f"   ❌ Data Error: {e}")

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE.")

if __name__ == "__main__":
    train_scalping_suite()