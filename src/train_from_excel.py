import pandas as pd
import numpy as np
import sys
import os

# Add project root to path so we can import src
sys.path.append(os.getcwd())
from src.ai_model import CryptoModel

def train_daily_model():
    # --- 1. LOAD DATA ---
    file_path = 'data/btc_daily_2010_2025_processed.xlsx'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Run fetch_history.py first!")
        return

    print(f"--- 📂 Loading Data from {file_path} ---")
    df = pd.read_excel(file_path)
    
    # --- 2. CLEANING (The "Janitor" Phase) ---
    print(f"Original Rows: {len(df)}")
    
    # A. Drop duplicates
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    
    # B. Replace "Infinity" with NaN (common in division errors)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # C. Drop rows with ANY empty values
    df.dropna(inplace=True)
    
    # D. Ensure Target is Integer (0 or 1), not 1.0
    df['target'] = df['target'].astype(int)
    
    print(f"Cleaned Rows:  {len(df)}")

    # --- 3. DEFINE FEATURES ---
    # We want to train on everything EXCEPT raw prices and timestamps
    # This logic automatically finds your numeric feature columns
    cols_to_exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df.columns if c not in cols_to_exclude]
    
    print(f"Training Features: {feature_cols}")
    


    # --- 4. TIME-BASED SPLIT ---
    
    # NEW LOGIC: Filter for "Modern Era" only (Post-2017)
    # 2011-2016 data is too different and confuses the AI.
    modern_df = df[df['timestamp'] >= '2018-01-01']
    
    print(f"Original History: {len(df)} rows")
    print(f"Modern History:   {len(modern_df)} rows (Using 2018+ only)")
    
    test_size = 365
    train_df = modern_df.iloc[:-test_size] 
    test_df = modern_df.iloc[-test_size:]
    
    # --- 5. TRAIN ---
    print("\n--- 🧠 Training Daily Model ---")
    
    # Initialize Model (Using XGBoost or LGBM is best for this size)
    bot = CryptoModel(model_type='lgbm')
    
    # Train
    bot.train(train_df)
    
    # --- 6. EVALUATE ON TEST SET ---
    print("\n--- 🔍 Evaluating on Unseen Data (Last Year) ---")
    
    # We manually predict to show custom stats
    # Filter test_df to just features
    actual_features = bot.features 
    print(f"Model used these {len(actual_features)} features: {actual_features}")
    
    X_test = test_df[actual_features] 
    y_test = test_df['target']
    
    predictions = bot.model.predict(X_test)
    
    # Calculate Precision manually
    true_buys = ((predictions == 1) & (y_test == 1)).sum()
    total_buys = (predictions == 1).sum()
    
    if total_buys > 0:
        precision = true_buys / total_buys
        print(f"Model generated {total_buys} BUY signals.")
        print(f"Correct Buys: {true_buys}")
        print(f"PRECISION: {precision:.2%} (Target: > 50%)")
    else:
        print("Model generated 0 BUY signals (It was too scared!)")

    # --- 7. SAVE ---
    # Save as a distinct "daily" model so we don't overwrite our hourly one
    bot.save_model("model_btc_daily.pkl")

if __name__ == "__main__":
    train_daily_model()