import ccxt
import pandas as pd
import time
import sys
import os
from datetime import datetime, timedelta

# Add the project root to python path so we can import src
sys.path.append(os.getcwd())
try:
    from src.features import FeatureEngineer
except ImportError:
    print("⚠️ Warning: Could not import FeatureEngineer. Make sure you are in the project root.")

# --- ⚙️ CONFIGURATION ---
SYMBOL = 'BTC/USDT'
EXCHANGE_ID = 'binance'

# 🛠️ DATA COLLECTION SETTINGS
# Since we are using CSV, we can fetch MUCH more data without crashing.
TIMEFRAME_CONFIG = {
    '1m':  {'start': '2021-01-01 00:00:00', 'threshold': 0.0005}, # 4+ Years of 1m data (~2M rows)
    '5m':  {'start': '2020-01-01 00:00:00', 'threshold': 0.0010}, # 5+ Years of 5m data
    '15m': {'start': '2019-01-01 00:00:00', 'threshold': 0.0015}, # 6+ Years
    '1h':  {'start': '2018-01-01 00:00:00', 'threshold': 0.0025}, # 7+ Years
}

def fetch_and_process(timeframe):
    config = TIMEFRAME_CONFIG[timeframe]
    start_date = config['start']
    threshold = config['threshold']
    
    # 1. SETUP EXCHANGE
    exchange = ccxt.binance({'enableRateLimit': True})
    since = exchange.parse8601(start_date)
    
    print(f"\n🚀 STARTING: {timeframe} Data for {SYMBOL}")
    print(f"   📅 From: {start_date}")
    print(f"   🎯 Target Threshold: {threshold*100}%")
    
    all_ohlcv = []
    
    # Calculate duration in ms for the specific timeframe
    duration_ms = exchange.parse_timeframe(timeframe) * 1000
    
    while True:
        try:
            # Fetch batch
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to avoid duplicates
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + duration_ms
            
            # Progress Log
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            print(f"   ⬇️  Fetched {len(all_ohlcv)} rows... (Reached: {last_date})", end='\r')
            
            # Stop if we passed "now"
            if since > exchange.milliseconds():
                break
            
            # Rate limit
            time.sleep(0.1) 
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            time.sleep(5)
            continue

    print(f"\n   ✅ Download Complete: {len(all_ohlcv)} candles.")

    # 3. PROCESS DATA
    if len(all_ohlcv) == 0:
        print("   ⚠️ No data found!")
        return

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Ensure output dir exists
    os.makedirs('data', exist_ok=True)

    # 4. ADD INDICATORS
    print("   ⚙️  Engineering Features...")
    try:
        engineer = FeatureEngineer(df)
        df_features = engineer.add_indicators()
        df_features = engineer.add_target(threshold=threshold)
        
        # 5. SAVE TO CSV (Unlimited Rows)
        filename = f'data/btc_{timeframe}_processed.csv'
        print(f"   💾 Saving to {filename}...")
        
        # 🚀 NO TRUNCATION NEEDED FOR CSV
        # We save the full dataset, even if it is 5 million rows.
        df_features.to_csv(filename, index=False)
        
        print(f"   ✨ Done! Saved {len(df_features)} rows.")
        
    except Exception as e:
        print(f"   ❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Loop through all timeframes
    targets = ['1h', '15m', '5m', '1m'] 
    
    for tf in targets:
        fetch_and_process(tf)
        print("-" * 50)