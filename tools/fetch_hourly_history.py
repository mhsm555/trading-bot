import ccxt
import pandas as pd
import time
import sys
import os

# Add the project root to python path so we can import our src classes
sys.path.append(os.getcwd())
from src.features import FeatureEngineer

def fetch_and_process_hourly_history():
    # 1. SETUP: Use Binance for modern hourly data (High volume)
    exchange = ccxt.binance({
        'enableRateLimit': True, 
    })
    symbol = 'BTC/USDT' # Standard pair for modern era
    timeframe = '1h'    # <--- CHANGED TO HOURLY
    
    # 2. DEFINE TIME RANGE
    # Starting from 2020 as requested
    start_date = "2020-01-01 00:00:00"
    since = exchange.parse8601(start_date)
    
    print(f"--- 🕰️ Starting Hourly Download for {symbol} from {start_date} ---")
    
    all_ohlcv = []
    
    while True:
        try:
            # Fetch batch of candles
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1 hour (3600000 ms)
            # Binance sometimes returns duplicates if we just add 1ms, so we add timeframe duration
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + (60 * 60 * 1000) 
            
            # Progress bar
            last_date = pd.to_datetime(last_timestamp, unit='ms')
            print(f"Downloaded {len(all_ohlcv)} candles... (Currently at {last_date})")
            
            # Stop if we reached today
            if since > exchange.milliseconds():
                break
            
            # Respect rate limits (Hourly fetches require more requests)
            time.sleep(0.1) 
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5) # Wait and retry
            continue

    # 3. CONVERT TO EXCEL
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Ensure directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save RAW data
    raw_filename = 'data/btc_hourly_2020_2026_raw.xlsx'
    print(f"\n💾 Saving raw data to {raw_filename}...")
    df.to_excel(raw_filename, index=False)
    
    # 4. VERIFY & PROCESS (Add Indicators)
    print("--- ⚙️ Adding Indicators & Targets ---")
    
    # Initialize your custom engineer
    engineer = FeatureEngineer(df)
    
    # Add Indicators
    df_features = engineer.add_indicators()
    
    # Add Target 
    # CRITICAL CHANGE: Hourly moves are smaller than Daily moves.
    # Daily Threshold was 0.01 (1%). 
    # Hourly Threshold should be around 0.0025 (0.25%) or 0.003 (0.3%).
    df_features = engineer.add_target(threshold=0.0025) 
    
    # 5. CHECK DATA
    print("\n--- DATA VERIFICATION ---")
    print(f"Total Rows: {len(df_features)}")
    print("\nFirst 5 Rows (2020 Start):")
    print(df_features[['timestamp', 'close', 'RSI', 'target']].head())
    
    print("\nLast 5 Rows (Present Day):")
    print(df_features[['timestamp', 'close', 'RSI', 'target']].tail())
    
    # Save PROCESSED data
    proc_filename = 'data/btc_hourly_2020_2026_processed.xlsx'
    print(f"\n💾 Saving processed data to {proc_filename}...")
    df_features.to_excel(proc_filename, index=False)
    print("Done! You can open the Excel file to verify.")

if __name__ == "__main__":
    fetch_and_process_hourly_history()