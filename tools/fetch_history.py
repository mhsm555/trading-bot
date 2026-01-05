import ccxt
import pandas as pd
import time
import sys
import os

# Add the project root to python path so we can import our src classes
sys.path.append(os.getcwd())
from src.features import FeatureEngineer

def fetch_and_process_history():
    # 1. SETUP: Use Bitstamp for long history (Binance is too new)
    exchange = ccxt.bitstamp({
        'enableRateLimit': True, # Important to avoid bans
    })
    symbol = 'BTC/USD' # Bitstamp uses USD, not USDT for the oldest data
    timeframe = '1d'   # Daily candles
    
    # 2. DEFINE TIME RANGE
    # Bitstamp data starts roughly late 2011
    start_date = "2010-01-01 00:00:00"
    since = exchange.parse8601(start_date)
    
    print(f"--- 🕰️ Starting Download for {symbol} from {start_date} ---")
    
    all_ohlcv = []
    
    while True:
        try:
            # Fetch batch of candles
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Update 'since' to the last timestamp + 1ms to get next batch
            since = ohlcv[-1][0] + 1
            
            # Progress bar
            last_date = pd.to_datetime(ohlcv[-1][0], unit='ms')
            print(f"Downloaded {len(all_ohlcv)} candles... (Currently at {last_date.date()})")
            
            # Stop if we reached today
            if since > exchange.milliseconds():
                break
                
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
    raw_filename = 'data/btc_daily_2010_2025_raw.xlsx'
    print(f"\n💾 Saving raw data to {raw_filename}...")
    df.to_excel(raw_filename, index=False)
    
    # 4. VERIFY & PROCESS (Add Indicators)
    print("--- ⚙️ Adding Indicators & Targets ---")
    
    # Initialize your custom engineer
    engineer = FeatureEngineer(df)
    
    # Add Indicators
    df_features = engineer.add_indicators()
    
    # Add Target (Using a larger threshold for Daily candles, e.g., 1% move)
    # Since daily moves are bigger than hourly, we want significant targets
    df_features = engineer.add_target(threshold=0.01) 
    
    # 5. CHECK DATA
    print("\n--- DATA VERIFICATION ---")
    print(f"Total Rows: {len(df_features)}")
    print("\nFirst 5 Rows (History Start):")
    print(df_features[['timestamp', 'close', 'RSI', 'target']].head())
    
    print("\nLast 5 Rows (Present Day):")
    print(df_features[['timestamp', 'close', 'RSI', 'target']].tail())
    
    # Save PROCESSED data
    proc_filename = 'data/btc_daily_2010_2025_processed.xlsx'
    print(f"\n💾 Saving processed data to {proc_filename}...")
    df_features.to_excel(proc_filename, index=False)
    print("Done! You can open the Excel file to verify.")

if __name__ == "__main__":
    fetch_and_process_history()