import ccxt
import pandas as pd
import time

class MarketDataHandler:
    def __init__(self, exchange_id, symbol, timeframe='1h'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = getattr(ccxt, exchange_id)()

    def fetch_data(self, limit=5000):
        """
        Smart Fetch: Loops through history to get 'limit' amount of rows.
        """
        print(f"--- 🔄 Fetching {limit} candles for {self.symbol} ---")
        
        # Calculate how many milliseconds in the past we need to go
        # 1h = 3600000 ms
        duration = self.exchange.parse_timeframe(self.timeframe) * 1000
        now = self.exchange.milliseconds()
        since = now - (duration * limit)
        
        all_ohlcv = []
        
        while len(all_ohlcv) < limit:
            try:
                # Fetch what we can (usually 500-1000 per call)
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since=since)
                
                if not ohlcv:
                    print("No more data available.")
                    break
                
                # Add to our master list
                all_ohlcv.extend(ohlcv)
                
                # Update 'since' to the last timestamp we got + 1ms
                since = ohlcv[-1][0] + 1
                
                # Update user (Show progress)
                print(f"Downloaded {len(all_ohlcv)} / {limit} candles...")
                
                # Respect rate limits (don't get banned)
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        # Trim to exact limit
        all_ohlcv = all_ohlcv[-limit:]
        
        # Convert to DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove duplicates just in case
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        
        print(f"✅ Data ready: {len(df)} rows.")
        return df