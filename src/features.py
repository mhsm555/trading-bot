# src/features.py
import pandas as pd
import pandas_ta as ta

class FeatureEngineer:
    def __init__(self, data):
        self.df = data.copy()

    def add_indicators(self):
        # ... (Keep existing RSI, Lag, BB code) ...
        self.df['RSI'] = self.df.ta.rsi(length=14)
        self.df['RSI_lag1'] = self.df['RSI'].shift(1)
        
        # Bollinger Bands (Keep your dynamic fix here)
        bb_features = self.df.ta.bbands(length=20, std=2)
        bb_p_col = [col for col in bb_features.columns if col.startswith('BBP')][0]
        bb_w_col = [col for col in bb_features.columns if col.startswith('BBB')][0]
        self.df['BB_pct'] = bb_features[bb_p_col]
        self.df['BB_width'] = bb_features[bb_w_col]

        # --- NEW INDICATORS ---
        
        # 1. MACD (Moving Average Convergence Divergence)
        # Returns MACD (line), MACD_H (histogram), MACD_S (signal)
        macd = self.df.ta.macd(fast=12, slow=26, signal=9)
        # We want the Histogram (h) because it shows momentum strength
        macd_h_col = [col for col in macd.columns if col.startswith('MACDh')][0]
        self.df['MACD_diff'] = macd[macd_h_col]

        # 2. OBV (On-Balance Volume) - Tracks buying pressure
        self.df['OBV'] = self.df.ta.obv()
        # OBV is a raw number (like 1,000,000), which is bad for ML. 
        # We need the slope (is volume rising or falling?)
        self.df['OBV_slope'] = self.df['OBV'].pct_change()

        # 3. ATR (Average True Range) - Measures Volatility (Risk)
        self.df['ATR'] = self.df.ta.atr(length=14)
        # Normalize ATR by price so it works at any price level
        self.df['ATR_pct'] = self.df['ATR'] / self.df['close']

        # Stationarity: SMA Dist & Price Change
        self.df['SMA_50'] = self.df.ta.sma(length=50)
        self.df['SMA_dist'] = (self.df['close'] - self.df['SMA_50']) / self.df['SMA_50']
        self.df['pct_change'] = self.df['close'].pct_change()

        self.df.dropna(inplace=True)
        return self.df

    def add_target(self, threshold=0.0025): # Lowered from 0.005
        """
        1 = Price goes up by > 0.25%
        """
        future_pct_change = self.df['close'].pct_change().shift(-1)
        
        # Check imbalance: Print how many Buy signals we actually have
        buy_signals = (future_pct_change > threshold).sum()
        total_signals = len(future_pct_change)
        print(f"--- DATA CHECK: Found {buy_signals} BUY signals out of {total_signals} rows ({buy_signals/total_signals:.1%}) ---")
        
        self.df['target'] = (future_pct_change > threshold).astype(int)
        self.df.dropna(inplace=True)
        return self.df