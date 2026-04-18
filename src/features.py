import pandas as pd
import pandas_ta as ta
import numpy as np

class FeatureEngineer:
    def __init__(self, data):
        self.df = data.copy()

    def add_indicators(self):
        """
        Engineers features for both Swing (1h) and Scalping (1m/5m).
        """
        # --- 1. EXISTING BASE FEATURES (Good for Trend/Swing) ---
        
        # RSI (Standard 14) + Lag
        self.df['RSI'] = self.df.ta.rsi(length=14)
        self.df['RSI_lag1'] = self.df['RSI'].shift(1)
        
        # Bollinger Bands (Volatility)
        bb_features = self.df.ta.bbands(length=20, std=2)
        if bb_features is not None:
            # Robust column extraction
            bb_p_col = [col for col in bb_features.columns if col.startswith('BBP')][0]
            bb_w_col = [col for col in bb_features.columns if col.startswith('BBB')][0]
            self.df['BB_pct'] = bb_features[bb_p_col]
            self.df['BB_width'] = bb_features[bb_w_col]

        # MACD (Momentum)
        macd = self.df.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None:
            macd_h_col = [col for col in macd.columns if col.startswith('MACDh')][0]
            self.df['MACD_diff'] = macd[macd_h_col]

        # OBV (Volume Flow)
        self.df['OBV'] = self.df.ta.obv()
        self.df['OBV_slope'] = self.df['OBV'].pct_change()

        # ATR (Normalized Risk)
        self.df['ATR'] = self.df.ta.atr(length=14)
        self.df['ATR_pct'] = self.df['ATR'] / self.df['close']

        # SMA Distance (Mean Reversion Long Term)
        self.df['SMA_50'] = self.df.ta.sma(length=50)
        self.df['SMA_dist'] = (self.df['close'] - self.df['SMA_50']) / self.df['SMA_50']

        # --- 2. NEW SCALPING FEATURES (High Frequency) ---
        
        # ⚡ Relative Volume (RVol) - Critical for Breakouts
        # "Is current volume 3x normal?"
        vol_ma = self.df['volume'].rolling(window=20).mean()
        self.df['RVol'] = self.df['volume'] / (vol_ma + 1e-9)

        # ⚡ Fast RSI (Length 7) - Quick Reversals
        self.df['RSI_fast'] = self.df.ta.rsi(length=7)

        # ⚡ VWAP Distance (Institutional Fair Value)
        # Using Typical Price * Volume accumulation
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        self.df['vwap'] = (tp * self.df['volume']).cumsum() / self.df['volume'].cumsum()
        self.df['dist_vwap'] = (self.df['close'] - self.df['vwap']) / self.df['vwap']

        # ⚡ Candle Wicks (Price Action)
        # Large Upper Wick = Rejection (Bearish)
        # Large Lower Wick = Buying Support (Bullish)
        self.df['upper_wick'] = (self.df['high'] - self.df[['open', 'close']].max(axis=1)) / self.df['close']
        self.df['lower_wick'] = (self.df[['open', 'close']].min(axis=1) - self.df['low']) / self.df['close']

        # --- 3. CLEANUP ---
        self.df['pct_change'] = self.df['close'].pct_change()
        
        # Drop raw/intermediate columns to keep the model focused
        cols_to_drop = ['OBV', 'ATR', 'SMA_50', 'vwap'] 
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Replace Inf values with 0 (Common in RVol)
        self.df.replace([np.inf, -np.inf], 0, inplace=True)
        self.df.dropna(inplace=True)
        
        return self.df

    def add_target(self, threshold=0.0025):
        """
        Creates the 'Target' column for Machine Learning.
        1 = Price goes up by > threshold
        """
        # Look 1 candle into the future
        future_pct_change = self.df['close'].pct_change().shift(-1)
        
        # Check imbalance: Print how many Buy signals we actually have
        buy_signals = (future_pct_change > threshold).sum()
        total_signals = len(future_pct_change)
        
        # Avoid division by zero in print
        ratio = buy_signals / total_signals if total_signals > 0 else 0
        print(f"--- 📊 DATA CHECK: Found {buy_signals} BUY signals out of {total_signals} ({ratio:.1%}) ---")
        
        self.df['target'] = (future_pct_change > threshold).astype(int)
        self.df.dropna(inplace=True)
        return self.df