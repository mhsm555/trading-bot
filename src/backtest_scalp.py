import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import sys
import pandas_ta as ta
from numpy.lib.stride_tricks import sliding_window_view  # Required for LSTM

# Ensure we can find 'src'
sys.path.append(os.getcwd())
try:
    from src.ai_model import CryptoModel
    from src.features import FeatureEngineer
    try:
        from stable_baselines3 import PPO
        HAS_RL = True
    except ImportError:
        HAS_RL = False
except ImportError:
    print("❌ Critical Error: Missing dependencies.")
    sys.exit(1)

# --- ⚡ CONFIGURATION ---
INITIAL_CAPITAL = 10000
TRADING_FEE = 0.0004     
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 1.5  
ATR_MULTIPLIER_TP = 2.5  
LONG_THRESHOLD = 0.60
SHORT_THRESHOLD = 0.40

def run_scalp_backtest(model_name, trend_bias):
    print(f"\n==========================================")
    print(f"⚡ STARTING UNIVERSAL BACKTEST: {model_name}")
    print(f"🌊 Trend Bias: {trend_bias}")
    print(f"==========================================")
    
    # --- 1. DETECT MODEL TYPE ---
    is_rl = "agent" in model_name or "ppo" in model_name
    bot = None
    rl_agent = None

    if is_rl:
        if not HAS_RL:
            print("❌ Error: 'stable_baselines3' not installed.")
            return
        print("🤖 Mode: Reinforcement Learning (Hybrid)")
        try:
            rl_agent = PPO.load(f"data/{model_name}")
            # RL needs the Analyst (XGBoost 5m) helper
            bot = CryptoModel('xgb')
            bot.load_model('model_xgb_5m')
        except Exception as e:
            print(f"❌ Error loading RL Agent: {e}")
            return
    else:
        print("🧠 Mode: Standard ML/DL")
        # ✅ FIX: Enhanced Detection for ALL model types
        init_type = 'xgb' # Default
        if 'rf' in model_name: init_type = 'rf'
        if 'lstm' in model_name: init_type = 'lstm'
        if 'ensemble' in model_name: init_type = 'ensemble'
        if 'lgbm' in model_name: init_type = 'lgbm'
        
        bot = CryptoModel(model_type=init_type)
        try:
            bot.load_model(model_name)
        except Exception as e:
            print(f"   ❌ Could not load {model_name}: {e}")
            return

    # --- 2. LOAD DATA ---
    # Detect Timeframe
    tf = '5m'
    if '1m' in model_name: tf = '1m'
    if '15m' in model_name: tf = '15m'
    if '1h' in model_name: tf = '1h'
    
    file_path = f'data/btc_{tf}_processed.csv'
    if not os.path.exists(file_path):
        print(f"❌ Data file not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # --- 3. RE-ENGINEER FEATURES ---
    fe = FeatureEngineer(df)
    df = fe.add_indicators()

    # Filter for Backtest Period
    test_start_date = '2024-01-01'
    test_df = df[df['timestamp'] >= test_start_date].copy().reset_index(drop=True)
    
    if test_df.empty:
        print(f"❌ No data found after {test_start_date}")
        return

    # --- 4. GENERATE SIGNALS (The Fix) ---
    print("🔮 Generating Analyst Probabilities...")
    feature_data = test_df[bot.features]
    scaled_features = bot.scaler.transform(feature_data)
    
    if bot.model_type == 'lstm':
        # ✅ FIX: Correct Sliding Window for LSTM (Handles lookback=10)
        lookback = bot.lookback
        # Create windows: (N, Lookback, Features)
        # Note: sliding_window_view is available in numpy >= 1.20
        try:
            X = sliding_window_view(scaled_features, window_shape=lookback, axis=0)
            X = np.swapaxes(X, 1, 2) # (N, Lookback, Features)
            
            # Predict
            raw_preds = bot.model.predict(X, verbose=0)
            
            # Pad the beginning (first 'lookback-1' rows have no prediction)
            padding = np.full(lookback - 1, 0.5) 
            probs = np.concatenate((padding, raw_preds.flatten()))
            
        except Exception as e:
            print(f"⚠️ LSTM Windowing Error: {e}. Falling back to simple reshape.")
            # Fallback for simple shape mismatch
            preds = bot.model.predict(scaled_features.reshape(len(scaled_features), 1, len(bot.features)), verbose=0)
            probs = preds.flatten()
    else:
        # Standard ML
        probs = bot.model.predict_proba(scaled_features)[:, 1]

    test_df['ml_signal'] = probs

    # --- 5. SIMULATION LOOP ---
    balance = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    position = None 

    print(f"⚡ Simulating {len(test_df)} candles...")

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        current_price = row['close']
        current_atr = row.get('ATR', row['close']*0.01)
        
        decision = 0 # 0=HOLD, 1=LONG, -1=SHORT, 99=CLOSE
        
        if is_rl:
            # 🤖 RL LOGIC
            pnl_pct = 0.0
            if position:
                if position['side'] == 'LONG': pnl_pct = (current_price - position['entry']) / position['entry']
                else: pnl_pct = (position['entry'] - current_price) / position['entry']
            
            obs = np.array([
                row['ml_signal'],
                0.0,  # Sentiment placeholder
                row.get('RSI', 50) / 100.0,
                row.get('ATR_pct', 0.01),
                pnl_pct
            ], dtype=np.float32)
            
            action, _ = rl_agent.predict(obs, deterministic=True)
            if action == 1: decision = 1
            elif action == 2: decision = -1
            elif action == 3: decision = 99 
            
        else:
            # 🧠 ML LOGIC
            prob = row['ml_signal']
            if prob >= LONG_THRESHOLD: decision = 1
            elif prob <= SHORT_THRESHOLD: decision = -1

        # Trend Filter
        if trend_bias == "LONG_ONLY" and decision == -1: decision = 0
        if trend_bias == "SHORT_ONLY" and decision == 1: decision = 0

        # --- EXECUTION ---
        # 1. Close Position
        should_close = False
        if decision == 99: should_close = True
        if decision != 0 and position and position['side'] != ('LONG' if decision==1 else 'SHORT'):
            should_close = True
            
        # ML SL/TP (Only if NOT RL)
        if not is_rl and position:
            if position['side'] == 'LONG':
                if row['low'] <= position['sl'] or row['high'] >= position['tp']: should_close = True
            else:
                if row['high'] >= position['sl'] or row['low'] <= position['tp']: should_close = True

        if should_close and position:
            if position['side'] == 'LONG': pnl = (current_price - position['entry']) * position['size']
            else: pnl = (position['entry'] - current_price) * position['size']
            
            fee = (current_price * position['size']) * TRADING_FEE
            balance += (position['collateral'] + pnl - fee)
            trades.append(pnl - fee)
            position = None

        # 2. Open Position
        if decision in [1, -1] and not position:
            cost = balance * 0.10 
            fee = cost * TRADING_FEE
            collateral = cost - fee
            balance -= cost
            
            size = collateral / current_price
            side = 'LONG' if decision == 1 else 'SHORT'
            
            sl = 0; tp = 0
            if not is_rl:
                # Dynamic ATR Stops
                if side == 'LONG':
                    sl = current_price - (current_atr * ATR_MULTIPLIER_SL)
                    tp = current_price + (current_atr * ATR_MULTIPLIER_TP)
                else:
                    sl = current_price + (current_atr * ATR_MULTIPLIER_SL)
                    tp = current_price - (current_atr * ATR_MULTIPLIER_TP)
            
            position = {'entry': current_price, 'size': size, 'side': side, 'collateral': collateral, 'sl': sl, 'tp': tp}

        # Track
        equity = balance
        if position:
            if position['side'] == 'LONG': u_pnl = (current_price - position['entry']) * position['size']
            else: u_pnl = (position['entry'] - current_price) * position['size']
            equity += u_pnl
        equity_curve.append(equity)

    # --- REPORT ---
    final_equity = equity_curve[-1]
    profit_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"\n🏁 RESULTS: {model_name}")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Return:       {profit_pct:.2f}%")
    print(f"🔢 Trades:       {len(trades)}")
    
    if not os.path.exists('assets'): os.makedirs('assets')
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Backtest: {model_name}")
    plt.savefig(f"assets/scalp_{model_name}.png")
    print(f"🖼️ Chart saved to assets/scalp_{model_name}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--trend', type=str, default='NEUTRAL')
    args = parser.parse_args()
    
    run_scalp_backtest(args.model, args.trend)