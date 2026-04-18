import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import sys

# Ensure we can find 'src'
sys.path.append(os.getcwd())
try:
    from src.ai_model import CryptoModel
except ImportError:
    print("❌ Critical Error: Could not import src.ai_model.")
    sys.exit(1)

# --- ⚙️ FUTURES CONFIGURATION ---
INITIAL_CAPITAL = 10000
TRADING_FEE = 0.0004     # 0.04% Futures Fee
STOP_LOSS_PCT = 0.02     # 2% Hard Stop Loss
TAKE_PROFIT_PCT = 0.04   # 4% Take Profit (Optional, can set to None)

# AI THRESHOLDS
LONG_THRESHOLD = 0.60    # Buy if > 60%
SHORT_THRESHOLD = 0.40   # Short if < 40%
EXIT_LONG_THRESHOLD = 0.50  # Exit Long if drops below 50%
EXIT_SHORT_THRESHOLD = 0.50 # Exit Short if rises above 50%

def run_backtest(model_name, trend_bias):
    print(f"\n==========================================")
    print(f"🔙 STARTING FUTURES BACKTEST")
    print(f"🤖 Model: {model_name}")
    print(f"🌊 Trend Bias: {trend_bias}")
    print(f"==========================================")
    
    # 1. Load Model
    if 'lstm' in model_name: init_type = 'lstm'
    elif 'xgb' in model_name: init_type = 'xgb'
    else: init_type = 'ensemble'
        
    bot = CryptoModel(model_type=init_type)
    try:
        bot.load_model(model_name)
    except Exception as e:
        print(f"❌ Could not load {model_name}: {e}")
        return

    # 2. Load Data
    if '1h' in model_name: tf = '1h'
    elif '15m' in model_name: tf = '15m'
    elif '5m' in model_name: tf = '5m'
    elif '1m' in model_name: tf = '1m'
    else: tf = '1h'

    file_path = f'data/btc_{tf}_processed.csv'
    if not os.path.exists(file_path):
        print(f"❌ Data file not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    test_start_date = '2024-01-01' 
    test_df = df[df['timestamp'] >= test_start_date].copy().reset_index(drop=True)
    
    print(f"📊 Testing on {len(test_df)} candles (from {test_start_date})...")
    
    # 3. Generate Predictions
    print("🧠 Generating AI predictions...")
    try:
        feature_data = test_df[bot.features]
        scaled_features = bot.scaler.transform(feature_data)
    except Exception:
        print("❌ Scaling Error. Retrain model.")
        return

    if bot.model_type == 'lstm':
        lookback = bot.lookback
        X_test = []
        for i in range(lookback, len(scaled_features)):
            X_test.append(scaled_features[i-lookback:i])
        X_test = np.array(X_test)
        if len(X_test) > 0:
            raw = bot.model.predict(X_test, verbose=0).flatten()
            probs = np.concatenate([np.zeros(lookback), raw])
        else: return
    else:
        probs = bot.model.predict_proba(scaled_features)[:, 1]

    # 4. Simulation Loop
    balance = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    
    # Position State
    position = None # None, 'LONG', 'SHORT'
    entry_price = 0
    entry_size = 0  # In BTC
    entry_time = None

    print("⚡ Simulating Futures trades...")
    
    for i in range(len(test_df)):
        current_close = test_df.loc[i, 'close']
        current_high = test_df.loc[i, 'high'] # Needed for Stop Loss
        current_low = test_df.loc[i, 'low']   # Needed for Stop Loss
        current_time = test_df.loc[i, 'timestamp']
        confidence = probs[i]

        # --- A. CHECK STOP LOSS / TAKE PROFIT (Intra-candle) ---
        if position == 'LONG':
            sl_price = entry_price * (1 - STOP_LOSS_PCT)
            if current_low <= sl_price:
                # HIT STOP LOSS
                exit_price = sl_price
                pnl = (exit_price - entry_price) * entry_size
                balance += pnl
                trades.append({'type': 'SL_LONG', 'pnl': pnl, 'time': current_time})
                position = None
                
        elif position == 'SHORT':
            sl_price = entry_price * (1 + STOP_LOSS_PCT)
            if current_high >= sl_price:
                # HIT STOP LOSS
                exit_price = sl_price
                pnl = (entry_price - exit_price) * entry_size # Short PnL logic reversed
                balance += pnl
                trades.append({'type': 'SL_SHORT', 'pnl': pnl, 'time': current_time})
                position = None

        # --- B. CHECK AI SIGNALS (If still in position) ---
        if position == 'LONG':
            if confidence < EXIT_LONG_THRESHOLD:
                pnl = (current_close - entry_price) * entry_size
                balance += pnl
                trades.append({'type': 'CLOSE_LONG', 'pnl': pnl, 'time': current_time})
                position = None
                
        elif position == 'SHORT':
            if confidence > EXIT_SHORT_THRESHOLD:
                pnl = (entry_price - current_close) * entry_size
                balance += pnl
                trades.append({'type': 'CLOSE_SHORT', 'pnl': pnl, 'time': current_time})
                position = None

        # --- C. ENTER NEW POSITION (If empty) ---
        if position is None:
            # 🛠️ TREND FILTER LOGIC
            allow_long = trend_bias in ['NEUTRAL', 'LONG_ONLY']
            allow_short = trend_bias in ['NEUTRAL', 'SHORT_ONLY']

            # LONG SIGNAL
            if confidence >= LONG_THRESHOLD and allow_long:
                position = 'LONG'
                entry_price = current_close
                cost = balance * 0.98 # Use 98% of equity
                entry_size = cost / entry_price
                entry_time = current_time
                balance -= (cost * TRADING_FEE) 
                
            # SHORT SIGNAL
            elif confidence <= SHORT_THRESHOLD and allow_short:
                position = 'SHORT'
                entry_price = current_close
                cost = balance * 0.98
                entry_size = cost / entry_price
                entry_time = current_time
                balance -= (cost * TRADING_FEE)

        # --- D. TRACK EQUITY ---
        unrealized_pnl = 0
        if position == 'LONG':
            unrealized_pnl = (current_close - entry_price) * entry_size
        elif position == 'SHORT':
            unrealized_pnl = (entry_price - current_close) * entry_size
            
        equity_curve.append(balance + unrealized_pnl)

    # 5. Report Results
    final_equity = equity_curve[-1]
    profit_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"\n🏁 RESULTS FOR {model_name} [{trend_bias}]")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Total Return: {profit_pct:.2f}%")
    print(f"🔢 Total Trades: {len(trades)}")
    
    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        win_rate = len(wins) / len(trades)
        print(f"🏆 Win Rate:     {win_rate:.2%}")
        
        sl_hits = [t for t in trades if 'SL_' in t['type']]
        print(f"🛑 Stop Losses:  {len(sl_hits)} ({len(sl_hits)/len(trades):.1%})")

    # 6. Save Chart
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['timestamp'], equity_curve, label=f'{model_name} (L/S)', color='blue')
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5)
    plt.title(f"Futures Backtest: {model_name} [{trend_bias}]")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"assets/backtest_futures_{model_name}_{trend_bias}.png"
    plt.savefig(output_file)
    print(f"🖼️  Chart saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model_lstm_1h')
    # 🆕 NEW ARGUMENT
    parser.add_argument('--trend', type=str, default='NEUTRAL', 
                        choices=['NEUTRAL', 'LONG_ONLY', 'SHORT_ONLY'],
                        help='Restrict trades to a specific direction')
    
    args = parser.parse_args()
    run_backtest(args.model, args.trend)