import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import sys
import yaml  # Import YAML library

# Ensure we can find 'src'
sys.path.append(os.getcwd())
try:
    from src.ai_model import CryptoModel
except ImportError:
    print(f"Critical Error: Could not import src.ai_model.")
    sys.exit(1)

def load_config():
    """Loads parameters from the YAML config file."""
    try:
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)['backtest']
        print("Successfully loaded backtest configuration.")
        return config
    except Exception as e:
        print(f"Critical Error: Could not load config/config.yaml. Make sure it exists. Error: {e}")
        sys.exit(1)

def apply_slippage(price, side, slippage_pct):
    """Applies slippage to the execution price."""
    if side == 'buy':
        return price * (1 + slippage_pct)
    elif side == 'sell':
        return price * (1 - slippage_pct)
    return price

def run_backtest(model_name, trend_bias):
    """
    Runs a futures backtest with realistic fees and slippage.
    """
    # 0. Load Configuration
    config = load_config()
    INITIAL_CAPITAL = config['initial_capital']
    TAKER_FEE = config['taker_fee']
    SLIPPAGE_PCT = config['slippage_pct']
    STOP_LOSS_PCT = config['stop_loss_pct']
    LONG_THRESHOLD = config['long_threshold']
    SHORT_THRESHOLD = config['short_threshold']
    EXIT_LONG_THRESHOLD = config['exit_long_threshold']
    EXIT_SHORT_THRESHOLD = config['exit_short_threshold']

    print(f"\n==========================================")
    print(f"🔙 STARTING FUTURES BACKTEST")
    print(f"🤖 Model: {model_name}")
    print(f"🌊 Trend Bias: {trend_bias}")
    print(f"⚙️ Config: Fee={TAKER_FEE*100:.3f}%, Slippage={SLIPPAGE_PCT*100:.3f}%")
    print(f"==========================================")
    
    # 1. Load Model
    if 'lstm' in model_name: init_type = 'lstm'
    elif 'xgb' in model_name: init_type = 'xgb'
    else: init_type = 'ensemble'
        
    bot = CryptoModel(model_type=init_type)
    try:
        bot.load_model(model_name)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return

    # 2. Load Data
    if '1h' in model_name: tf = '1h'
    elif '15m' in model_name: tf = '15m'
    elif '5m' in model_name: tf = '5m'
    elif '1m' in model_name: tf = '1m'
    else: tf = '1h'

    file_path = f'data/btc_{tf}_processed.csv'
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
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
    
    print("⚡ Simulating Futures trades...")
    
    for i in range(len(test_df)):
        current_close = test_df.loc[i, 'close']
        current_high = test_df.loc[i, 'high']
        current_low = test_df.loc[i, 'low']
        current_time = test_df.loc[i, 'timestamp']
        confidence = probs[i]

        # --- A. CHECK STOP LOSS / TAKE PROFIT ---
        if position == 'LONG':
            sl_price = entry_price * (1 - STOP_LOSS_PCT)
            if current_low <= sl_price:
                # HIT STOP LOSS (treat as market sell)
                exit_price = apply_slippage(sl_price, 'sell', SLIPPAGE_PCT)
                trade_value = entry_size * exit_price
                fee = trade_value * TAKER_FEE
                pnl = (exit_price - entry_price) * entry_size - fee
                balance += pnl
                trades.append({'type': 'SL_LONG', 'pnl': pnl, 'time': current_time})
                position = None
                
        elif position == 'SHORT':
            sl_price = entry_price * (1 + STOP_LOSS_PCT)
            if current_high >= sl_price:
                # HIT STOP LOSS (treat as market buy)
                exit_price = apply_slippage(sl_price, 'buy', SLIPPAGE_PCT)
                trade_value = entry_size * exit_price
                fee = trade_value * TAKER_FEE
                pnl = (entry_price - exit_price) * entry_size - fee
                balance += pnl
                trades.append({'type': 'SL_SHORT', 'pnl': pnl, 'time': current_time})
                position = None

        # --- B. CHECK AI SIGNALS (If still in position) ---
        if position == 'LONG' and confidence < EXIT_LONG_THRESHOLD:
            exit_price = apply_slippage(current_close, 'sell', SLIPPAGE_PCT)
            trade_value = entry_size * exit_price
            fee = trade_value * TAKER_FEE
            pnl = (exit_price - entry_price) * entry_size - fee
            balance += pnl
            trades.append({'type': 'CLOSE_LONG', 'pnl': pnl, 'time': current_time})
            position = None
                
        elif position == 'SHORT' and confidence > EXIT_SHORT_THRESHOLD:
            exit_price = apply_slippage(current_close, 'buy', SLIPPAGE_PCT)
            trade_value = entry_size * exit_price
            fee = trade_value * TAKER_FEE
            pnl = (entry_price - exit_price) * entry_size - fee
            balance += pnl
            trades.append({'type': 'CLOSE_SHORT', 'pnl': pnl, 'time': current_time})
            position = None

        # --- C. ENTER NEW POSITION (If empty) ---
        if position is None:
            allow_long = trend_bias in ['NEUTRAL', 'LONG_ONLY']
            allow_short = trend_bias in ['NEUTRAL', 'SHORT_ONLY']

            if confidence >= LONG_THRESHOLD and allow_long:
                position = 'LONG'
                exec_price = apply_slippage(current_close, 'buy', SLIPPAGE_PCT)
                entry_price = exec_price
                cost_basis = balance * 0.98 # Use 98% of equity
                trade_fee = cost_basis * TAKER_FEE
                entry_size = (cost_basis - trade_fee) / entry_price
                balance -= trade_fee # Deduct entry fee
                
            elif confidence <= SHORT_THRESHOLD and allow_short:
                position = 'SHORT'
                exec_price = apply_slippage(current_close, 'sell', SLIPPAGE_PCT)
                entry_price = exec_price
                cost_basis = balance * 0.98
                trade_fee = cost_basis * TAKER_FEE
                entry_size = (cost_basis - trade_fee) / entry_price
                balance -= trade_fee # Deduct entry fee

        # --- D. TRACK EQUITY ---
        unrealized_pnl = 0
        if position == 'LONG':
            unrealized_pnl = (current_close - entry_price) * entry_size
        elif position == 'SHORT':
            unrealized_pnl = (entry_price - current_close) * entry_size
            
        equity_curve.append(balance + unrealized_pnl)

    # 5. Report Results
    final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
    profit_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"\n🏁 RESULTS FOR {model_name} [{trend_bias}]")
    print(f"💰 Final Equity: ${final_equity:,.2f}")
    print(f"📈 Total Return: {profit_pct:.2f}%")
    print(f"🔢 Total Trades: {len(trades)}")
    
    if len(trades) > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        print(f"🏆 Win Rate:      {win_rate:.2%}")
        print(f"📈 Payoff Ratio:  {payoff_ratio:.2f}")
        print(f"🥇 Avg Win:      ${avg_win:,.2f}")
        print(f"👎 Avg Loss:     ${avg_loss:,.2f}")
        
        sl_hits = [t for t in trades if 'SL_' in t['type']]
        print(f"🛑 Stop Losses:   {len(sl_hits)} ({len(sl_hits)/len(trades):.1%})")

    # 6. Save Chart
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(test_df['timestamp']), equity_curve, label=f'Equity Curve', color='blue')
    plt.axhline(INITIAL_CAPITAL, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    plt.title(f"Futures Backtest (Fee & Slippage Adjusted): {model_name} [{trend_bias}]")
    plt.ylabel("Equity ($)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_file = f"backtest_futures_{model_name}_{trend_bias}.png"
    plt.savefig(output_file)
    print(f"🖼️  Chart saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a futures trading backtest with realistic costs.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model file to test (e.g., model_lstm_1h).')
    parser.add_argument('--trend', type=str, default='NEUTRAL', 
                        choices=['NEUTRAL', 'LONG_ONLY', 'SHORT_ONLY'],
                        help='Restrict trades to a specific direction.')
    
    args = parser.parse_args()
    run_backtest(args.model, args.trend)