import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from src.ai_model import CryptoModel

# --- CONFIGURATION ---
INITIAL_CAPITAL = 10000
TRADING_FEE = 0.001  # 0.1% fee
BUY_THRESHOLD = 0.60  # Buy if AI is 60% sure
SELL_THRESHOLD = 0.50 # Sell if AI drops below 50%

def run_backtest(model_name):
    print(f"\n==========================================")
    print(f"🔙 STARTING BACKTEST: {model_name}")
    print(f"==========================================")
    
    # 1. Load Model
    # Initialize with a dummy type; load_model will fix it based on metadata.
    if 'lstm' in model_name:
        init_type = 'lstm'
    else:
        init_type = 'ensemble' 
        
    bot = CryptoModel(model_type=init_type)
    
    try:
        bot.load_model(model_name)
    except Exception as e:
        print(f"❌ Could not load {model_name}: {e}")
        return

    # 2. Load Data
    file_path = 'data/btc_hourly_2020_2026_processed.xlsx'
    if not os.path.exists(file_path):
        print("❌ Data file not found.")
        return
        
    df = pd.read_excel(file_path)
    
    # Filter for the "Test" period (e.g., 2023-2026)
    test_df = df[df['timestamp'] >= '2023-01-01'].copy().reset_index(drop=True)
    
    print(f"📊 Testing on {len(test_df)} hours (from 2023-01-01)...")
    
    # 3. Generate Predictions
    print("🧠 Generating AI predictions...")
    
    probs = []
    
    # --- LOGIC FOR LSTM (Deep Learning) ---
    if bot.model_type == 'lstm':
        print("⚙️  Mode: Deep Learning (Sequences)")
        # Scale data
        scaled_data = bot.scaler.transform(test_df[bot.features])
        
        # Create Sequences
        lookback = bot.lookback
        X_test = []
        for i in range(lookback, len(scaled_data)):
            X_test.append(scaled_data[i-lookback:i])
        X_test = np.array(X_test)
        
        # Predict
        raw_predictions = bot.model.predict(X_test, verbose=0).flatten()
        
        # Pad the beginning (since we can't predict the first few rows)
        padding = np.zeros(lookback)
        probs = np.concatenate([padding, raw_predictions])
        
    # --- LOGIC FOR ML (XGB, RF, Ensemble) ---
    else:
        print(f"⚙️  Mode: Machine Learning ({bot.model_type.upper()})")
        features = test_df[bot.features]
        # predict_proba returns [prob_loss, prob_win]. We want column 1.
        probs = bot.model.predict_proba(features)[:, 1]

    # --- 4. DEBUG SCAN (Now works for ALL models) ---
    print(f"\n--- 🧠 {model_name.upper()} BRAIN SCAN ---")
    print(f"Max Confidence: {np.max(probs):.2%}")
    print(f"Avg Confidence: {np.mean(probs):.2%}")
    print(f"Trades > {BUY_THRESHOLD:.0%}:   {np.sum(probs > BUY_THRESHOLD)}")
    print("----------------------------\n")

    # 5. Simulation Loop
    balance = INITIAL_CAPITAL
    btc_held = 0
    equity_curve = []
    trades = []
    in_position = False
    
    print("⚡ Simulating trades...")
    
    for i in range(len(test_df)):
        current_price = test_df.loc[i, 'close']
        current_time = test_df.loc[i, 'timestamp']
        confidence = probs[i]
        
        # --- STRATEGY ---
        # BUY Signal
        if confidence >= BUY_THRESHOLD and not in_position:
            amount_to_buy = (balance * 0.99) / current_price 
            cost = amount_to_buy * current_price
            fee = cost * TRADING_FEE
            
            balance -= (cost + fee)
            btc_held = amount_to_buy
            in_position = True
            
            trades.append({'type': 'BUY', 'time': current_time, 'price': current_price, 'val': balance})

        # SELL Signal
        elif confidence < SELL_THRESHOLD and in_position:
            revenue = btc_held * current_price
            fee = revenue * TRADING_FEE
            
            balance += (revenue - fee)
            btc_held = 0
            in_position = False
            
            trades.append({'type': 'SELL', 'time': current_time, 'price': current_price, 'val': balance})
            
        # Track Equity
        current_equity = balance + (btc_held * current_price)
        equity_curve.append(current_equity)

    # 6. Report Results
    final_equity = equity_curve[-1]
    profit_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"\n🏁 RESULTS FOR {model_name}")
    print(f"💰 Final Equity: ${final_equity:.2f}")
    print(f"📈 Total Return: {profit_pct:.2f}%")
    print(f"🔢 Total Trades: {len(trades)}")
    
    if len(trades) > 0:
        wins = [t for i, t in enumerate(trades) if t['type'] == 'SELL' and t['val'] > trades[i-1]['val']]
        win_rate = len(wins) / (len(trades)/2) if len(trades) > 0 else 0
        print(f"🏆 Win Rate:     {win_rate:.2%}")

    # 7. Save Chart
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['timestamp'], equity_curve, label=f'{model_name} Strategy', color='blue')
    
    # Add Buy & Hold for comparison
    first_price = test_df.iloc[0]['close']
    buy_hold = [INITIAL_CAPITAL * (p / first_price) for p in test_df['close']]
    plt.plot(test_df['timestamp'], buy_hold, label='Buy & Hold BTC', color='gray', alpha=0.5, linestyle='--')
    
    plt.title(f"Backtest: {model_name} (Threshold {BUY_THRESHOLD})")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"backtest_{model_name}.png"
    plt.savefig(output_file)
    print(f"🖼️  Chart saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Backtest a specific AI model.')
    parser.add_argument('--model', type=str, default='model_btc_lstm', 
                        help='Name of the model (e.g., model_xgb, model_btc_lstm)')
    
    args = parser.parse_args()
    
    run_backtest(args.model)