🤖 Quantitative AI Trading System (Bitcoin)
A full-stack algorithmic trading platform designed for high-frequency Bitcoin analysis and automated execution. This system utilizes a Multi-Model Ensemble approach—combining Deep Learning with Gradient Boosting—to predict price direction with high confidence.

The platform features a modular execution engine capable of switching between Spot and Futures (Leverage) trading modes, visualized in real-time via a React WebSocket dashboard.

🧠 AI Architecture: The "Ensemble 5"
Instead of relying on a single algorithm, this bot aggregates predictions from 5 distinct models to filter out noise and reduce false positives.

LSTM (Long Short-Term Memory): Deep Learning model optimized for time-series sequence and pattern recognition.

XGBoost: Gradient boosting framework for capturing non-linear relationships in market data.

LightGBM: High-efficiency gradient boosting model focusing on speed and accuracy.

Random Forest: Ensemble of decision trees to reduce overfitting and improve generalization.

Voting Ensemble: A meta-learner that aggregates weighted votes from all models to make the final trading decision.

🚀 Key Features
Dual-Mode Execution Engine:

Spot Mode: Traditional accumulation strategy (Buy & Hold).

Futures Mode: Leverage-ready engine with Shorting capabilities and Margin management.

Advanced Feature Engineering: Real-time calculation of MACD, RSI, Bollinger Bands, ATR (Volatility), and OBV Slope.

Real-Time Dashboard: React.js frontend displaying live candles, model confidence, and portfolio equity via WebSockets.

Professional Backtesting: Event-driven backtester accounting for fees (0.1%), slippage, and spread.

DevOps Ready: Fully containerized with Docker and CI/CD pipelines via GitHub Actions.

📊 Performance (2023-2026 Backtest)
Backtest results on hourly (1h) Bitcoin data:

![performane_spot](asset/img1.png)

🛠️ Tech Stack
Quantitative Core
Language: Python 3.10

Data Science: Pandas, NumPy, TA-Lib (Technical Analysis)

AI Frameworks: TensorFlow (Keras), Scikit-Learn, XGBoost, LightGBM

Full Stack Infrastructure
Backend API: FastAPI (Async/Await)

Real-Time Transport: WebSockets

Frontend: React.js, TradingView Lightweight Charts

CI/CD: GitHub Actions, Docker

⚡ Quick Start
1. Clone & Setup

git clone https://github.com/mhsm555/bitcoin-trading-bot.git
cd bitcoin-trading-bot
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

2. Train the Brain
Train all 5 models and save the ensemble weights.

python -m src.train_ml      # Trains XGB, RF, LightGBM
python -m src.train_dl      # Trains LSTM

3. Run the Backtest
Verify performance before going live.

python -m src.backtest --model model_ensemble

4. Launch the Platform
Start the backend API and the WebSocket server.

uvicorn src.server:app --reload

The React Dashboard will be available at http://localhost:3000 (requires npm start in /frontend).

⚠️ Disclaimer
This software is for educational and research purposes only. Do not trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred by using this bot.