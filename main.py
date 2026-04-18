import time
from src.engine import TradingEngine

# This is just for CLI testing
if __name__ == "__main__":
    print("🖥️ Running Bot in CLI Mode (No Server)")
    
    engine = TradingEngine(mode="FUTURES")
    engine.start()
    
    try:
        while True:
            # If the engine produced data, print it
            if not engine.message_queue.empty():
                data = engine.message_queue.get()
                print(f"📊 Equity: ${data['equity']:.2f} | Action: {data['decision']}")
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()