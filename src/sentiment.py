from transformers import pipeline
from datetime import datetime
import requests

class NewsSentimentEngine:
    def __init__(self):
        # Load FinBERT (Download happens once, then caches)
        print("🧠 Loading FinBERT for Sentiment Analysis...")
        self.analyzer = pipeline("text-classification", model="ProsusAI/finbert", device=-1) # device=0 for GPU

    def fetch_crypto_news(self, limit=5):
        """
        Fetches latest news from a provider (Example: CryptoPanic or NewsAPI)
        For this demo, we simulate headlines to show how FinBERT works.
        """
        # In production, replace this with requests.get("https://cryptopanic.com/api/...")
        headlines = [
            "Bitcoin breaks resistance as institutional inflows surge to record highs",
            "SEC delays decision on Ethereum ETF, causing market uncertainty",
            "Binance faces new regulatory hurdles in Europe",
            "Solana network outage fixed, transaction speed up 20%",
            "Whales are accumulating BTC at $60k support level"
        ]
        return headlines[:limit]

    def get_market_sentiment(self):
        """
        Returns a Score: -1.0 (Bearish) to +1.0 (Bullish)
        """
        news = self.fetch_crypto_news()
        if not news: return 0.0

        total_score = 0
        
        for title in news:
            result = self.analyzer(title)[0] # {'label': 'positive', 'score': 0.95}
            label = result['label']
            confidence = result['score']
            
            # Convert label to numeric score
            if label == 'positive': score = 1 * confidence
            elif label == 'negative': score = -1 * confidence
            else: score = 0 # Neutral
            
            total_score += score
            
        # Normalize
        avg_sentiment = total_score / len(news)
        return round(avg_sentiment, 2)

# Usage Example:
# engine = NewsSentimentEngine()
# print(f"Current Sentiment: {engine.get_market_sentiment()}")