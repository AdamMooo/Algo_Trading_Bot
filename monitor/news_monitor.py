import finnhub
import datetime
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import re
import time
from functools import lru_cache

class NewsMonitor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("FINNHUB_API_KEY")
        self.client = finnhub.Client(api_key=self.api_key)
        
        # Keywords for sentiment analysis
        self.bullish_words = set([
            'beats', 'exceeds', 'raises', 'higher', 'growth', 'positive',
            'upgrade', 'buy', 'strong', 'launch', 'success', 'partnership',
            'breakthrough', 'approved', 'wins', 'record'
        ])
        
        self.bearish_words = set([
            'miss', 'lower', 'down', 'cut', 'negative', 'downgrade',
            'sell', 'weak', 'decline', 'drop', 'investigation', 'lawsuit',
            'delay', 'recall', 'fails', 'warning'
        ])
        
    @lru_cache(maxsize=100)
    def get_company_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Get recent news for a company with caching"""
        try:
            # Add rate limiting
            time.sleep(0.1)  # 100ms delay between requests
            
            end = datetime.datetime.now()
            start = end - datetime.timedelta(hours=hours_back)
            
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            try:
                news = self.client.company_news(symbol, _from=start_str, to=end_str)
                news = sorted(news, key=lambda x: x['datetime'], reverse=True)
                return news
            except Exception as e:
                if "API limit reached" in str(e):
                    print(f"Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)  # Wait 60 seconds if rate limited
                    # Try one more time
                    news = self.client.company_news(symbol, _from=start_str, to=end_str)
                    return sorted(news, key=lambda x: x['datetime'], reverse=True)
                raise e
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def analyze_headline_sentiment(self, headline: str) -> Tuple[str, float]:
        """Analyze sentiment of a single headline"""
        headline = headline.lower()
        bull_count = sum(1 for word in self.bullish_words if word in headline)
        bear_count = sum(1 for word in self.bearish_words if word in headline)
        
        if bull_count > bear_count:
            strength = min((bull_count - bear_count) * 0.2, 1.0)
            return ('bullish', strength)
        elif bear_count > bull_count:
            strength = min((bear_count - bull_count) * 0.2, 1.0)
            return ('bearish', strength)
        return ('neutral', 0.0)
    
    def analyze_news_sentiment(self, news: List[Dict]) -> Dict:
        """Analyze news sentiment, volume, and momentum"""
        if not news:
            return {
                'sentiment': 'neutral',
                'sentiment_strength': 0.0,
                'news_volume': 0,
                'latest_headlines': [],
                'momentum_signal': 0
            }
        
        # Analyze sentiment of recent headlines
        sentiments = [
            self.analyze_headline_sentiment(article['headline'])
            for article in news[:5]  # Focus on most recent news
        ]
        
        # Calculate overall sentiment
        bull_strength = sum(strength for sent, strength in sentiments if sent == 'bullish')
        bear_strength = sum(strength for sent, strength in sentiments if sent == 'bearish')
        
        # Determine momentum signal (-1 to 1)
        momentum = bull_strength - bear_strength
        
        # News volume analysis
        news_volume = len(news)
        volume_multiplier = min(news_volume/1.5, 2.0)  # Cap at 2.0
        
        return {
            'sentiment': 'bullish' if momentum > 0 else 'bearish' if momentum < 0 else 'neutral',
            'sentiment_strength': abs(momentum),
            'news_volume': news_volume,
            'volume_multiplier': volume_multiplier,
            'latest_headlines': [article['headline'] for article in news[:3]],
            'momentum_signal': momentum * volume_multiplier  # Combine sentiment with volume
        }

    def get_position_sizing_factor(self, symbol: str) -> float:
        """Get position sizing multiplier based on news (0.5 to 2.0)"""
        news = self.get_company_news(symbol, hours_back=4)
        analysis = self.analyze_news_sentiment(news)
        
        # Base multiplier on sentiment strength and news volume
        base_factor = 1.0
        if analysis['sentiment'] == 'bullish':
            base_factor += analysis['sentiment_strength']
        elif analysis['sentiment'] == 'bearish':
            base_factor -= analysis['sentiment_strength']
            
        # Adjust for news volume
        volume_boost = min((analysis['news_volume'] - 5) * 0.1, 0.5)
        factor = base_factor + volume_boost
        
        # Constrain between 0.5 and 2.0
        return max(0.5, min(2.0, factor))

    def should_skip_trade(self, symbol: str, signal: float) -> Tuple[bool, float]:
        """
        Check if we should skip trade and return position sizing factor
        Returns: (should_skip, position_size_multiplier)
        """
        news = self.get_company_news(symbol, hours_back=2)
        analysis = self.analyze_news_sentiment(news)
        
        # Get base position sizing factor
        pos_factor = self.get_position_sizing_factor(symbol)
        
        # Skip if extremely high news volume with conflicting sentiment
        if analysis['news_volume'] > 20 and (
            (signal > 0 and analysis['sentiment'] == 'bearish') or
            (signal < 0 and analysis['sentiment'] == 'bullish')
        ):
            print(f"Skipping {symbol}: High news volume with conflicting sentiment")
            return True, pos_factor
            
        # Skip if very strong sentiment opposite to our signal
        if (signal > 0 and analysis['momentum_signal'] < -0.8) or \
           (signal < 0 and analysis['momentum_signal'] > 0.8):
            print(f"Skipping {symbol}: News sentiment conflicts with signal")
            return True, pos_factor
            
        return False, pos_factor

    def print_latest_news(self, symbol: str):
        """Print latest news with sentiment analysis"""
        news = self.get_company_news(symbol, hours_back=4)
        if news:
            analysis = self.analyze_news_sentiment(news)
            print(f"\nLatest news for {symbol}:")
            print(f"Sentiment: {analysis['sentiment'].upper()} (strength: {analysis['sentiment_strength']:.2f})")
            print(f"News Volume: {analysis['news_volume']} articles")
            print(f"Momentum Signal: {analysis['momentum_signal']:.2f}")
            
            for article in news[:3]:
                sentiment, strength = self.analyze_headline_sentiment(article['headline'])
                print(f"- {article['headline']}")
                print(f"  {datetime.datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Sentiment: {sentiment.upper()} ({strength:.2f})")
        else:
            print(f"No recent news for {symbol}")