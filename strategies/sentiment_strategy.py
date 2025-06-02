# sentiment_strategy.py
from .base_strategy import BaseStrategy
import backtrader as bt

class SentimentStrategy(BaseStrategy):
    """
    A strategy that uses sentiment data to drive trade decisions.
    
    The strategy computes a positivity rate as:
        positivity_rate = positive / (positive + negative)
    It goes long if the positivity rate exceeds the specified threshold,
    provided there is a minimum number of news articles. It exits the position
    if the negativity rate exceeds a threshold.
    
    Assumes that the data feed includes these additional columns:
      - positive
      - negative
      - neutral (optional, for reference)
    """
    params = (
        ("min_news_count", 1),          # Minimum news articles required for a valid signal
        ("positivity_threshold", 0.6),    # If positivity rate >= 0.6, then buy
        ("negativity_threshold", 0.6),    # If negativity rate >= 0.6, then sell
        # Optionally, you can add a window for smoothing the sentiment ratio
    )
    
    def __init__(self):
        super().__init__()
        # You can add additional indicators here if needed (e.g., a moving average of the sentiment ratio)
    
    def next(self):
        # Retrieve sentiment counts from the current bar
        pos = self.data.positive[0]
        neg = self.data.negative[0]
        # Calculate the total number of news articles (ignoring neutral if you prefer)
        total_news = pos + neg
        
        # Compute positivity and negativity rates if there is enough news data
        if total_news >= self.params.min_news_count:
            positivity_rate = pos / total_news
            negativity_rate = neg / total_news
        else:
            # If not enough news, default to neutral
            positivity_rate = 0.5
            negativity_rate = 0.5

        # Debug (optional):
        # print(f"Date: {self.data.datetime.date(0)}, Positivity: {positivity_rate:.2f}, Negativity: {negativity_rate:.2f}, Total News: {total_news}")

        # If not in a position, check if bullish sentiment triggers a buy
        if not self.position:
            if positivity_rate >= self.params.positivity_threshold:
                self.buy()
        else:
            # If in a position, check if bearish sentiment triggers an exit
            if negativity_rate >= self.params.negativity_threshold:
                self.sell()
                
    def stop(self):
        # Ensure any open position is closed at the end of the backtest.
        if self.position:
            self.close()
