import backtrader as bt

class AdvancedData(bt.feeds.PandasData):
    # Add extra lines for sentiment data
    lines = ('positive', 'negative', 'neutral',
             'prediction')
    params = (
        # Default parameters for standard fields:
        ('date', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        # Map the extra columns by name (Sentiment)
        ('positive', -1),
        ('negative', -1),
        ('neutral', -1),
        # (Prediction)
        ('prediction', -1),
    )