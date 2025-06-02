import backtrader as bt

class PredictionData(bt.feeds.PandasData):
    # Add extra lines for sentiment data
    lines = ('prediction',)
    params = (
        # Default parameters for standard fields:
        ('date', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        # Map the extra columns by name (assuming your DataFrame columns are named exactly as these)
        ('prediction',0)
    )