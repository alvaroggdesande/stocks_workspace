# ml_prediction_strategy.py
from strategies.base_strategy import BaseStrategy
import backtrader as bt

class MLRegressionStrategy(BaseStrategy):
    """
    A strategy that uses an ML model's predicted next-day return to generate trading signals.
    
    Assumes that the data feed includes an extra column called 'prediction' which represents
    the predicted next-day percentage change in the close price (regression output).
    
    Parameters:
      - buy_threshold (float): The threshold above which a buy signal is generated.
      - sell_threshold (float): The threshold below which a sell signal is generated.
    """
    params = (
        ("buy_threshold", 0.001),
        ("sell_threshold", -0.001),
    )
    
    def __init__(self):
        super().__init__()
        # No additional indicators needed; we rely on the provided 'prediction' column.
    
    def next(self):
        # Retrieve the prediction value from the current bar.
        try:
            pred = float(self.data.prediction[0])
        except Exception as e:
            print("Error retrieving prediction:", e)
            return
            
        """# Debug: print current date and prediction
        #current_date = self.data.datetime.date(0)
        #print(f"Date: {current_date}, Prediction: {pred}")"""

        # If not in a position, and prediction is above the buy threshold, then buy.
        if not self.position:
            if pred > float(self.params.buy_threshold):
                self.buy()
        else:
            # If already in a position and prediction falls below the sell threshold, then sell.
            if pred < float(self.params.sell_threshold):
                self.sell()
    
    def stop(self):
        # Force close any open position at the end of the backtest.
        if self.position:
            self.close()
