# buy_and_hold_strategy.py
from .base_strategy import BaseStrategy
import backtrader as bt

class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.has_bought = False

    def next(self):
        if not self.position and not self.has_bought:
            self.buy()
            self.has_bought = True
        elif len(self) == len(self.data) - 1:
            self.close()

