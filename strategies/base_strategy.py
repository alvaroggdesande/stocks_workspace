# base_strategy.py
import backtrader as bt

class BaseStrategy(bt.Strategy):
    def __init__(self):
        # Initialize common attributes
        self.trade_details = []
        self.entry_time = None
        self.entry_price = None
        self.exit_time = None
        self.exit_price = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_time = self.data.datetime.datetime(0)
                self.entry_price = order.executed.price
                print("Buy order executed at", self.entry_time, self.entry_price)
            elif order.issell():
                self.exit_time = self.data.datetime.datetime(0)
                self.exit_price = order.executed.price
                print("Sell order executed at", self.exit_time, self.exit_price)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_details.append({
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'entry_price': self.entry_price,
                'exit_price': self.exit_price,
                'pnl': trade.pnl,
                'size': trade.size,
            })
            print("Trade closed:")
            print(f"  Entry Time: {self.entry_time}")
            print(f"  Exit Time: {self.exit_time}")
            print(f"  Entry Price: {self.entry_price:.2f}")
            print(f"  Exit Price: {self.exit_price:.2f}")
            print(f"  Trade Size: {trade.size}")
            print(f"  PnL: {trade.pnl:.2f}")

    def stop(self):
        print("stop() called.")
        # If we have bought and no trade details are recorded, force record the exit
        if self.has_bought and not self.trade_details:
            # Use the data from the last bar as exit details
            self.exit_time = self.data.datetime.datetime(0)
            self.exit_price = self.data.close[0]
            pnl = (self.exit_price - self.entry_price) * 1  # assuming size of 1
            self.trade_details.append({
                'entry_time': self.entry_time,
                'exit_time': self.exit_time,
                'entry_price': self.entry_price,
                'exit_price': self.exit_price,
                'pnl': pnl,
                'size': 1,
            })
