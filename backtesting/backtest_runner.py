import backtrader as bt
from strategies.base_strategy import BaseStrategy
from strategies.optimized_strategy import OptimizedStrategy
from strategies.buy_and_hold_strategy import BuyAndHoldStrategy

from backtesting.advanced_data import AdvancedData

from .backtest_reporting import generate_report

class BacktestRunner:
    @staticmethod
    def run_backtest(data, stock_ticker, start_date, end_date
                     , strategy=BuyAndHoldStrategy, strategy_params=None, cash=None):
        """
        Run Backtrader backtest with the provided data.

        Args:
            data (pd.DataFrame): Merged stock and sentiment data.
            stock_ticker (str): Stock Ticker name.
            start_date (str): Start date for backtesting.
            end_date (str): End date for backtesting.
            strategy (class): Strategy class to run.
            strategy_params (dict, optional): Additional parameters for the strategy.
        """
        cerebro = bt.Cerebro()
        # Convert data to Backtrader format
        #data_feed = bt.feeds.PandasData(dataname=data)
        data_feed = AdvancedData(dataname=data)
        cerebro.adddata(data_feed)

        # Add strategy with parameters if provided, otherwise add default strategy
        if strategy_params:
            cerebro.addstrategy(strategy, **strategy_params)
        else:
            cerebro.addstrategy(strategy)

        # Set initial cash and commission
        cerebro.broker.set_cash(cash)
        cerebro.broker.setcommission(commission=0.001)

        # Add built-in analyzers
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
        cerebro.addanalyzer(bt.analyzers.SQN)
        #cerebro.addanalyzer(bt.analyzers.VWR)  # Optional
        cerebro.addanalyzer(bt.analyzers.PyFolio)

        thestrats = cerebro.run()
        thestrat = thestrats[0]

        # Get the report as DataFrames
        df_report, df_trades = generate_report(cerebro, stock_ticker, start_date, end_date, thestrat)
        
        # You can now return the DataFrames, or save them, plot them, etc.
        return df_report, df_trades
