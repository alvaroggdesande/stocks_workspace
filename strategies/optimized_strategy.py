import backtrader as bt
from .base_strategy import BaseStrategy

class OptimizedStrategy(BaseStrategy):
    """
    Custom Backtrader strategy with advanced technical indicators.

    Parameters:
    - fast_ma (int): Period for the fast moving average.
    - slow_ma (int): Period for the slow moving average.
    - rsi_period (int): Period for the Relative Strength Index (RSI).
    - rsi_oversold (float): RSI level considered as oversold for buying.
    - rsi_overbought (float): RSI level considered as overbought for selling.
    - bollinger_window (int): Period for Bollinger Bands.
    - bollinger_dev (float): Standard deviation factor for Bollinger Bands.
    - ema_window (int): Period for Exponential Moving Average (EMA).
    - envelopes_ema_window (int): Period for EMA used in Envelopes indicator.
    - envelopes_percentage (float): Percentage for Envelopes indicator.
    - macd_short_window (int): Short window period for MACD.
    - macd_long_window (int): Long window period for MACD.
    - macd_signal_window (int): Signal window period for MACD.
    - stochastic_k_window (int): Window period for Stochastic Oscillator %K.
    - stochastic_d_window (int): Window period for Stochastic Oscillator %D.
    """

    params = (
        ("fast_ma", 20),
        ("slow_ma", 50),
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("bollinger_window", 20),
        ("bollinger_dev", 2),
        ("ema_window", 20),
        ("envelopes_ema_window", 20),
        ("envelopes_percentage", 5),
        ("macd_short_window", 12),
        ("macd_long_window", 26),
        ("macd_signal_window", 9),
        ("stochastic_k_window", 14),
        ("stochastic_d_window", 3),
    )

    def __init__(self):
        """
        Initializes the AdvancedStrategy.

        Creates and initializes the required technical indicators:
        - fast_ma: Fast Simple Moving Average (SMA)
        - slow_ma: Slow Simple Moving Average (SMA)
        - rsi: Relative Strength Index (RSI)
        - bollinger: Bollinger Bands
        - ema: Exponential Moving Average (EMA)
        - macd: Moving Average Convergence Divergence (MACD)
        - stochastic: Stochastic Oscillator
        """
        super().__init__()
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_ma)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_ma)
        self.rsi = bt.indicators.RelativeStrengthIndex(period=self.params.rsi_period)
        self.bollinger = bt.indicators.BollingerBands(self.data.close, period=self.params.bollinger_window,
                                                      devfactor=self.params.bollinger_dev)
        self.ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.params.ema_window)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.params.macd_short_window,
                                       period_me2=self.params.macd_long_window,
                                       period_signal=self.params.macd_signal_window)
        self.stochastic = bt.indicators.Stochastic(self.data, period=self.params.stochastic_k_window,
                                                   period_dfast=self.params.stochastic_d_window)
        
    def next(self):
        """
        Executes the trading logic on each iteration.

        Buys if conditions for a bullish trend are met:
        - RSI is below the oversold threshold
        - MACD is positive
        - Close price is above the lower Bollinger Band
        - Close price is above the EMA

        Sells if conditions for a bearish trend are met:
        - RSI is above the overbought threshold
        - MACD is negative
        - Close price is below the upper Bollinger Band
        - Close price is below the EMA
        """
        buy_condition = (
                self.rsi < self.params.rsi_oversold and
                self.macd.macd > 0 and
                self.data.close > self.bollinger.lines.bot and
                self.data.close > self.ema
            )

        sell_condition = (
                self.rsi > self.params.rsi_overbought or
                self.macd.macd < 0 or
                self.data.close < self.bollinger.lines.top or
                self.data.close < self.ema
            )

        if not self.position:
                if buy_condition:
                    self.buy()
        else:
                if sell_condition:
                    self.sell()

    def stop(self):
    # If there's an open position at the end, close it
        if self.position:
            self.close()
