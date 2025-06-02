from typing import Optional, List, Dict, Union
import time
import pandas as pd
import numpy as np
from Utils.logger_config import logger
from Utils.custom_types import DataFrameLike
from ml.base_transformer import BaseTransformer

def generate_lag_features(df: pd.DataFrame, columns: List[str], lags: int) -> pd.DataFrame:
    """
    Generates lag features for the specified columns.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        columns (list): List of column names to create lag features for.
        lags (int): Number of lag periods.
        
    Returns:
        pd.DataFrame: DataFrame with lag features (same index as df).
    """
    lag_df = pd.DataFrame(index=df.index)
    for col in columns:
        for i in range(1, lags + 1):
            lag_df[f"{col}_lag_{i}"] = df[col].shift(i)
    return lag_df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_technical_indicators(df: pd.DataFrame,
                                   feature_groups: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
    """
    Calculates additional technical indicators and adds them to the DataFrame.
    
    Available groups (default all True):
      - ma: Simple Moving Averages (20 and 50 period)
      - ema: Exponential Moving Averages (20 and 50 period)
      - rsi: Relative Strength Index (14 period)
      - macd: MACD (using EMA12, EMA26, and EMA signal of 9)
      - bollinger: Bollinger Bands (20 period, 2 standard deviations)
      - atr: Average True Range (14 period)
      - volatility: 14-day volatility (rolling std of daily returns)
      - momentum: Rate of change over 10 days
    """
    if feature_groups is None:
        feature_groups = {
            "ma": True,
            "ema": True,
            "rsi": True,
            "macd": True,
            "bollinger": True,
            "atr": True,
            "volatility": True,
            "momentum": True
        }
        
    if feature_groups.get("ma", False):
        df["sma_20"] = df["close"].rolling(window=20).mean().shift(1)
        df["sma_50"] = df["close"].rolling(window=50).mean().shift(1)
            
    if feature_groups.get("ema", False):
        # EMA is recursive, but if you want to ensure causality you can shift by one
        df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean().shift(1)
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean().shift(1)
            
    if feature_groups.get("rsi", False):
        df["rsi_14"] = compute_rsi(df["close"], period=14).shift(1)
            
    if feature_groups.get("macd", False):
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean().shift(1)
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean().shift(1)
        df["macd"] = (df["ema12"] - df["ema26"]).shift(1)
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().shift(1)
            
    if feature_groups.get("bollinger", False):
        sma20 = df["close"].rolling(window=20).mean().shift(1)
        rstd = df["close"].rolling(window=20).std().shift(1)
        df["bollinger_top"] = sma20 + 2 * rstd
        df["bollinger_bot"] = sma20 - 2 * rstd
            
    if feature_groups.get("atr", False):
        df["tr1"] = abs(df["high"] - df["low"])
        df["tr2"] = abs(df["high"] - df["close"].shift(1))
        df["tr3"] = abs(df["low"] - df["close"].shift(1))
        df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
        df["atr_14"] = df["tr"].rolling(window=14).mean().shift(1)
        df.drop(["tr1", "tr2", "tr3", "tr"], axis=1, inplace=True)
            
    if feature_groups.get("volatility", False):
        df["daily_return"] = df["close"].pct_change().shift(1)
        df["volatility_14"] = df["daily_return"].rolling(window=14).std().shift(1)
        df.drop("daily_return", axis=1, inplace=True)
            
    if feature_groups.get("momentum", False):
        df["roc_10"] = df["close"].pct_change(periods=10).shift(1)
        
    return df

class NormalTransformer(BaseTransformer):
    """
    A transformer that processes raw data and creates an ML-ready dataset.
    It filters the data by date, computes a target variable (next-day % change in close price,
    optionally binary), adds lag features, and can also add technical indicator features.
    
    Parameters:
      - lags (int): Number of lag periods.
      - target_type (str): 'binary' for binary target or 'regression' for continuous percentage change.
      - lag_columns (list): Columns to generate lag features for.
      - feature_groups (dict): Dictionary of booleans controlling which technical indicator groups to include.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 lags: int = 1,
                 target_type: str = "binary",
                 lag_columns: Optional[List[str]] = None,
                 feature_groups: Optional[Dict[str, bool]] = None
                 ) -> None:
        super().__init__()
        self.data = self._transform_date_types(df)
        self.lags = lags
        self.target_type = target_type.lower()
        self.feature_groups = feature_groups
        if lag_columns is None:
            self.lag_columns = ['close', 'volume', 'positive', 'negative', 'neutral']
        else:
            self.lag_columns = lag_columns

    def create_dataset(self,
                       start_date: str,
                       end_date: str,
                       ) -> pd.DataFrame:
        """Creates an ML-ready dataset with target, lag features, and technical indicators.
        
        Args:
            start_date (str): Start date (YYYY-MM-DD) for filtering.
            end_date (str): End date (YYYY-MM-DD) for filtering.
            
        Returns:
            pd.DataFrame: Processed dataset.
        """
        logger.info(f"Creating dataset from {start_date} to {end_date}.")
        try:
            df = self.transformer_payload["df"]
        except KeyError:
            raise ValueError('`transform` method must be called first.')
        
        # Filter rows by date and sort by index
        df = df[(df.index >= pd.to_datetime(start_date)) &
                (df.index <= pd.to_datetime(end_date))].sort_index()
        
        # Compute target variable: next-day percentage change in close price
        df['price_change'] = df['close'].pct_change().shift(-1)
        if self.target_type == "binary":
            threshold = 1e-6
            df['target'] = df['price_change'].apply(lambda x: 1 if x > threshold else 0)
        elif self.target_type == "regression":
            df['target'] = df['price_change']
        else:
            raise ValueError("target_type must be either 'binary' or 'regression'")
        
        # Generate lag features
        lag_df = generate_lag_features(df, self.lag_columns, self.lags)
        df = pd.concat([df, lag_df], axis=1)
        
        # Optionally add technical indicators if feature_groups is provided
        if self.feature_groups is not None:
            df = calculate_technical_indicators(df, self.feature_groups)
        
        # Drop rows with NaNs (due to shifting, etc.)
        df = df.dropna()
        
        # Optionally, further process data types using data_type_handler
        df = self.data_type_handler(df)
        
        logger.info(f"Dataset created with shape {df.shape}.")
        return df

    def transform(self,
                  df: DataFrameLike,
                  training_scoring_mode: str = "training",
                  columns_include: Optional[List[str]] = None,
                  date: Optional[pd.Timestamp] = None
                  ) -> None:
        logger.info("Transforming data...")
        start_time = time.time()
        if training_scoring_mode == 'training':
            pass
        else:
            raise ValueError(
                "training_scoring_mode should be either 'training' or 'scoring', not "
                f"{training_scoring_mode}")
        
        # Save transformed dataframe into transformer_payload
        self.transformer_payload = {"df": df}
        logger.info(f"Data transformation complete. Took {round(time.time()-start_time, 4)} s.")
        return None


