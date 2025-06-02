import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Sequence, Any

import numpy as np
import pandas as pd
from datetime import timedelta
#from deprecated import deprecated
from sklearn.model_selection import train_test_split

from Utils.logger_config import logger
from Utils.custom_types import DataFrameLike


class BaseTransformer(ABC):
    def __init__(self, date_dtype: str = 'datetime64[ns]'):
        super().__init__()
        self._date_dtype = date_dtype
        self._date_tz = None

    @abstractmethod
    def transform(**kwargs: Any) -> None:
        pass

    @abstractmethod
    def create_dataset(**kwargs: Any) -> pd.DataFrame:
        pass

    def _transform_date_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Given a pandas DataFrame and a base type, converts all columns containing date/time
        data to the specified type. If the DataFrame is None, returns None.
        Note: Internal method, date_dtype should be set in __init__.
        """
        # extracts dtype and timezone info from type string
        # if type string has no tz info, np dtype is used
        try:
            date_type = pd.DatetimeTZDtype.construct_from_string(
                self._date_dtype)
            tz = date_type.tz  # type: ignore
        except TypeError:  # type string has no tz info
            date_type = np.dtype(self._date_dtype)
            tz = None
        date_cols = BaseTransformer._find_date_types(df)
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(
                    df[col]).dt.tz_localize(tz).astype(date_type)
            # TypeError: Already tz-aware, use tz_convert to convert.
            except TypeError:
                df[col] = pd.to_datetime(df[col]).astype(date_type)
        self._date_tz = tz
        return df

    @staticmethod
    def prepare_dataset(
            training_dataset: pd.DataFrame,
            columns_exclude: list = [],
            target_column: str = 'ResponseValue',
            sort_col: str = 'ResponseTimestamp'):
        """Prepare the dataset for training and testing."""
        if target_column in training_dataset.columns:
            X = training_dataset.drop([target_column], axis=1)
            y = training_dataset[target_column]
        else:
            logger.warn(
                (f"No target column `{target_column}` found in dataset. "
                 "If this is an unseen prediction dataset, this is expected "
                 "and can be ignored."))
            X, y = training_dataset, None
        if sort_col in X.columns:
            X.sort_values(by=[sort_col], inplace=True)
        elif sort_col is not None:
            logger.warn(
                (f"No sort column `{sort_col}` found in dataset. Sorting will be skipped. "
                 "Beware that `shuffle=False` may not work as expected in the "
                 "`create_train_test_split` method."))
        if columns_exclude:
            # ignore errors if columns to drop do not exist
            to_drop = list(set(columns_exclude))
            logger.debug(f"Dropping columns: {to_drop}")
            X = X.drop(to_drop, axis=1, errors="ignore")
        return X, y

    @staticmethod
    def create_train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42):
        """Create a train/test split of the data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state)
        return X_train, X_test, y_train, y_test

    @staticmethod
    def create_windows(
        df_window_raw: pd.DataFrame,
        window_enddate: Union[pd.Timestamp, Sequence[pd.Timestamp]],
        date_col: str,
        agg_col: str,
        time_windows: dict,
        feat_names: List[str] = ["total_count", "avg_amount", "total_amount"],
        agg_on: List[str] = ["ContactId"]
    ) -> pd.DataFrame:
        """Create time windows for a given dataframe based on a specified end date and
        time window periods.

        Args:
            - df_window_raw:  pd.DataFrame, the input dataframe.
            - window_enddate: Union[pd.Timestamp, pd.Series], the end date for the time
                windows. Can be a single date or a series of dates. If a series of
                dates, the length of the series should match the length of the input 
                dataframe.
            - date_col: str, the name of the column containing the date information.
            - agg_col: str, the name of the column to aggregate on.
            - time_windows: dict, a dictionary containing the time windows to create.
                Format should be as follows: {window_name: days}.
                Example: {'3month': 90, '6month': 180, '1year': 365, '2year': 730}
            - feat_names: List[str], a list of feature names for the aggregated columns.
            - agg_on: list, the names of the columns to group by.

        Returns:
            pd.DataFrame, the input dataframe with additional columns for each window.
        """
        if not isinstance(df_window_raw, pd.DataFrame):
            raise TypeError('df_window_raw should be a pd.DataFrame')
        if not isinstance(window_enddate, (pd.Timestamp, pd.Series)):
            raise TypeError(
                'window_enddate should be a pd.Timestamp or pd.Series')
        if isinstance(window_enddate, pd.Series):
            if len(window_enddate) != len(df_window_raw):
                raise ValueError(
                    'window_enddate and df_window_raw should have the same length')
        if not isinstance(date_col, str):
            raise TypeError('date_col should be a string')
        if not isinstance(time_windows, dict):
            raise TypeError('time_windows should be a dict')
        if not all(isinstance(key, str) for key in time_windows.keys()):
            raise TypeError('time_windows keys should be strings')
        if not all(isinstance(value, int) and value > 1 for value in time_windows.values()):
            raise TypeError(
                'time_windows values should be integers greater than 1')
        if not isinstance(feat_names, list):
            raise TypeError('feat_names should be a list')
        if not all(isinstance(feat, str) for feat in feat_names):
            raise TypeError('feat_names should be a list of strings')
        if len(set(feat_names)) != 3:
            raise ValueError('feat_names should have 3 distinct elements')
        if not isinstance(agg_on, list):
            raise TypeError('agg_on should be a list')
        if not all(isinstance(col, str) for col in agg_on):
            raise TypeError('agg_on should be a list of strings')

        dfs = []  # create new dfs with calculations for each time window
        for window, days in time_windows.items():
            response_timestamp = window_enddate - timedelta(days)
            df_window = df_window_raw[df_window_raw[date_col]
                                      > response_timestamp]
            # calculate metrics for time window
            df = df_window.groupby(agg_on).agg(**{
                f'{feat_names[0]}_{window}': (agg_col, 'count'),
                f'{feat_names[1]}_{window}': (agg_col, 'mean'),
                f'{feat_names[2]}_{window}': (agg_col, 'sum')
            }).reset_index()
            dfs.append(df)
        # merge all window dfs
        df_window_raw = pd.merge(df_window_raw, dfs[0], on=agg_on, how='left')
        for df in dfs[1:]:
            df_window_raw = pd.merge(df_window_raw, df, on=agg_on, how='left')
        return df_window_raw

    @staticmethod
    def data_type_handler(df: pd.DataFrame):
        """This method takes a pandas DataFrame as input and performs a number of operations
        on handling date types.

        Data types are unified to avoid issues with downstream operations. Boolean values are
        first handled as strings to remove the pd.NaN dtypes. 
        """
        # Select float, int, bool, and object columns
        df_float = df.select_dtypes(include=[float])  # type: ignore
        df_int = df.select_dtypes(include=[int])  # type: ignore
        df_bool = df.select_dtypes(include=[bool])  # type: ignore
        df_obj = df.select_dtypes(exclude=[np.number, "datetimetz", np.datetime64])  # type: ignore

        # Convert float columns to float64
        df = df.astype({col: 'float64' for col in df_float.columns})

        # Fill NaN values in int columns with a default (e.g., 0) before converting to int64
        df = df.fillna({col: 0 for col in df_int.columns})
        df = df.astype({col: 'int64' for col in df_int.columns})

        # Convert bool columns to string
        df = df.astype({col: 'string' for col in df_bool.columns})

        # Handle object columns with categorical types
        for col in df_obj.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].cat.add_categories(['NaN'])
        
        # Fill NaN in object columns with 'NaN'
        df = df.fillna({col: 'NaN' for col in df_obj.columns})

        # Convert boolean columns to object type
        df = df.astype({col: 'object' for col in df_bool.columns})

        logger.info("Data types adjusted.")
        return df

    @staticmethod
    def _find_date_types(df):
        """Finds all columns in a pandas DataFrame that contain dates."""
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    # ignore warning due to not specifying format
                    # this is expected to fail for non-date columns
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[col])
                    date_cols.append(col)
                except ValueError:
                    pass
        return date_cols

    def __repr__(self):
        return f"<{self.__class__.__name__}()>"