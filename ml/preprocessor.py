from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from Utils.logger_config import logger


FILL_VALUE_LOGICAL_NA = -999
"""Predefined fill value for SimpleImputer. Used to impute predefined features with an
unrealistic value."""


def clean_column_name(col_name):
    """Cleans a column name by replacing special characters with underscores or other
    valid characters."""
    col_name = str(col_name)
    col_name = col_name.replace("]", ")").replace("[", "(").replace(" ", "_")
    col_name = col_name.replace(">", "").replace("<", "").replace("=", "")
    return col_name


class PreprocessingService(BaseEstimator, TransformerMixin):
    """A class used to preprocess data for machine learning models.

    Attributes
        max_n_unique (int): The maximum number of unique values a categorical feature can
            have before it is dropped.
        zero_impute_cols (list): A list of columns that should be imputed with zero.
        logical_na_impute_cols (list): A list of columns that should be imputed with a
            logical NA, e.g. number_of_days_since_last_positive.
        use_multivariate_imputer (bool): Whether to use the IterativeImputer for numerical
            columns.
        missing_indicator_threshold (float): The threshold for missing values in a column
            before adding a missing indicator column to the data.

    Methods
        fit_preprocessor()
            Fits the preprocessor on the training data.
        transform(X)
            Transforms the input data using the preprocessor.
    """

    def __init__(
        self,
        max_n_unique: int = 50,
        zero_impute_cols: list = [],
        logical_na_impute_cols: list = [],
        use_multivariate_imputer: bool = False,
        missing_indicator_threshold: Optional[float] = None,
        outlier_factor: float = 1.5,
    ) -> None:
        self.feature_names = None
        self.preprocessor = None
        self.max_n_unique = max_n_unique
        self.zero_impute_cols = zero_impute_cols
        self.logical_na_impute_cols = logical_na_impute_cols
        self.use_multivariate_imputer = use_multivariate_imputer
        self.missing_indicator_threshold = missing_indicator_threshold
        self.dropped_cols = []
        self.outlier_factor = outlier_factor

    def clean_column_names(self, X):
        """Cleans the column names of the DataFrame."""
        X = X.copy()
        X.columns = [clean_column_name(col) for col in X.columns]
        return X

    def fit_preprocessor(self, X_train):
        """Fits the preprocessor on the training data.
        The dataset is devided into numerical and categorical features.

        Note: Featue names may be in a different order after preprocessing.

        Args:
            X_train (pandas.DataFrame): The training data to fit the preprocessor on.
        """
        X_train = self.check_data(X_train)
        if X_train.shape[1] < 2:
            raise ValueError("X_train must have at least 2 rows.")
        self.required_feature_names = X_train.columns.tolist()
        logger.debug(f"required features: {self.required_feature_names}")

        num_transformers = self.numeric_columns_handler(X_train)
        cat_transformers = self.categorical_columns_handler(X_train)
        if self.dropped_cols:
            X_train.drop(self.dropped_cols, axis=1, inplace=True, errors="ignore")

        preprocessor = ColumnTransformer(
            transformers=num_transformers + cat_transformers,
            remainder="passthrough",
        )
        self.preprocessor = preprocessor.fit(X_train)
        self.set_feature_names()

    def fit(self, X, y=None):
        """Fits the preprocessor on the training data. Needed for the sklearn pipeline.

        Parameters
            X : pandas.DataFrame
                The training data to fit the preprocessor on.
            y : pandas.Series, optional
                The target data to fit the preprocessor on.
        """
        self.fit_preprocessor(X)
        return self

    def transform(self, X):
        """Transforms the input data using the preprocessor.

        Parameters
            X : pandas.DataFrame
                The data to transform.

        Returns
            pandas.DataFrame
                The transformed data.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fit yet.")
        X = self.check_data(X)
        if self.dropped_cols:
            logger.info(
                "The preprocessor droped the following columns: "
                f"{self.dropped_cols}. They will be dropped from the input data."
            )
            X.drop(self.dropped_cols, axis=1, inplace=True, errors="ignore")
        return self.preprocessor.transform(X)

    def get_feature_names_out(self):
        """Gets the feature names after preprocessing."""
        return self.feature_names

    def set_feature_names(self):
        """Sets the feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fit yet.")
        if self.feature_names is not None:
            return self.feature_names

        f_names_all = []
        for transformer in self.preprocessor.transformers_:
            pipeline = transformer[1]
            for x, step in pipeline.steps[::-1]:
                if hasattr(step, "get_feature_names_out"):
                    try:
                        f_names = step.get_feature_names_out()
                        f_names_all.extend(f_names)
                        break  # only take the feature names of the last step
                    except NotFittedError:
                        continue  # skip unused transformers

        self.feature_names = [clean_column_name(f) for f in f_names_all]
        logger.debug(f"set feature names: {self.feature_names}")
        return self.feature_names

    def inverse_transform(self, X: np.ndarray, out="np"):
        """Inverse transforms a preprocessed dataset to a more intepretable form.
        Inverse transformations only include the StandardScaler for the numeric
        features.

        Parameters
            X : numpy.ndarray
                The preprocessed data to transform.
            out : str, optional
                The output format of the transformed data. Default is "np".

        Returns
            numpy.ndarray or pandas.DataFrame
                The inverse transformed data.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fit yet.")
        if out not in ["np", "pd"]:
            raise ValueError("`out` must be 'np' or 'pd'.")
        # check if X is a numpy array
        if not isinstance(X, np.ndarray):
            logger.warning("Input data is not a np.array. Converting to np.array.")
            X = np.array(X)
        # slices holds information about the indices of the different feature types
        X_inv = X.copy()
        transformers = self.preprocessor.transformers_
        slices = self.preprocessor.output_indices_
        # map slices to transformer names

        for transformer in transformers:
            name, pipeline, _ = transformer
            scaler = pipeline.steps[-1][-1]
            slice_ = slices.get(name, slice(None, None, None))
            try:
                X_inv[:, slice_] = scaler.inverse_transform(X_inv[:, slice_])
            except (NotFittedError, ValueError):
                # NotFittetError: Unused transformer, e.g. if logical_na_numerical not set
                # ValueError: Transformer is no numerical transfomer, just skip
                continue
        if out == "pd":
            return pd.DataFrame(X_inv, columns=self.feature_names)
        return X_inv

    @staticmethod
    def outlier_removal(X, factor) -> pd.DataFrame:
        """Removes outliers from the input data using the interquartile range (IQR)
        method.

        Parameters:
            X : array-like
                Input data to remove outliers from.
            factor : float
                A factor to multiply the IQR by to determine the outlier threshold.

        Returns:
            X : pandas.DataFrame
                A copy of the input data with outliers replaced with NaN values.
        """
        X = pd.DataFrame(X).copy()
        col_names = X.columns
        if factor is None:
            return X
        for i in range(X.shape[1]):
            x = pd.Series(X.iloc[:, i]).copy()
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            X.iloc[(X.iloc[:, i] < lower_bound), i] = lower_bound
            X.iloc[(X.iloc[:, i] > upper_bound), i] = upper_bound
        X.columns = col_names
        return X

    @staticmethod
    def check_inf(X: pd.DataFrame) -> pd.DataFrame:
        """Check if the input data contains infinite values and replaces them with NaN.

        Parameters
            X (array-like): Input data to check for infinite values.

        Returns
            (bool): True if the input data contains infinite values, False otherwise.
        """
        X = X.copy()
        if X.isin([np.inf, -np.inf]).sum().sum():
            logger.warning("Infinite values detected in the data. Replacing with NaN.")
            X = X.replace([np.inf, -np.inf], np.nan)
        return X

    @staticmethod
    def check_booleans(X: pd.DataFrame) -> pd.DataFrame:
        """Check if the input data contains boolean values and converts them to strings.

        Parameters
            X (array-like): Input data to check for boolean values.

        Returns
            X (array-like): Input data with boolean values converted to strings.
        """
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == "bool" or X[col].dtype == "boolean":
                X[col] = X[col].astype(str)
        return X

    def check_datatypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform all columns that are not numeric to string."""
        X = X.copy()
        for col in X.columns:
            if X[col].dtype not in ["float", "int"]:
                X[col] = X[col].astype(str)
        return X

    def check_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform a number of checks on the input data."""
        X = X.copy()
        X = self.check_inf(X)
        X = self.check_booleans(X)
        X = self.check_datatypes(X)
        # replace pd.NA with np.nan to avoid errors in the pipeline
        X = X.where(pd.notnull(X), None)
        return X

    def numeric_columns_handler(self, X: pd.DataFrame) -> List[tuple]:
        X = X.copy()
        numerical_cols = X.select_dtypes(include=["float", "int"]).columns.tolist()

        # columns, that should be imputed with zero, e.g. counts
        zero_impute = [
            col
            for col in numerical_cols
            if any(keyword in col for keyword in self.zero_impute_cols)
        ]

        # columns, that should be imputed with a logical NA, e.g. amounts
        logical_na_impute = [
            col
            for col in numerical_cols
            if any(keyword in col for keyword in self.logical_na_impute_cols)
        ]
        logical_na_impute = list(set(logical_na_impute) - set(zero_impute))

        # all other numerical columns
        default_impute = list(
            set(numerical_cols) - set(zero_impute) - set(logical_na_impute)
        )
        logger.debug(
            f"zero_impute: {zero_impute}; logical_na_impute: {logical_na_impute}; "
            f"default_impute: {default_impute}"
        )

        outlier_remover = FunctionTransformer(
            self.outlier_removal, kw_args={"factor": self.outlier_factor}
        )

        if self.use_multivariate_imputer:
            numeric_imputer = IterativeImputer(max_iter=10, random_state=0)
        else:
            numeric_imputer = SimpleImputer(strategy="median", add_indicator=False)

        zero_transformer = Pipeline(
            steps=[
                ("outlier_remover", outlier_remover),
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")  # to preserve column names

        logical_na_transformer = Pipeline(
            steps=[
                ("outlier_remover", outlier_remover),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=FILL_VALUE_LOGICAL_NA,
                        add_indicator=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")  # to preserve column names

        default_transformer = Pipeline(
            steps=[
                ("imputer", numeric_imputer),
                ("outlier_remover", outlier_remover),
                ("scaler", StandardScaler()),
            ]
        ).set_output(transform="pandas")  # to preserve column names

        transformers = [
            ("default_numerical", default_transformer, default_impute),
            ("zero_numerical", zero_transformer, zero_impute),
            ("logical_na_numerical", logical_na_transformer, logical_na_impute),
        ]

        if self.missing_indicator_threshold is not None:
            transformers = [
                self._missing_indicator_split(X, transformer_tuple)
                for transformer_tuple in transformers
            ]
            transformers = [i for sub in transformers for i in sub]  # flatten list
        return transformers

    def categorical_columns_handler(self, X: pd.DataFrame) -> List[tuple]:
        X = X.copy()
        # categorical features
        categorical_cols = X.select_dtypes(exclude=["float", "int"]).columns.tolist()
        # check how many unique values each categorical feature has
        for col in categorical_cols:
            n_unique = X[col].nunique()
            if n_unique > self.max_n_unique:
                logger.warning(
                    f"Feature {col} has {n_unique} unique values. Feature is dropped."
                )
                self.dropped_cols.append(col)
        # drop categorical features with too many unique values
        if self.dropped_cols:
            categorical_cols = [
                col for col in categorical_cols if col not in self.dropped_cols
            ]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
            ]
        ).set_output(transform="pandas")  # to preserve column names
        return [("categorical", categorical_transformer, categorical_cols)]

    def _missing_indicator_split(
        self, X: pd.DataFrame, transformer_tuple: tuple
    ) -> List[tuple]:
        """Split the transformer tuple into two based on the missing indicator threshold."""
        id_ = transformer_tuple[0]
        transformer, cols = transformer_tuple[1], transformer_tuple[2]

        missing_indicator_cols = [
            col for col in cols if self._should_add_indicator(X, col)
        ]
        non_missing_indicator_cols = [
            col for col in cols if col not in missing_indicator_cols
        ]

        # todo: add check that transformer is an imputer
        missing_transformer = clone(transformer)
        missing_transformer.steps[0][1].add_indicator = True
        return [
            (id_, transformer, non_missing_indicator_cols),
            (id_ + "threshold_exceeded", missing_transformer, missing_indicator_cols),
        ]

    def _should_add_indicator(self, data, col):
        """Check if the missing indicator should be added based on missing data threshold."""
        return data[col].isnull().mean() > self.missing_indicator_threshold

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(preprossesor={self.preprocessor})>"