import warnings

import xgboost
import pandas as pd

from ml.base_model import ModelHandler
from Utils.logger_config import logger


NUM_BOOST_ROUND = 100
VERBOSE_EVAL = 10
EARLY_STOPPING_ROUNDS = 10


class XGBoostModelHandler(ModelHandler):

    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.params = None
        self.model = None
        self.model_alias = "xgb"

    def fit(self, d_train, d_test, y_train, y_test, params, verbose_eval=VERBOSE_EVAL, **kwargs):
        dm_train, dm_test = self._prepare_training_data(d_train, d_test, y_train, y_test)
        logger.info("Fitting Model: XGBoost")
        model = xgboost.train(
            params,
            dm_train,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dm_train, "train"), (dm_test, "test")],
            verbose_eval=verbose_eval,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            **kwargs)

        self.params = params
        self.model = model
        logger.debug(f"Model {self} fitted successfully.")
        return model

    def cv(self, d_train, y_train, params, nfold=5, num_boost_round=NUM_BOOST_ROUND, **kwargs) -> pd.DataFrame:
        dm_train = xgboost.DMatrix(d_train, label=y_train,
                                   feature_names=self.feature_names)

        # set CV evaluation metrics based on task type
        task_type = params.get('objective', '')
        if ('binary:' in task_type) or ('multi:' in task_type):
            metrics = 'auc'
        elif 'reg:' in task_type:
            metrics = 'rmse'
        else:
            logger.warning(
                f"Task type {task_type} not recognized. Setting metrics to 'auc'.")
            metrics = 'auc'

        cv_results = xgboost.cv(
            params,
            dm_train,
            num_boost_round=num_boost_round,
            nfold=nfold,
            metrics=metrics,
            as_pandas=True,
            seed=123,
            **kwargs)
        return cv_results  # type: ignore

    def _prepare_training_data(self, d_train, d_test, y_train, y_test):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            dm_train = xgboost.DMatrix(
                d_train, label=y_train, feature_names=self.feature_names)
            dm_test = xgboost.DMatrix(
                d_test, label=y_test, feature_names=self.feature_names)
        return dm_train, dm_test

    @staticmethod
    def prepare_data_for_prediction(X, feature_names) -> xgboost.DMatrix:
        return xgboost.DMatrix(X, feature_names=feature_names)

    def predict(self, X_pred_tr, **kwargs):
        if self.model is None:
            raise ValueError("Model not fitted yet.")
        dm_pred = self.prepare_data_for_prediction(X_pred_tr, self.feature_names)
        predict_score = self.model.predict(dm_pred, **kwargs)
        self.predict_score = predict_score
        logger.info(f"Prediction completed {predict_score.shape}")
        return predict_score