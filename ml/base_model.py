from abc import ABC, abstractmethod
from typing import Any

from Utils.custom_types import DataFrameLike


class ModelHandler(ABC):
    def __init__(self, params: dict, *args, **kwargs):
        self.params = params
        self.model = None
        self.model_alias = None

    @abstractmethod
    def fit(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(self, **kwargs: Any) -> DataFrameLike:
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(params={self.params})>"