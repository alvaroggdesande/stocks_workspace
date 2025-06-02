import pickle
import numpy as np
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ml.base_transformer import BaseTransformer
from ml.base_model import ModelHandler
from ml.preprocessor import PreprocessingService
from Utils.custom_types import DataFrameLike
from Utils.logger_config import logger


@dataclass
class PipelineData:
    """A class to store and manage various pipeline-related attributes."""
    X_train: Optional[DataFrameLike]
    X_test: Optional[DataFrameLike]
    X_pred: Optional[DataFrameLike]
    y_train: Optional[DataFrameLike]
    y_test: Optional[DataFrameLike]
    y_pred: Optional[DataFrameLike]
    y_scores: Optional[np.ndarray]


class Result:
    """A class to store and manage various result-related attributes.

    Attributes:
        - transformer (BaseTransformer): The transformer used.
        - preprocessor (PreprocessingService): The preprocessor used.
        - model_handler (ModelHandler): The model handler used to train and predict.

    Methods:
        - save(path, io_service): Serialize the current instance and save it to a specified
            file path.
        - load(path, io_service): Deserialize an instance stored in a file and replace the
            attributes of the
    """

    def __init__(self,
                 model_type: str,
                 preprocessor: PreprocessingService,
                 model_handler: ModelHandler,
                 transformer: Optional[BaseTransformer] = None,
                 model_evaluation: Optional[dict] = None,
                 pipeline_data: Optional[PipelineData] = None,
                 result_label: Optional[str] = None,
                 features: Optional[list] = None,
                 additional_info: Optional[dict] = None
                 ) -> None:
        """Initializes an empty Result object with attributes set to None."""
        self.model_type = model_type
        self.transformer = transformer
        self.preprocessor = preprocessor
        self.model_handler = model_handler
        self.pipeline_data = pipeline_data
        self.model_evaluation = model_evaluation
        self.result_label = result_label
        self.features = features
        self.additional_info = additional_info
        self.date_created = datetime.now()
        self.id = self._create_id()

    def as_dict(self):
        """Return the current instance as a dictionary."""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "transformer": self.transformer,
            "preprocessor": self.preprocessor,
            "model_handler": self.model_handler,
            "pipeline_data": self.pipeline_data,
            "model_evaluation": self.model_evaluation,
            "result_label": self.result_label,
            "date_created": self.date_created}
    
    def set_custom_id(self, custom_id: str) -> "Result":
        """Set a custom id for the current instance. This can be useful when automating
        the creation and loading of Result instances.

        Args:
            custom_id (str): The custom id to set.
        """
        self.id = custom_id
        return self

    def _create_id(self) -> str:
        """Create a unique id for the current instance.

        Returns:
            int: A unique id.
        """
        return f"result_{self.date_created.strftime('%Y-%m-%d %H:%M:%S')}_{self.model_type}_{self.model_handler.model_alias}_{self.result_label}_{id(self)}"

