from typing import Union

import pandas as pd
import numpy as np
import scipy.sparse as sp

# Custom type for dataframes and series.
DataFrameLike = Union[pd.DataFrame, pd.Series, np.ndarray, sp.spmatrix]