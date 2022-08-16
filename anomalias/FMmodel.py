from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from anomalias import log
from polylearn import FactorizationMachineRegressor

logger = log.logger('FMmodel')

class FactorizationMachineAnomalyDetector:
    """
    Factorization Machine Anomaly Detector
    """
    def __init__(self, **kwargs):
        logger.info('Creating FM model.')
        self.__model = FactorizationMachineRegressor(**kwargs)

    def train(self, train_data, **kwargs):
        logger.info('Fitting model...')
        self.__model.train(train_data, **kwargs)
        logger.info(f'Model fitted.\n{self.__model.summary()}')

    def detect(self, observations):
        logger.info('Detecting anomalies...')
        return self.__model.detect(observations)


class PreprocessingFMTransformer(BaseEstimator, TransformerMixin):
    """
    Preprocess time series data to use on FM model
    """
    def __init__(self, window_size: int = 5, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size
        logger.debug('Creating preprocessing FM transformer.')

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        assert X.index.dtype == datetime, 'Data must be a pandas DataFrame with datetime index.'
        assert y.name in X.columns, 'Target must be a column of data.'
        return self

    def transform(self, X: pd.Dataframe, y: pd.Series = None):
        X_out = np.empty(((len(X) - self.window_size)//self.step_size, X.shape[1]*self.window_size + 4))
        for i in range(self.window_size, len(X), self.step_size):
            # TODO
            pass

