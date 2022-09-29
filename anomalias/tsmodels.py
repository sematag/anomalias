"""
Time Series Models
"""
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.api import SARIMAX

from anomalias import log
import numpy as np
import pandas as pd

logger = log.logger('ssmad')


class SsmAD:
    def __init__(self, th, df, model_type, params=None, **kwargs):
        if model_type == 'SARIMAX':
            logger.info('Creating SARIMAX model.')
            self.__df_train = df
            self.__th = th
            self.__model = SARIMAX(self.__df_train, **kwargs)
            if params is None:
                self.__model_fit = self.__model.fit()
                logger.info('Model fitted. \n %s', self.__model_fit.summary())
            else:
                self.__model.update(params=params)
                self.__model_fit = self.__model.filter()
        else:
            logger.error('Model type not found: %s', model_type)
            raise ValueError('Model type not found')

    def train(self, train_data, params=None):
        logger.info('Fitting model...')
        self.__df_train = train_data
        if params is None:
            self.__model_fit.apply(endog=train_data, refit=True)
            logger.info('Model fitted. \n %s', self.__model_fit.summary())
        else:
            self.__model.update(params=params)
            self.__model_fit.apply(endog=train_data)
        logger.info('Model fitted. \n %s', self.__model_fit.summary())

    def detect(self, observations):
        self.__model_fit = self.__model_fit.extend(observations)
        prediction = self.__model_fit.get_prediction()

        prediction_error = observations - prediction.predicted_mean
        sigma = np.sqrt(prediction.var_pred_mean)
        idx_anomaly = np.abs(prediction_error) > self.__th * sigma
        anomaly_th_lower = prediction.predicted_mean - self.__th * sigma
        anomaly_th_upper = prediction.predicted_mean + self.__th * sigma

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper


class ExpAD:
    def __init__(self, th, df, model_type, **kwargs):
        self.__th = th
        self.__model = ExponentialSmoothing(df, **kwargs,
                                            trend="add",
                                            seasonal="add",
                                            use_boxcox=True,
                                            initialization_method="estimated")
        self.__model_fit = None

    def fit(self, df=None):
        # Fit params
        self.__model_fit = self.__model.fit()

        logger.debug('%s', self.__model_fit.summary())
        logger.debug('Model fitted. Params: %s', self.__model_fit.params)

    def fit_detect(self, df):
        # Fit params
        self.__model_fit = self.__model.fit()

        logger.debug('%s', self.__model_fit.summary())
        logger.debug('Model fitted. Params: %s', self.__model_fit.params)

        prediction_error = self.__model_fit.resid
        idx_anom = self.__model_fit.resid > self.__th

        return idx_anom

    def detect(self, df):
        self.__model = ExponentialSmoothing(df,
                                            seasonal_periods=self.__model.seasonal_periods,
                                            trend="add",
                                            seasonal="add")
        with self.__model.fix_params({'smoothing_level': self.__model_fit.params['smoothing_level'],
                                      'smoothing_trend': self.__model_fit.params['smoothing_trend'],
                                      'smoothing_seasonal': self.__model_fit.params['smoothing_seasonal']}):
            self.__model_fit = self.__model.fit()

        prediction_error = self.__model_fit.resid
        idx_anomaly = self.__model_fit.resid > self.__th

        self.__model_fit.fittedvalues.fillna(0.0, inplace=True)

        anomaly_th_lower = self.__model_fit.fittedvalues - pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)
        anomaly_th_upper = self.__model_fit.fittedvalues + pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

    def set_th(self, th):
        self.__th = th
