"""
Time Series Models
"""
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.statespace.api import SARIMAX, ExponentialSmoothing

from anomalias import log
import numpy as np
import pandas as pd

logger = log.logger('ssmad')


class SsmAD:
    def __init__(self, th, df, model_type, params=None, **kwargs):
        logger.info('Setting SARIMAX model.')
        self.__th = th
        self.__model = SARIMAX(df, **kwargs)
        self.__init = True

        if params is None:
            self.__model_fit = self.__model.fit()
            logger.debug('%s', self.__model_fit.summary())
        else:
            self.__model.update(params=params)
            self.__model_fit = self.__model.filter()

    def fit(self, df):
        # Fit params
        self.__model_fit.apply(endog=df, refit=True)
        logger.debug('%s', self.__model_fit.summary())
        logger.info('Model fitted. Params: %s', self.__model_fit.params)

    def fit_detect(self, df):
        # Fit params
        self.__model_fit.apply(endog=df, refit=True)
        logger.debug('%s', self.__model_fit.summary())
        logger.info('Model fitted. Params: %s', self.__model_fit.params)

        pred = self.__model_fit.get_prediction()

        prediction_error = df - pred.predicted_mean
        sigma = np.sqrt(pred.var_pred_mean)
        idx_anom = np.abs(prediction_error) > self.__th * sigma

        return idx_anom

    def detect(self, df):
        if self.__init:
            self.__model_fit.apply(endog=df, refit=False)
            self.__init = False
        else:
            self.__model_fit = self.__model_fit.extend(df)

        prediction = self.__model_fit.get_prediction()

        predicted_mean = prediction.predicted_mean
        predicted_sigma = np.sqrt(prediction.var_pred_mean)
        logger.info('1111: %s', predicted_mean)
        logger.info('2222: %s', predicted_sigma)
        logger.info('3333: %s', df)

        idx_anomaly = np.abs(df - predicted_mean) > self.__th * predicted_sigma
        logger.info('4444: %s', df)
        anomaly_th_lower = prediction.predicted_mean - self.__th * predicted_sigma
        anomaly_th_upper = prediction.predicted_mean + self.__th * predicted_sigma


        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

    def get_pred_vars(self):
        pred = self.__model_fit.get_prediction()
        return pred.predicted_mean, self.__th * np.sqrt(pred.var_pred_mean)

    def set_th(self, th):
        self.__th = th


class ExpAD:
    def __init__(self, th, df, model_type, **kwargs):
        self.__th = th
        self.__model = ExponentialSmoothing(df, **kwargs)
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
        self.__model = ExponentialSmoothing(df, seasonal=self.__model.seasonal_periods)

        with self.__model.fix_params({'smoothing_level': self.__model_fit.params['smoothing_level'],
                                      'smoothing_trend': self.__model_fit.params['smoothing_trend'],
                                      'smoothing_seasonal': self.__model_fit.params['smoothing_seasonal']}):
            self.__model_fit = self.__model.fit()

        # prediction_error = self.__model_fit.resid
        idx_anomaly = self.__model_fit.resid > self.__th

        self.__model_fit.fittedvalues.fillna(0.0, inplace=True)

        anomaly_th_lower = self.__model_fit.fittedvalues - pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)
        anomaly_th_upper = self.__model_fit.fittedvalues + pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

    def set_th(self, th):
        self.__th = th
