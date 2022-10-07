"""
Time Series Models
"""
from statsmodels.tsa.statespace.api import SARIMAX, ExponentialSmoothing

from anomalias import log
import numpy as np
import pandas as pd

logger = log.logger('ssmad')


class SsmAD:
    def __init__(self, df, th, params=None, **kwargs):
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

        logger.debug('%s', self.__model_fit.fittedvalues)

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
            self.__model_fit.apped(endog=df, refit=False)
            self.__init = False
        else:
            try:
                self.__model_fit = self.__model_fit.extend(df)
            except ValueError as ve:
                logger.debug('%s', ve)
                self.__model_fit.append(endog=df, refit=False)

        prediction = self.__model_fit.get_prediction()

        predicted_mean = prediction.predicted_mean
        logger.debug('################')
        logger.debug('%s', predicted_mean)
        predicted_mean = predicted_mean[predicted_mean.index.isin(df.index)]

        predicted_sigma = np.sqrt(prediction.var_pred_mean)
        predicted_sigma = predicted_sigma[predicted_sigma.index.isin(df.index)]

        predicted_mean.columns = df.columns
        predicted_sigma.columns = df.columns

        if isinstance(predicted_mean, pd.Series):
            predicted_mean = predicted_mean.to_frame()
            predicted_sigma = predicted_sigma.to_frame()

        idx_anomaly = np.abs(df.values - predicted_mean.values) > (self.__th * predicted_sigma).values

        idx_anomaly = pd.DataFrame(idx_anomaly,
                                   columns=df.columns, index=df.index)

        anomaly_th_lower = predicted_mean.values - self.__th * predicted_sigma.values
        anomaly_th_upper = predicted_mean.values + self.__th * predicted_sigma.values

        anomaly_th_lower = pd.DataFrame(anomaly_th_lower,
                                        columns=df.columns, index=df.index)
        anomaly_th_upper = pd.DataFrame(anomaly_th_upper,
                                        columns=df.columns, index=df.index)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

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

        idx_anomaly = self.__model_fit.resid > self.__th

        self.__model_fit.fittedvalues.fillna(0.0, inplace=True)

        anomaly_th_lower = self.__model_fit.fittedvalues - pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)
        anomaly_th_upper = self.__model_fit.fittedvalues + pd.Series([self.__th] * len(self.__model_fit.fittedvalues),
                                                                     index=self.__model_fit.fittedvalues.index)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

    def set_th(self, th):
        self.__th = th
