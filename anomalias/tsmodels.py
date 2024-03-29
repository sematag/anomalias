"""
Time Series Models
"""
from statsmodels.tsa.statespace.api import SARIMAX, ExponentialSmoothing

from anomalias import log
import numpy as np
import pandas as pd

logger = log.logger('ssmad')


class SsmAD:
    def __init__(self, df, th_sigma, th_lower=None, th_upper=None, pre_log=False, log_cnt=1, **kwargs):
        logger.info('Setting SARIMAX model.')
        self.__th_sigma = th_sigma
        self.__th_lower = th_lower
        self.__th_upper = th_upper

        if pre_log:
            self.__pre = lambda x: np.log(x + log_cnt)
            self.__inv_pre = lambda x: np.exp(x) - log_cnt
        else:
            self.__pre = lambda x: x
            self.__inv_pre = lambda x: x

        self.__model = SARIMAX(self.__pre(df), **kwargs)
        self.__init = True
        self.__trained = False
        self.__model_fit = None

    def fit(self, df):
        # Fit params
        if not self.__trained:
            self.__model_fit = self.__model.fit()
            self.__trained = True
        else:
            self.__model_fit.apply(endog=self.__pre(df), refit=True)
        self.__init = True
        logger.debug('%s', self.__model_fit.summary())
        logger.info('Model fitted. Params: %s', self.__model_fit.params)

    def detect(self, df):
        if self.__init:
            self.__model_fit = self.__model_fit.apply(endog=self.__pre(df), refit=False)
            self.__init = False
        else:
            try:
                self.__model_fit = self.__model_fit.extend(self.__pre(df))
            except ValueError as ve:
                logger.debug('%s', ve)
                self.__model_fit = self.__model_fit.apply(endog=self.__pre(df), refit=False)

        prediction = self.__model_fit.get_prediction()

        predicted_mean = prediction.predicted_mean
        predicted_mean = predicted_mean[predicted_mean.index.isin(df.index)]

        predicted_sigma = np.sqrt(prediction.var_pred_mean)
        predicted_sigma = predicted_sigma[predicted_sigma.index.isin(df.index)]

        predicted_mean.columns = df.columns
        predicted_sigma.columns = df.columns

        if isinstance(predicted_mean, pd.Series):
            predicted_mean = predicted_mean.to_frame()
            predicted_sigma = predicted_sigma.to_frame()

        anomaly_th_lower = predicted_mean.values - self.__th_sigma * predicted_sigma.values
        anomaly_th_upper = predicted_mean.values + self.__th_sigma * predicted_sigma.values

        anomaly_th_lower = self.__inv_pre(pd.DataFrame(anomaly_th_lower,
                                                       columns=df.columns, index=df.index))
        anomaly_th_upper = self.__inv_pre(pd.DataFrame(anomaly_th_upper,
                                                       columns=df.columns, index=df.index))

        if self.__th_lower is not None:
            anomaly_th_lower.clip(lower=self.__th_lower, inplace=True)
        else:
            anomaly_th_lower.clip(lower=np.nanmin(anomaly_th_lower[anomaly_th_lower != -np.inf]), inplace=True)

        if self.__th_upper is not None:
            anomaly_th_upper.clip(upper=self.__th_upper, inplace=True)
        else:
            anomaly_th_upper.clip(upper=np.nanmax(anomaly_th_upper[anomaly_th_upper != np.inf]), inplace=True)

        idx_anomaly = (df > anomaly_th_upper) | (df < anomaly_th_lower)

        # idx_anomaly = pd.DataFrame(idx_anomaly,
        # columns=df.columns, index=df.index)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper

    def set_th_sigma(self, th_sigma):
        self.__th_sigma = th_sigma

    def set_th_lower(self, th_lower):
        self.__th_lower = th_lower

    def set_th_upper(self, th_upper):
        self.__th_upper = th_upper


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
