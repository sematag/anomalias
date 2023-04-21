from anomalias import log
import pandas as pd

from sklearn.cluster import KMeans, Birch
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

from adtk.detector import *
from adtk.transformer import *
from adtk.aggregator import *
from adtk.pipe import Pipeline, Pipenet
import numpy as np

nro_detectores = 15

logger = log.logger('adtk')


# OutlierDetector
def detector_0(param=0.05):
    return OutlierDetector(LocalOutlierFactor(contamination=param))


def detector_1(param=0.05):
    return OutlierDetector(IsolationForest(contamination=param, warm_start=True))


def detector_2(param=5):
    return InterQuartileRangeAD(c=param)


def detector_3(param=0.01):
    return QuantileAD(high=1-param, low=param)


def detector_4(param=0.01):
    return GeneralizedESDTestAD(alpha=param)


# Point/Level/Volatility Change
def detector_5(param=0.1):
    return PersistAD(c=param)


def detector_6(param=3.0):
    return LevelShiftAD(c=param, side='both', window=5)


def detector_7(param=3.0):
    return VolatilityShiftAD(c=param, side='negative', window=5)


# Seasonal - Regression
def detector_8(params):
    eps = params[0]
    steps2 = [
        ("ETS", ClassicSeasonalDecomposition(freq=params[1])),
        ("quantile_ad", QuantileAD(high=1.0 - eps, low=eps))
    ]
    return Pipeline(steps2)


def detector_9(params):
    return SeasonalAD(c=params[0], freq=params[1], side="both")


# Otros
def detector_10(params):
    return PcaAD(c=params[0], k=params[1])


def detector_11(param=3):
    return MinClusterDetector(KMeans(n_clusters=param))


detector_lst = [detector_0, detector_1, detector_2, detector_3, detector_4,
                detector_5, detector_6, detector_7, detector_8, detector_9,
                detector_10, detector_11]


class AdtkAD:
    def __init__(self, models_idx, params,  nvot=1):
        self.__df_train = []
        logger.info('Creating ADTK model.')
        self.__model = [detector_lst[i](params[i]) for i in models_idx]
        self.__nvot = nvot

    def fit(self, train_data):
        logger.info('Fitting model.')
        self.__df_train = train_data
        for i, detector in enumerate(self.__model):
            logger.debug('Fit Adtk id: %s.', i)
            self.__model[i].fit(train_data)

    def detect(self, observations):

        for i, detector in enumerate(self.__model):
            logger.debug('Detect Adtk id: %s.', i)
            anom = self.__model[i].detect(observations)
            anom.index.rename('_time', inplace=True)

            if isinstance(anom, pd.Series):
                anom = anom.to_frame()[[0] * observations.shape[-1]]
                anom.columns = observations.columns

            anom = anom.stack().to_frame().reset_index().set_index('_time')

            anom.columns = ['series', 'label' + str(i)]

            anom = anom[anom['label' + str(i)] == 1]

            anom['label' + str(i)] = anom['label' + str(i)].astype(int)
            if i == 0:
                idx_anomaly = anom
            else:
                idx_anomaly = pd.merge(idx_anomaly, anom, how="outer", on=["_time", "series"])

        idx_anomaly = idx_anomaly.set_index(['series'], append=True)

        idx_anomaly['nvot'] = idx_anomaly.sum(axis=1)

        idx_anomaly['anomaly'] = idx_anomaly['nvot'] >= self.__nvot

        idx_anomaly = idx_anomaly[idx_anomaly['anomaly'] == 1].astype(bool)
        idx_anomaly = idx_anomaly['anomaly']
        idx_anomaly = idx_anomaly.reset_index().set_index('_time')

        idx_anomaly.index.rename(observations.index.name, inplace=True)

        idx_anomaly = idx_anomaly.pivot(columns='series', values='anomaly')

        anomaly_th_lower = None
        anomaly_th_upper = None

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper
