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
    steps1 = [
        ("rollingagg", DoubleRollingAggregate(
            agg="mean",
            window=5,
            center=True,
            diff="l1"
        )),
        ("Interquantile_ad", InterQuartileRangeAD(c=param))
    ]
    return Pipeline(steps1)


def detector_3(param=0.01):
    return QuantileAD(high=1-param, low=param/2)


def detector_4(param=5):
    return InterQuartileRangeAD(c=param)


def detector_5(param=0.01):
    return GeneralizedESDTestAD(alpha=param)


# Point/Level/Volatility Change
def detector_6(param=0.1):
    steps1 = [
        ("rollingagg", DoubleRollingAggregate(
            agg="mean",
            window=3,
            center=True,
            diff="l1"
        )),
        ("Interquantile_ad", PersistAD(c=param))
    ]
    return Pipeline(steps1)


def detector_7(param=3.0):
    return LevelShiftAD(c=param, side='both', window=5)


def detector_8(param=3.0):
    return VolatilityShiftAD(c=param, side='negative', window=5)


# Seasonal - Regression
def detector_9(param=0.05):
    eps = param
    steps2 = [
        ("ETS", ClassicSeasonalDecomposition(freq=288)),
        ("quantile_ad", QuantileAD(high=1.0 - eps, low=eps))
    ]
    return Pipeline(steps2)


def detector_10(param=3.0):
    return SeasonalAD(c=param,freq=288, side="both")


def detector_11(param=3.0):
    return AutoregressionAD(n_steps=30, step_size=288, c=param)


# Otros
def detector_12(param=3): ## id 9
    return PcaAD(c=param, k=3)


def detector_13(param=3):
    return MinClusterDetector(KMeans(n_clusters=param))


detector_lst = [detector_0, detector_1, detector_2, detector_3, detector_4,
                detector_5, detector_6, detector_7, detector_8, detector_9,
                detector_10, detector_11, detector_12, detector_13]


params = [0.00203, 0.00809, 8.700000000000001, 0.00102, 1.8181818181818181, 0.9898989899999999,
          6.565656565656566, 8.080808080808081, 0.4444444444444445, 0.001011090909090909,
          5.05050505050505, 7.878787878787879, 16.95959595959596, 8]


class AdtkAD:
    def __init__(self, models_idx, nvot=1, **kargs):
        self.__df_train = []
        logger.info('Creating ADTK model.')
        self.__model = [detector_lst[i] for i in models_idx]
        self.__nvot = nvot
        self.__params = [params[i] for i in models_idx]

    def fit(self, train_data):
        logger.info('Fitting model.')
        # self.__df_train = train_data
        # self.__model.fit(train_data)

    def detect(self, observations):

        for i, detector in enumerate(self.__model):
            print(i)
            anom = detector(param=params[i]).fit_detect(observations)
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

        idx_anomaly = idx_anomaly[idx_anomaly['anomaly'] == 1].astype(float)
        idx_anomaly = idx_anomaly.reset_index().set_index('_time')

        idx_anomaly.index.rename(observations.index.name, inplace=True)

        anomaly_th_lower = None
        anomaly_th_upper = None

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper
