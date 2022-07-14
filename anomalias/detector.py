import pandas as pd
import anomalias.log as log
from threading import Condition
from anomalias.tsmodels import SSM_AD
from anomalias.adtk import Adtk_AD

logger = log.logger('Detector')


class Detector:
    def __init__(self, len):
        # Series
        self.__len = len
        self.__available = Condition()
        # self.endog = np.zeros(self.__len)
        self.dataFrame = pd.DataFrame([])
        # self.idx_anom = [False] * len
        self.idx_anom = pd.DataFrame([])

        self.__training = False
        self.__paused = False

    def ssm_ad(self, th, endog, model_type, **kwargs):
        with self.__available:
            # Model
            logger.info('Creating Anomaly Detector.')
            self.__model = SSM_AD(th, endog, model_type, **kwargs)

    def adtk_ad(self, model_type, **kargs):
        with self.__available:
            self.__model = Adtk_AD(model_type, **kargs)
        self.__available.notify()

    def fit(self, serie):
        with self.__available:
            self.__model.fit(serie)
        self.__available.notify()

    def detect(self, observations):
        with self.__available:
            self.__available.wait(1)
            # Series Update
            self.dataFrame = pd.concat([self.dataFrame, observations]).iloc[-self.__len:]
            # Detection
            idx_anom = self.__model.detect(observations)
            self.idx_anom = pd.concat([self.idx_anom, idx_anom]).iloc[-self.__len:]