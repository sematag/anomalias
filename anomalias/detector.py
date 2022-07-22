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
        self.__dataFrame = pd.DataFrame([])
        self.__anom = pd.DataFrame([],dtype='boolean')
        self.__dataFrame_last = pd.DataFrame([])
        self.__anom_last = pd.DataFrame([],dtype='boolean')

        self.__training = False
        self.__paused = False

    def ssm_ad(self, th, endog, model_type, **kwargs):
        with self.__available:
            # Model
            logger.info('Creating Anomaly Detector.')
            self.__model = SSM_AD(th, endog, model_type, **kwargs)
            self.__available.notify()

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
            # Series Update
            if not observations.empty:
                self.__dataFrame_last = observations[~observations.index.isin(self.__dataFrame.index)]
                self.__dataFrame = pd.concat([self.__dataFrame, self.__dataFrame_last]).iloc[-self.__len:]
                # Detection
                self.__anom_last = self.__model.detect(self.__dataFrame_last).astype('boolean')
                self.__anom = pd.concat([self.__anom, self.__anom_last]).iloc[-self.__len:]
            self.__available.notify()

    def get_detection(self):
        with self.__available:
            self.__available.wait()
            return self.__dataFrame_last, self.__anom_last