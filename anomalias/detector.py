import pandas as pd
import anomalias.log as log
from threading import Condition
from anomalias.tsmodels import SsmAD
from anomalias.adtk import AdtkAD

logger = log.logger('Detector')


class Detector:
    def __init__(self, df_len):
        # Series
        self.__len = df_len
        self.__available = Condition()
        self.__dataFrame = pd.DataFrame([])
        self.__anomalies = pd.DataFrame([], dtype='boolean')
        self.__anomaly_th_upper = pd.DataFrame([])
        self.__anomaly_th_lower = pd.DataFrame([])
        self.__dataFrame = pd.DataFrame([])
        self.__model = None
        self.__allObs = False

        self.__training = False
        self.__paused = False

    def set_model(self, model):
        with self.__available:
            self.__model = model
            self.__available.notify()

    def fit(self, df):
        with self.__available:
            self.__model.fit(df)
            anomalies, anomaly_th_lower, anomaly_th_upper = self.__model.detect(df)
            anomalies = anomalies.astype('boolean')
            self.__available.notify()

            return anomalies, anomaly_th_lower, anomaly_th_upper

    def detect(self, observations):
        with self.__available:
            # Series Update
            df = observations[~observations.index.isin(self.__dataFrame.index)]
            self.__dataFrame = pd.concat([self.__dataFrame, df]).iloc[-self.__len:]

            logger.debug('detector.py: call to detect(), data:')
            logger.debug('\n %s', df)

            # Detection
            if not df.empty:
                if self.__allObs:
                    anomalies, anomaly_th_lower, anomaly_th_upper = self.__model.detect(self.__dataFrame)
                    anomalies = anomalies.loc[df.index.intersection(anomalies.index)]
                    if anomaly_th_lower is not None:
                        anomaly_th_lower = anomaly_th_lower.loc[df.index]
                    if anomaly_th_upper is not None:
                        anomaly_th_upper = anomaly_th_upper.loc[df.index]
                else:
                    anomalies, anomaly_th_lower, anomaly_th_upper = self.__model.detect(df)

                anomalies = anomalies.astype('boolean')
                self.__anomalies = pd.concat([self.__anomalies, anomalies]).iloc[-self.__len:]

                if anomaly_th_lower is not None and anomaly_th_upper is not None:
                    self.__anomaly_th_lower = pd.concat([self.__anomaly_th_lower, anomaly_th_lower]).iloc[-self.__len:]
                    self.__anomaly_th_upper = pd.concat([self.__anomaly_th_upper, anomaly_th_upper]).iloc[-self.__len:]
            else:
                anomalies = df.copy()
                anomaly_th_lower = None
                anomaly_th_upper = None

            self.__available.notify()

            return df, anomalies, anomaly_th_lower, anomaly_th_upper

    def get_detection(self):
        return self.__dataFrame.copy(), self.__anomalies.copy()

    def set_all_obs_detect(self, bol):
        self.__allObs = bol

