import queue
from threading import Thread
from anomalias import detector, log
import pandas as pd

logger = log.logger('Series')


class DataFrame(Thread):
    def __init__(self, df_id, df_len, api, zbx_alert=True):
        Thread.__init__(self, name=df_id)
        self.id = df_id
        self.ad = detector.Detector(df_len=df_len)

        self.__exit = False
        self.__paused = False
        self.__observations = queue.Queue()
        self.__api = api
        self.__zbx_alert = zbx_alert

        logger.info('New series created, id %s', self.id)

    def run(self):
        while not self.__exit:
            obs = self.__observations.get()
            if not obs.empty:
                try:
                    df, anomalies, anomaly_th_lower, anomaly_th_upper = self.ad.detect(obs)

                    if isinstance(anomalies, pd.Series):
                        anomalies = anomalies.to_frame()[[0] * df.shape[-1]]
                        anomalies.columns = df.columns

                    logger.debug('Data for detection:')
                    logger.debug('\n %s', df)
                    logger.debug('Anomalies:')
                    logger.debug('\n %s', anomalies)

                    self.__api.write(df, anomalies, anomaly_th_lower, anomaly_th_upper, measurement=self.id,
                                     zbx_alert=self.__zbx_alert)

                except Exception as e:
                    logger.error('%s', e, exc_info=True)

    def append(self, obs):
        self.__observations.put(obs)

    def exit(self, bol=False):
        self.__exit = bol
        self.__api.close()

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False
