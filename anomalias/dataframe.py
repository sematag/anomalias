import queue
from threading import Thread
from anomalias import detector, log

logger = log.logger('Series')


class DataFrame(Thread):
    def __init__(self, id, len, api):
        Thread.__init__(self, name=id)
        self.id = id
        self.ad = detector.Detector(len=len)

        self.__exit = False
        self.__paused = False
        self.__observations = queue.Queue()
        self.__api = api

        logger.info('New series created, id %s', self.id)

    def run(self):
        while not self.__exit:
            obs = self.__observations.get()
            if not obs.empty:
                dataFrame, anomalies = self.ad.detect(obs)

                logger.debug('Data for detection:')
                logger.debug('\n %s', dataFrame)
                logger.debug('Anomalies:')
                logger.debug('\n %s', anomalies)

                df_out = dataFrame.rename(columns={0: 'label'})
                df_out = df_out[anomalies]
                logger.debug('api.py: anomalies to write:')
                logger.debug('\n %s', df_out)

                self.__api.write(record=df_out[['label']], data_frame_measurement_name=self.id)


    def append(self, obs):
        self.__observations.put(obs)

    def exit(self, bol=False):
        self.__exit = bol
        self.__api.close()

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False


