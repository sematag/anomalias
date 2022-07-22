import queue
from threading import Thread
from anomalias import detector as anomdetect, log

logger = log.logger('Series')


class DataFrame(Thread):
    def __init__(self, id, len):
        Thread.__init__(self)
        self.id = id
        self.ad = anomdetect.Detector(len=len)

        self.__exit = False
        self.__paused = False
        self.__observations = queue.Queue()

        logger.info('New series created, id %s', self.id)

    def run(self):
        while not self.__exit:
            obs = self.__observations.get()
            self.ad.detect(obs)

    def append(self, obs):
        self.__observations.put(obs)

    def exit(self, bol=False):
        self.__exit = bol

    def pause(self):
        self.__paused = True

    def resume(self):
        self.__paused = False


