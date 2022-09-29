from adtk.detector import SeasonalAD
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from anomalias import log

logger = log.logger('adtk')


class AdtkAD:
    def __init__(self, model_type, **kargs):
        self.__df_train = []
        if model_type == 'SeasonalAD':
            logger.info('Creating SeasonalAD model.')
            self.__model = SeasonalAD()
        elif model_type == 'MinClusterAD':
            logger.info('Creating MinClusterAD model.')
            self.__model = MinClusterDetector(KMeans(**kargs))
        else:
            logger.error('Model type not found: %s', model_type)
            raise ValueError('Model type not found')

    def fit(self, train_data):
        logger.info('Fitting model.')
        self.__df_train = train_data
        self.__model.fit(train_data)

    def detect(self, observations):
        idx_anomaly = self.__model.detect(observations)
        anomaly_th_lower = None
        anomaly_th_upper = None

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper
