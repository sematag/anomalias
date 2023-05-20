import anomalias.log as log
import anomalias.dataframe as dataframe

logger = log.logger('Detectors')


class Detectors:
    def __init__(self):
        self.__dataframes = []
        self.__df_id = []

    def add(self, df_id, df_len, api, zbx_alert=True):
        try:
            if not self.__exist_id(df_id):
                self.__dataframes.append(dataframe.DataFrame(df_id=df_id, df_len=df_len, api=api, zbx_alert=zbx_alert))
                self.__df_id.append(df_id)
                return "OK"
            else:
                logger.warning('Already exists a series with identity: %s', df_id)
                return "ERROR"
        except Exception as e:
            logger.error('%s', e)
            return "ERROR"
            raise

    def set_model(self, df_id, model):
        try:
            if self.__exist_id(df_id):
                (self.__dataframes[self.__df_id.index(df_id)]).ad.set_model(model)
        except Exception as e:
            logger.error('%s', e)
            return None

    def remove(self, df_id):
        try:
            if self.__exist_id(df_id):
                index = self.__df_id.index(df_id)
                self.__dataframes[index].exit()
                del self.__dataframes[index]
                self.__df_id.remove(df_id)
                return "OK"
        except Exception as e:
            logger.error('%s', e)
            return "ERROR"

    def list_ad(self):
        # List all detectors
        return self.__df_id

    def active_ad(self):
        # List running detectors
        active_ad = [df_id for df_id in self.__df_id if self.__dataframes[self.__df_id.index(df_id)].is_alive()]
        return active_ad

    def start(self, df_id):
        try:
            if self.__exist_id(df_id):
                if not self.__dataframes[self.__df_id.index(df_id)].is_alive():
                    self.__dataframes[self.__df_id.index(df_id)].start()
                else:
                    logger.warning('Series is running, id: %s', df_id)
        except Exception as e:
            logger.error('%s', e)

    def append(self, df_id, obs):
        if self.__exist_id(df_id):
            self.__dataframes[self.__df_id.index(df_id)].append(obs)

    def fit(self, df_id, df):
        if self.__exist_id(df_id):
            anomalies, anomaly_th_lower, anomaly_th_upper = self.__dataframes[self.__df_id.index(df_id)].ad.fit(df)
            return anomalies, anomaly_th_lower, anomaly_th_upper
        else:
            return None, None, None

    def get_detection(self, df_id):
        try:
            if self.__exist_id(df_id):
                if self.__dataframes[self.__df_id.index(df_id)].is_alive():
                    df, anomalies = (self.__dataframes[self.__df_id.index(df_id)]).ad.get_detection()
                    return df, anomalies
                else:
                    logger.info('Time series is not running, id: %s', df_id)
                    return None
        except Exception as e:
            logger.error('%s', e)
            return None

    def __exist_id(self, df_id):
        try:
            if self.__df_id.__contains__(df_id):
                return True
            else:
                return False
        except Exception as e:
            logger.error('%s', e)
            return False

    def set_all_obs_detect(self, df_id, bol):
        try:
            if self.__exist_id(df_id):
                (self.__dataframes[self.__df_id.index(df_id)]).ad.set_all_obs_detect(bol)
        except Exception as e:
            logger.error('%s', e)
            return None
