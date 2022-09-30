from fastapi import FastAPI
import anomalias.log as log

import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel
from typing import List
import nest_asyncio
import uvicorn
import configparser

from anomalias.tsmodels import SsmAD, ExpAD
from anomalias.adtk import AdtkAD

logger = log.logger('API')

config = configparser.ConfigParser()
config.read("config.ini")

token = config.get("influx", "token")
org = config.get("influx", "org")
bucket = config.get("influx", "bucket")
influx_url = config.get("influx", "influx_url")
timeout = config.get("influx", "timeout")

logger.debug('%s:', influx_url)


class DataFrame(BaseModel):
    index: list
    values: list
    metrics: List[str]


class InfluxApi:
    def __init__(self):
        self.__client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        self.__write_api = self.__client.write_api()

    def write(self, df, anomalies, anomaly_th_lower, anomaly_th_upper):
        for metric in df:
            df_out = df[metric].to_frame()
            anomalies_out = anomalies[metric]
            anomalies_out = anomalies[anomalies_out].rename(columns={metric: 'anomaly'}).astype(int)
            logger.debug('api.py: anomalies to write (metric %s):', metric)
            logger.debug('\n %s', df_out)
            logger.debug('\n %s', anomalies_out)
            self.__write_api.write(bucket, org, record=df_out, data_frame_measurement_name=metric,
                                   data_frame_tag_columns=None)
            self.__write_api.write(bucket, org, record=anomalies_out, data_frame_measurement_name=metric,
                                   data_frame_tag_columns=None)
            if anomaly_th_lower is not None and anomaly_th_upper is not None:
                anomaly_th_lower_out = anomaly_th_lower[metric].to_frame()
                anomaly_th_upper_out = anomaly_th_upper[metric].to_frame()
                self.__write_api.write(bucket, org, record=anomaly_th_lower_out, data_frame_measurement_name=metric,
                                       data_frame_tag_columns=None)
                self.__write_api.write(bucket, org, record=anomaly_th_upper_out, data_frame_measurement_name=metric,
                                       data_frame_tag_columns=None)

    def close(self):
        self.__client.close()


def start(detectors):
    api = FastAPI()

    @api.post("/newTS")
    def new_ts(df_len: int, df_id: str):
        influx_api = InfluxApi()
        detectors.add(df_len=df_len, df_id=df_id, api=influx_api)

    @api.post("/setAD")
    def set_ad(df_id: str, model_id: str, data: DataFrame):
        if model_id == 'MinClusterAD':
            model = AdtkAD(model_id, n_clusters=2)
            detectors.set_ad(df_id, model)
        elif model_id == 'ExpAD':
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(pd.infer_freq(df.index))

            model = ExpAD(th=1,
                          df=df,
                          model_type="Exp",
                          seasonal=12,
                          initialization_method='concentrated')
            detectors.set_ad(df_id, model)
        elif model_id == 'SsmAD':
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(pd.infer_freq(df.index))

            logger.debug('api.py: call to set_ad(), data:')
            logger.debug('\n %s', df)

            model = SsmAD(th=5,
                          df=df,
                          model_type="SARIMAX",
                          order=(1, 1, 2),
                          seasonal_periods=12)
            detectors.set_ad(df_id, model)

    @api.post("/startAD")
    def start_ad(df_id: str):
        detectors.start(df_id=df_id)

    @api.post("/fit/{df_id}")
    def fit(df_id: str, data: DataFrame):
        df = pd.DataFrame(list(zip(data.values, data.metrics)),
                          columns=['values', 'metrics'], index=pd.to_datetime(data.index))
        df = df.pivot(columns='metrics', values='values')
        df = df.asfreq(pd.infer_freq(df.index))

        logger.debug('api.py: call to fit(), data:')
        logger.debug('\n %s', df)

        detectors.fit(df_id, df)

    @api.post("/detect/{df_id}")
    async def detect(df_id: str, data: DataFrame):
        try:
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(pd.infer_freq(df.index))

            logger.debug('api.py: call to detect(), data:')
            logger.debug('\n %s', df)

            detectors.append(df_id, df)  # Detection
        except Exception as e:
            logger.error('%s', e, exc_info=True)

    nest_asyncio.apply()
    uvicorn.run(api, port=8000, host="0.0.0.0")
