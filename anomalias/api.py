from fastapi import FastAPI
import anomalias.log as log

import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel
from typing import List
import nest_asyncio
import uvicorn

logger = log.logger('API')

class DataFrame(BaseModel):
    #id: str
    index: list
    values: list
    metrics: List[str]

token = "r_Kvm50LqcjFRPADDcTcgOuJFgtsI6Yiu82Lh_PUKyldGO3cRgKvYnOvgGf7DluEhnXJCTWUrAr8sPKvtPuyfw=="
org = "IIE"
bucket = "anomalias"
influx_url = "http://localhost:8086"
timeout = 200

class influx_api():
    def __init__(self):
        self.__client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        self.__write_api = self.__client.write_api()

    def write(self, dataFrame, anomalies):
        for metric in dataFrame:
            df_out = dataFrame[metric].to_frame()
            df_out = df_out.rename(columns={metric: 'anomaly'})
            df_out = df_out[anomalies[metric]]
            logger.debug('api.py: anomalies to write (metric %s):', metric)
            logger.debug('\n %s', df_out)
            self.__write_api.write(bucket,org,record=df_out, data_frame_measurement_name=metric, data_frame_tag_columns=None)

    def close(self):
        self.__client.close()

def start(detectors):

        api = FastAPI()

        @api.post("/newTS")
        def newTS(len: int, id: str):
            api = influx_api()
            detectors.add(len=len, id=id, api=api)


        @api.post("/setAD")
        def setAD(id: str):
            detectors.adtk_ad(id=id, model_type='MinClusterAD', n_clusters=2)


        @api.post("/start")
        def start(id: str):
            detectors.start(id=id)


        @api.post("/fit/{id}")
        def fit(id: str, data: DataFrame):
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                                columns =['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')

            logger.debug('api.py: call to fit(), data:')
            logger.debug('\n %s', df)

            detectors.fit(id, df)


        @api.post("/detect/{id}")
        async def detect(id: str, data: DataFrame):
            try:
                df = pd.DataFrame(list(zip(data.values, data.metrics)),
                                  columns=['values', 'metrics'], index=pd.to_datetime(data.index))
                df = df.pivot(columns='metrics', values='values')

                logger.debug('api.py: call to detect(), data:')
                logger.debug('\n %s', df)

                detectors.append(id, df)  # Detection
            except Exception as e:
                logger.error('%s', e, exc_info=True)

        nest_asyncio.apply()
        uvicorn.run(api, port=8000, host="0.0.0.0")