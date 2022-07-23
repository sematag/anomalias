from fastapi import FastAPI
import anomalias.log as log
logger = log.logger('API')

import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel
import nest_asyncio
import uvicorn

class DataFrame(BaseModel):
    idx: list
    dat: list

token = "r_Kvm50LqcjFRPADDcTcgOuJFgtsI6Yiu82Lh_PUKyldGO3cRgKvYnOvgGf7DluEhnXJCTWUrAr8sPKvtPuyfw=="
org = "IIE"
bucket = "anomalias"
influx_url = "http://localhost:8086"
timeout = 200

class influx_api():
    def __init__(self):
        self.__client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        self.__write_api = self.__client.write_api()

    def write(self,record,data_frame_measurement_name,data_frame_tag_columns=None):
        self.__write_api.write(bucket,org,record=record, data_frame_measurement_name=data_frame_measurement_name, data_frame_tag_columns=data_frame_tag_columns)

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
            detectors.fit(id, pd.DataFrame(data.dat, index=pd.to_datetime(data.idx)))


        @api.post("/detect/{id}")
        async def detect(id: str, data: DataFrame):
            try:
                logger.debug('api.py: call to detect(), data:')
                logger.debug('\n %s', pd.DataFrame(data.dat, index=pd.to_datetime(data.idx)))
                detectors.append(id, pd.DataFrame(data.dat, index=pd.to_datetime(data.idx)))  # Detection
            except Exception as e:
                logger.error('%s', e, exc_info=True)

        nest_asyncio.apply()
        uvicorn.run(api, port=8000, host="0.0.0.0")