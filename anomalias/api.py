from fastapi import FastAPI
import anomalias.log as log

import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel
from typing import List
import nest_asyncio
import uvicorn
from datetime import datetime

logger = log.logger('API')

class DataFrame(BaseModel):
    #id: str
    index: list
    values: list
    metrics: List[str]

token = "spI_l176puCr13ymJbBWkx8ImeX-SXPOHb1HxJonLoMnMwLrlz4U2Qkoko62aVI5bnisix-DDcEihUzfXcQTAA=="
org = "fing"
bucket = "carparts"
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

        
        @api.post("/stop")
        def stop(id: str):
            detectors.remove(id=id)



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


        
        """
        Ruta /write/ para HTTP GET
        Escribe datos en la base de datos influxdb

        Recibe los siguientes parametros:    
            -data_group: Nombre de la medida que se realiza (por ejemplo, ventas)
            -y_axis: Nombre del campo que se mide (ejemplo para carparts, ventas_mensuales)
            -values: string con valores a registrar en la base (ejemplo 0,2,3,4,1,2,3)
            -tag: Nombre del tag para la medida (ejemplo para carpats, numero_parte)
            -tag_value: Valor del tag (ejemplo para carparts, 21047181)

        """
        @api.get("/write/")
        def write_db(data_group,y_axis,tag,tag_value,values):

            json_data = []    
            for value in values.split(","):

                record = [
                    {
                        "measurement": data_group,
                        "tags": {
                            tag: tag_value
                        },
                        "time": datetime.now(),
                        "fields": {
                            y_axis:  float(value)
                        }
                    }
                ]

                json_data.append(record)
            
            write_api =  InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout).write_api()
            write_api.write(bucket=bucket, org=org, record=json_data)

            print(json_data)

            return {"data_group": data_group, "y_axis":y_axis, "tag": tag, "tag_value": tag_value, "values": values.split(",")}


        nest_asyncio.apply()
        uvicorn.run(api, port=8000, host="0.0.0.0")