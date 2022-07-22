from fastapi import FastAPI, Request
import anomalias.log as log
logger = log.logger( 'API' )

from anomalias.anomd import Anomd
import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel


class DataFrame(BaseModel):
    idx: list
    dat: list

import time

token = "r_Kvm50LqcjFRPADDcTcgOuJFgtsI6Yiu82Lh_PUKyldGO3cRgKvYnOvgGf7DluEhnXJCTWUrAr8sPKvtPuyfw=="
org = "IIE"
bucket = "anomalias"
influx_url = "http://localhost:8086"
timeout = 200

def influx_write(record,data_frame_measurement_name,data_frame_tag_columns=None):
    client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
    write_api = client.write_api()
    write_api.write(bucket,org,record=record, data_frame_measurement_name=data_frame_measurement_name, data_frame_tag_columns=data_frame_tag_columns)
    client.close()


app = FastAPI()
anomd = Anomd()


@app.post("/newTS")
def newTS(len: int, id: str):
    id = anomd.add(len=len, id=id)
    #anomd.start(id)


@app.post("/setAD")
def setAD(id: str):
    anomd.adtk_ad(id=id, model_type='MinClusterAD', n_clusters=2)


@app.post("/start")
def start(id: str):
    anomd.start(id=id)


@app.post("/fit/{id}")
def fit(id: str, data: DataFrame):
    anomd.fit(id, pd.DataFrame(data.dat, index=pd.to_datetime(data.idx)))

@app.post("/detect/{id}")
async def detect(id: str, data: DataFrame):
    anomd.append(id, pd.DataFrame(data.dat, index=pd.to_datetime(data.idx)))  # Detection
    try:
        df, anom = anomd.get_detection(id)
        df_out = df.rename(columns={0: 'label'})
        df_out = df_out[anom[0]]
        print(df_out)
        influx_write(record=df_out[['label']], data_frame_measurement_name=id)
    except Exception as e:
        logger.error('%s', e)


import nest_asyncio
import uvicorn

nest_asyncio.apply()
uvicorn.run(app, port=8000, host="0.0.0.0")