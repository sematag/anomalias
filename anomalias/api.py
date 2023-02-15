from fastapi import FastAPI
import anomalias.log as log

import pandas as pd
import os
import pickle

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
bucket_train = config.get("influx", "bucket_train")
influx_url = config.get("influx", "influx_url")
timeout = config.get("influx", "timeout")
port = int(config.get("influx", "port"))

logger.debug('%s:', influx_url)


class DataFrame(BaseModel):
    index: list
    values: list
    metrics: List[str]
    freq: str


class DataModel(BaseModel):
    index: list
    values: list
    metrics: List[str]
    freq: str
    threshold: float = 4
    order: list = (1, 1, 2)
    th_lower: float = None
    th_upper: float = None
    pre_log: bool = False


class InfluxApi:
    def __init__(self):
        self.__client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
        self.__write_api = self.__client.write_api()
        self.__delete_api = self.__client.delete_api()

    def delete(self, measurement):
        self.__delete_api.delete("1970-01-01T00:00:00Z", "2073-12-12T00:00:00Z", '_measurement="' + measurement + '"',  bucket=bucket_train, org=org)

    def write(self, df, anomalies, anomaly_th_lower, anomaly_th_upper, measurement, train=False):
        if train:
            bk = str(bucket_train)
        else:
            bk = str(bucket)

        if not df.empty:
            for metric in df:
                df_out = df[metric].to_frame()

                anomalies_out = anomalies[metric]
                anomalies_out = anomalies[anomalies_out].rename(columns={metric: 'anomaly'}).astype(int)
                logger.debug('api.py: anomalies to write (metric %s):', metric)
                logger.debug('\n %s', df_out)
                logger.debug('\n %s', anomalies_out)

                logger.debug('\n %s', bk)

                self.__write_api.write(bk, org, record=df_out, data_frame_measurement_name=measurement,
                                       data_frame_tag_columns=None)
                self.__write_api.write(bk, org, record=anomalies_out, data_frame_measurement_name=measurement,
                                       data_frame_tag_columns=None)
                if anomaly_th_lower is not None and anomaly_th_upper is not None:
                    anomaly_th_lower_out = anomaly_th_lower[metric].to_frame()
                    anomaly_th_lower_out = anomaly_th_lower_out.rename(columns={metric: 'anomalyThL'})
                    anomaly_th_upper_out = anomaly_th_upper[metric].to_frame()
                    anomaly_th_upper_out = anomaly_th_upper_out.rename(columns={metric: 'anomalyThU'})

                    self.__write_api.write(bk, org, record=anomaly_th_lower_out, data_frame_measurement_name=measurement,
                                           data_frame_tag_columns=None)
                    self.__write_api.write(bk, org, record=anomaly_th_upper_out, data_frame_measurement_name=measurement,
                                           data_frame_tag_columns=None)

    def close(self):
        self.__client.close()


def init(detectors):
    api = FastAPI()

    @api.post("/newTS")
    def new_ts(df_len: int, df_id: str):
        influx_api = InfluxApi()
        res = detectors.add(df_len=df_len, df_id=df_id, api=influx_api)

        with open('state/state.ini', 'w') as file:
            file.write('\n'.join(detectors.list_ad()))
        return res

    @api.post("/setAD")
    def set_ad(df_id: str, model_id: str, data: DataModel):
        if model_id == 'MinClusterAD':
            model = AdtkAD(model_id, n_clusters=2)
            detectors.set_model(df_id, model)
        elif model_id == 'ExpAD':
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(data.freq)

            model = ExpAD(th=1,
                          df=df,
                          model_type="Exp",
                          seasonal=12,
                          initialization_method='concentrated')
            detectors.set_model(df_id, model)
        elif model_id == 'SsmAD':
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(data.freq)

            model = SsmAD(df=df,
                          th_sigma=data.threshold,
                          th_lower=data.th_lower,
                          th_upper=data.th_upper,
                          order=data.order,
                          pre_log=data.pre_log
                          )
            detectors.set_model(df_id, model)

        with open('state/' + df_id + '.model', 'w+') as file:
            file.write(model_id)
            file.close()

        with open('state/'+df_id+'_DataModel.pkl', 'wb') as file:
            pickle.dump(data, file)
            file.close()

    @api.post("/startAD")
    def start_ad(df_id: str):
        detectors.start(df_id=df_id)

    @api.get("/removeAD")
    def remove_ad(df_id: str):
        res = detectors.remove(df_id=df_id)
        influx_api = InfluxApi()
        influx_api.delete(df_id)
        influx_api.close()
        logger.debug('\n %s', res)

        with open('state/state.ini', 'w') as file:
            file.write('\n'.join(detectors.list_ad()))
            file.close()

        os.remove('state/' + df_id + '.DataModel')
        os.remove('state/' + df_id + '.DataFrame')
        os.remove('state/' + df_id + '.model')

        return res

    @api.post("/fit")
    def fit(df_id: str, data: DataFrame):
        df = pd.DataFrame(list(zip(data.values, data.metrics)),
                          columns=['values', 'metrics'], index=pd.to_datetime(data.index))
        df = df.pivot(columns='metrics', values='values')
        df = df.asfreq('5T')

        logger.debug('api.py: call to fit(), data:')
        logger.debug('\n %s', df)

        anomalies, anomaly_th_lower, anomaly_th_upper = detectors.fit(df_id, df)
        influx_api = InfluxApi()
        influx_api.delete(df_id)
        influx_api.write(df, anomalies, anomaly_th_lower, anomaly_th_upper, measurement=df_id, train=True)
        influx_api.close()

        with open('state/' + df_id + '_DataFrame.pkl', 'wb') as file:
            pickle.dump(data, file)
            file.close()

        return "OK"

    @api.post("/detect")
    def detect(df_id: str, data: DataFrame):
        try:
            df = pd.DataFrame(list(zip(data.values, data.metrics)),
                              columns=['values', 'metrics'], index=pd.to_datetime(data.index))
            df = df.pivot(columns='metrics', values='values')
            df = df.asfreq(data.freq)

            logger.debug('api.py: call to detect(), data:')
            logger.debug('\n %s', df)

            detectors.append(df_id, df)  # Detection
        except Exception as e:
            logger.error('%s', e, exc_info=True)

    @api.get("/listAD")
    def list_ad():
        return set(detectors.list_ad())

    # Read system state
    with open('state/state.ini') as file:
        metrics = [line.rstrip('\n') for line in file]
        file.close()
    if metrics:
        for metric in metrics:
            logger.debug('Read system state (metric %s):', metric)
            with open('state/' + metric + '.model') as file:
                model = file.read()
                file.close()

            with open('state/' + metric + '_DataModel.pkl', 'rb') as file:
                dat_model = pickle.load(file)
                file.close()

            with open('state/' + metric + '_DataFrame.pkl', 'rb') as file:
                dat_frame = pickle.load(file)
                file.close()

            new_ts(15, metric)
            set_ad(metric, model, dat_model)
            fit(metric, dat_frame)
            start_ad(metric)

    nest_asyncio.apply()
    cfg = uvicorn.Config(api, port=port, host="0.0.0.0", log_level="info")
    server = uvicorn.Server(cfg)
    server.run()

