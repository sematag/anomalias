


import requests
import pandas as pd

from influxdb_client import InfluxDBClient

from datetime import datetime
import time

token = "-mEXQY9kTDyuKDpAn8V3Kyh7K2ZKrKYCCnNZVs7jtJ96nUUV-83gItPT9cmTdoxDljH0am1oCK7qSqDfD30FTA=="
org = "fing"
bucket = "carparts"
influx_url = "http://localhost:8086"
org_id = "25d4ce73026cc614"
timeout = 200

if __name__ == "__main__":
    with open("influx_fit_carparts","r") as f:
        script = "".join(f.readlines())


    headers = {
        "Authorization" : "API {}".format(token),
        "Content-type": "application/json"
    }

    json_data = {
        "name" : "Fit Carparts",
        "description" : "Obtiene datos de carparts y llama a la api para entrenar",
        "orgID" : "25d4ce73026cc614",
        "script" : script
    }

    response = requests.post(influx_url, headers=headers, json=json_data)

    print(response.content)
        
