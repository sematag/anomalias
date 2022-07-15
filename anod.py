from fastapi import FastAPI, Request

from anomalias.anomd import Anomd
import pandas as pd

from influxdb_client import InfluxDBClient

from pydantic import BaseModel
class DataFrame(BaseModel):
    idx: list
    dat: list

token = "r_Kvm50LqcjFRPADDcTcgOuJFgtsI6Yiu82Lh_PUKyldGO3cRgKvYnOvgGf7DluEhnXJCTWUrAr8sPKvtPuyfw=="
org = "IIE"
bucket = "anomalias"
influx_url = "http://localhost:8086"
timeout = 200

app = FastAPI()
anomd = Anomd()

#@app.put("/new/{len}")
def new_ts(len: int):
    id = anomd.add(len=len)
    anomd.start(id)
    return {"id": id}

#@app.put("/set_ad/{id}")
def set_ad(id: int):
    anomd.adtk_ad(id=id, model_type='MinClusterAD', n_clusters=10)

#@app.put("/fit_detect/{id}")
def fit(id: int, data):
    anomd.fit(data, id)

@app.post("/detect")
async def detect(data: DataFrame):
        print(data)
        #anomd.append(data, id) # Detection
        #endog, idx_anom = anomd.get_detection(id)
        #print(idx_anom)

def load_influx_data(measurement="go_memstats_gc_cpu_fraction"):
    client = InfluxDBClient(url=influx_url, token=token, org=org, timeout=timeout)
    query_api = client.query_api()

    query = '''from(bucket: "''' + bucket + '''")
              |> range(start: -1h)
              |> filter(fn: (r) => r["_measurement"] == "''' + measurement + '''")
              |> drop(columns:["_start", "_stop", "_field", "_measurement"])'''

    df = query_api.query_data_frame(query)

    df["_time"] = pd.to_datetime(df["_time"].astype(str))
    df = df.drop(columns=["result", "table"])
    df = df.set_index("_time")
    client.close()

    return df



new_ts(10)
set_ad(0)

ts = fit(0, load_influx_data())

