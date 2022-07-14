from fastapi import FastAPI

from anomalias.anomd import Anomd
import pandas as pd
import time

from influxdb_client import InfluxDBClient

token = "GeJH-3jCUiEGojt716N8LD3NXmlPoHLy-AG5iAWuLrobpqV3BQNn1MjyBwQ_Z9u4sglWZmgC_8_j7ghlen8hcQ=="
org = "IIE"
client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
query_api = client.query_api()
write_api = client.write_api()

app = FastAPI()
anomd = Anomd()

@app.put("/new/{len}")
def new_ts(len: int):
    id = anomd.add(len=len)
    anomd.start(id)
    return {"id": id}

@app.put("/set_ad/{id}")
def set_ad(id: int):
    anomd.adtk_ad(id=id, model_type='MinClusterAD', n_clusters=10)

@app.put("/fit_detect/{id}")
def fit_detect(id: int):
    query = '''from(bucket: "fmv_anomalias")
      |> range(start: -30m)
      |> filter(fn: (r) => r["_measurement"] == "go_memstats_heap_idle_bytes")
      |> filter(fn: (r) => r["_field"] == "gauge")
      |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

    df = query_api.query_data_frame(query)

    df["_time"] = pd.to_datetime(df["_time"].astype(str))
    df = df.drop(columns=["result", "table"])
    df = df.set_index("_time")

    anomd.fit(df, id)

@app.put("/runAD/{id}")
def run_ad(id: int):
    query = '''from(bucket: "fmv_anomalias")
          |> range(start: -15m)
          |> filter(fn: (r) => r["_measurement"] == "go_memstats_heap_idle_bytes")
          |> filter(fn: (r) => r["_field"] == "gauge")
          |> drop(columns:["_start", "_stop", "host", "_field", "_measurement"])'''

    df = query_api.query_data_frame(query)

    df["_time"] = pd.to_datetime(df["_time"].astype(str))
    df = df.drop(columns=["result", "table"])
    df = df.set_index("_time")
    anomd.append(df, id) # Detection
    time.sleep(3)
    endog, idx_anom = anomd.get_detection(id)

    df2w = endog[idx_anom[0] == True]
    df2w.columns = ["an1"]

    write_api.write("fmv_anomalias", "IIE", record=df2w, data_frame_measurement_name='go_memstats_heap_idle_bytes')

#new_ts(100)
#set_ad(0)
#ts = fit_detect(0)

#run_ad(0)

