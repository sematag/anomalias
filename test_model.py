import warnings

warnings.filterwarnings("ignore")

from anomalias.dcvaemodel import DcvaeAD
import pandas as pd


index = pd.date_range('1/1/2000', periods=256, freq='5T')
dat = pd.DataFrame({'AACallCostHome': [3]*256} , index=index)
dat.index.rename('_time', inplace=True)

print(dat.shape)

model = DcvaeAD(scaler_filename='AACallCostHome_scaler', th_lower=0, th_upper=10)

model.fit(dat)

idx_anomaly, anomaly_th_lower, anomaly_th_upper = model.detect(dat)

print(idx_anomaly.head())
print(anomaly_th_lower.head())
print(anomaly_th_upper.head())