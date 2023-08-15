import warnings

warnings.filterwarnings("ignore")

from anomalias.dcvaemodel import DcvaeAD
import pandas as pd
import numpy as np


index = pd.date_range('1/1/2000', periods=44, freq='15T')
dat = pd.DataFrame({'AACallCostHome': range(44)} , index=index)
dat.index.rename('_time', inplace=True)

print(dat)
dat = dat.asfreq(freq='5T', method='ffill')
print(dat)

print(dat.shape)

model = DcvaeAD(th_lower=0, th_upper=10)

model.fit(dat)

idx_anomaly, anomaly_th_lower, anomaly_th_upper = model.detect(dat)


print(idx_anomaly)
print(idx_anomaly.asfreq(freq='10T'))

print(anomaly_th_lower)
print(anomaly_th_lower.asfreq(freq='10T'))

print(anomaly_th_upper)
print(anomaly_th_upper.asfreq(freq='10T'))