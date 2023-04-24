import warnings

warnings.filterwarnings("ignore")

from dase.das.lstm import LSTM_AD
import pandas as pd


index = pd.date_range('1/1/2000', periods=400, freq='5T')
dat = pd.DataFrame({'field1': [3, 4, 5, 6]*100,
                    'field2': [3, 4, 8, 4]*100}, index=index)
dat.index.rename('_time', inplace=True)

model = LSTM_AD()

model.fit(dat['field1'].to_frame())
idx_anomaly = model.detect(dat['field2'].to_frame())

print(idx_anomaly)