import pandas as pd
import numpy as np


"""
Clase que normaliza una serie moviendo su codominio desde [min(serie), max(serie)] a [0,1]
"""
class MinMaxNormalizer():
    def __init__(self):
        self.minimuns = {}
        self.maximuns = {}

    def fit(self, data : pd.DataFrame, name : str, column : str):
        self.minimuns[name] = data[column].min()
        self.maximuns[name] = data[column].max()

    def transform(self, data : pd.DataFrame, name : str, column : str):
        data = data.copy()
        data[column] = (data[column] - self.minimuns[name]) / (self.maximuns[name] - self.minimuns[name])
        return data 

    def fitTransform(self, data : pd.DataFrame, name : str, column : str):
        self.fit(data,name,column)
        return self.transform(data,name,column)


"""
Clase que periodiaza la serie a una frecuencia seleccionada.
Ejemplo:
    freq = "5min"
    freq = "20min"
    freq = "1h"
"""
class Periodizer():
    
    def __init__(self, freq : str="5min"):
        self.freq = freq
    
    def fit(self, data : pd.DataFrame, name : str, column : str):
        pass

    def transform(self, data : pd.DataFrame, name : str, column : str):
        periodic_data = pd.DataFrame()
        old_index = data.index
        new_index = pd.date_range(data.index[0],data.index[-1],freq=self.freq)
        
        new_data = []
        old_index_position = 0
        index_counter = 0
        for timestamp in new_index:
            while old_index[old_index_position] <= timestamp:
                index_counter+=data[column][old_index_position]
                old_index_position+=1
            new_data.append(index_counter)
            index_counter = 0

        periodic_data[column] = new_data
        periodic_data.index = new_index

        return periodic_data

    def fitTransform(self, data : pd.DataFrame, name : str, column : str):
        self.fit(data,name,column)
        return self.transform(data,name,column)


"""
Clase que elimina el ruido de la serie (picos de bajo valor) utilizando un treshold basado en media + factor*std
Esto permite convertir la serie a algo mas parecido a demanda intermitente
"""
class NoiseFilter():
    def __init__(self, deviation_factor : float=1.0, zero_value : float=0.0):
        self.deviation_factor = deviation_factor
        self.zero_value = zero_value
        self.means = {}
        self.stds = {}
    
    def fit(self, data : pd.DataFrame, name : str, column : str):
        self.means[name] = data[column].mean()
        self.stds[name] = data[column].std()

    def transform(self, data : pd.DataFrame, name : str, column : str):
        
        data = data.copy()
        data[column] = [self.zero_value if value < self.means[name]  + self.deviation_factor*self.stds[name] else value for value in data[column] ]
        return data

    def fitTransform(self, data : pd.DataFrame, name : str, column : str):
        self.fit(data,name,column)
        return self.transform(data,name,column)