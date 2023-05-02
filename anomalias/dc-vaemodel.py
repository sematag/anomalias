# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:03:05 2022

@author: gastong@fing.edu.uy
"""



import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from utils import scaler01, samples2model
from sklearn.preprocessing import StandardScaler

from anomalias import log
logger = log.logger('dcvae')


@keras.utils.register_keras_serializable()
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self, name=None, k=1, **kwargs):
        super(Sampling, self).__init__(name=name)
        self.k = k
        super(Sampling, self).__init__(**kwargs)
        
    def get_config(self):
        config = super(Sampling, self).get_config()
        config['k'] = self.k
        return config #dict(list(config.items()))
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        seq = K.shape(z_mean)[1]
        dim = K.shape(z_mean)[2]
        epsilon = K.random_normal(shape=(batch, seq, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class DCVAE:
    def __init__(self, th_sigma, th_lower=None, th_upper=None, 
                 model_path='', 
                 model_name='dc-vae_best_model',
                 scaler_filename='',
                 T=128,
                 batch_size=32,
                 epochs=100,
                 validation_split=0.2,
                   **kwargs):
        logger.info('Setting DCVAE model.')
        self.__th_sigma = th_sigma
        self.__th_lower = th_lower
        self.__th_upper = th_upper

        self.__T = T
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__scaler = StandardScaler()
        self.__validation_split = validation_split
        self.__model_name = model_name
        self.__scaler_filename = scaler_filename

        logger.info('Model name: '+model_name)
        self.__model = None
        self.__init = True
        self.__trained = True
        self.__model_fit = keras.models.load_model(model_path+model_name+'.h5',
                                                    custom_objects={'sampling': Sampling},
                                                    compile = True)
        self.info('Model Encoder:', self.__model_fit.encoder.summary())
        self.info('Model Decoder:', self.__model_fit.decoder.summary())
        self.info('Model VAE:', self.__model_fit.vae.summary())

    def fit(self, df=None):
        '''
        Por ahora no creemos conveniente habilitar el entrenamiento para este método.
        Como mucho se podría hacer un ajuste para una nueva serie. Pero eso queda para más adelante.
        '''
        return self
         
    def detect(self, df):
        
        # Data preprocess
        # Normalization
        df_norm = scaler01(df, self.__scaler_filename, 'transform')
        sam_val, sam_info = samples2model(df_norm)
        
        # Predictions
        prediction = self.vae.predict((sam_val, sam_info))

        predicted_mean = pd.DataFrame(values=prediction[0], index=df.index, 
                                      columns=df.columns)
        predicted_sigma = pd.DataFrame(values=np.sqrt(np.exp(prediction[1])), index=df.index, 
                                      columns=df.columns)
        if isinstance(predicted_mean, pd.Series):
            predicted_mean = predicted_mean.to_frame()
            predicted_sigma = predicted_sigma.to_frame()

        predicted_mean = scaler01(predicted_mean, 'inverse')
        predicted_sigma = scaler01(predicted_sigma, 'inverse')
        
        anomaly_th_lower = predicted_mean.values - self.__th_sigma * predicted_sigma.values
        anomaly_th_upper = predicted_mean.values + self.__th_sigma * predicted_sigma.values

        if self.__th_lower is not None:
            anomaly_th_lower.clip(lower=self.__th_lower, inplace=True)
        else:
            anomaly_th_lower.clip(lower=np.nanmin(anomaly_th_lower[anomaly_th_lower != -np.inf]), inplace=True)

        if self.__th_upper is not None:
            anomaly_th_upper.clip(upper=self.__th_upper, inplace=True)
        else:
            anomaly_th_upper.clip(upper=np.nanmax(anomaly_th_upper[anomaly_th_upper != np.inf]), inplace=True)

        idx_anomaly = (df > anomaly_th_upper) | (df < anomaly_th_lower)

        return idx_anomaly, anomaly_th_lower, anomaly_th_upper
        