# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 07:03:05 2022

@author: gastong@fing.edu.uy
"""



import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow import keras
import pandas as pd
import numpy as np
import pickle
from anomalias.utils import samples2model
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


class DcvaeAD:
    def __init__(self, th_sigma=1, th_lower=None, th_upper=None,
                 model_path='./anomalias/model_files/', 
                 model_name='dc-vae_global_best_model',
                 scaler_path='./anomalias/scaler_files/',
                 T=128,
                 freq='5T'):
        
        logger.info('Setting DCVAE model.')
        self.__th_sigma = th_sigma
        self.__th_lower = th_lower
        self.__th_upper = th_upper

        self.__T = T
        self.__freq = freq
        self.__model_name = model_name

        logger.info('Model name: '+model_name)
        self.__model = None
        self.__init = True
        self.__trained = True
        self.__model_fit = keras.models.load_model(model_path+model_name+'.h5',
                                                    custom_objects={'sampling': Sampling},
                                                    compile = True)
        #logger.info('Model Encoder:', self.__model_fit.encoder.summary())
        #logger.info('Model Decoder:', self.__model_fit.decoder.summary())
        #logger.info('Model VAE:', self.__model_fit.summary())

        

    def fit(self, df=None):
        '''
        Por ahora no creemos conveniente habilitar el entrenamiento para este método.
        Como mucho se podría hacer un ajuste para una nueva serie. Pero eso queda para más adelante.
        '''
        self.param_norm = df.quantile(0.98)

        return self
         
    def detect(self, df):
        
        # Data preprocess
        # Normalization

        df_X = df.asfreq(freq='5T', method='ffill')
        df_X = df_X/self.param_norm
        
        X = samples2model(df_X, T=self.__T)
        X = np.array(X)
        #X = np.expand_dims(X, axis=0)

        # Predictions
        prediction = self.__model_fit(X)

        predicted_mean_values = np.concatenate((prediction[0][0,:,:], prediction[0][1:,-1,:]))
        predicted_sigma_values = np.concatenate((prediction[1][0,:,:], prediction[1][1:,-1,:]))

        predicted_mean_values = np.squeeze(predicted_mean_values)
        predicted_sigma_values = np.squeeze(np.sqrt(np.exp(predicted_sigma_values)))

        # Only the newest predictions are taken
        predicted_mean = pd.DataFrame(data=predicted_mean_values, index=df_X.index, 
                                      columns=df_X.columns)
        predicted_sigma = pd.DataFrame(data=predicted_sigma_values, index=df_X.index, 
                                      columns=df_X.columns)
        if isinstance(predicted_mean, pd.Series):
            predicted_mean = predicted_mean.to_frame()
            predicted_sigma = predicted_sigma.to_frame()

        predicted_mean = predicted_mean.asfreq(freq=self.__freq)
        predicted_sigma = predicted_sigma.asfreq(freq=self.__freq)
        
        
        anomaly_th_lower = (predicted_mean - self.__th_sigma * predicted_sigma) * self.param_norm
        anomaly_th_upper = (predicted_mean + self.__th_sigma * predicted_sigma) * self.param_norm

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
        