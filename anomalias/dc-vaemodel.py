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
from utils import scaler01, MTS2UTS_cond
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

    def fit(self, df=None):
        '''
        Por ahora no creemos conveniente habilitar el entrenamiento para este método.
        Como mucho se podría hacer un ajuste para una nueva serie. Pero eso queda para más adelante.
        '''
        return self
         
    def detect(self, df):

        # Inference model. Auxiliary model so that in the inference 
        # the prediction is only the last value of the sequence
        inp = Input(shape=(self.T, 1))
        x = self.__model_fit(inp) # apply trained model on the input
        out = Lambda(lambda y: [y[0][:,-1,:], y[1][:,-1,:]])(x)
        inference_model = Model(inp, out)
        
        # Data preprocess
        # Normalization
        df = scaler01(df, self.__scaler_filename, 'transform')

        sam_val, sam_info = MTS2UTS_cond(df, T=self.__T)
        
        # Predictions
        prediction = self.vae.predict(np.stack(sam_val))
        # The first T-1 data of each sequence are discarded
        reconstruct = prediction[0]
        sig = np.sqrt(np.exp(prediction[1]))
        
        # Data evaluate (The first T-1 data are discarded)
        df_evaluate = UTS2MTS(sam_val, sam_ix, sam_class)
        df_reconstruct = UTS2MTS(reconstruct, sam_ix, sam_class)
        df_sig = UTS2MTS(sig, sam_ix, sam_class)
        
        # Thresholds
        if len(alpha_set) == self.M:
            alpha = np.array(alpha_set)
        elif load_alpha:
            with open(self.name + '_alpha.pkl', 'rb') as f:
                alpha = pickle.load(f)
                f.close()
        else:
            alpha = self.alpha
            
        thdown = df_reconstruct.values - alpha*df_sig.values
        thup = df_reconstruct.values + alpha*df_sig.values
        
        # Evaluation
        pred = (df_evaluate.values < thdown) | (df_evaluate.values > thup)
        df_predict = pd.DataFrame(pred, columns=df_X.columns, index=df_X.iloc[self.T-1:].index)
        
        if only_predict:
            return df_predict
        else:
            latent_space = self.encoder.predict(np.stack(sam_val))[2]
            return df_predict, df_reconstruct, df_sig, latent_space, sam_ix, sam_class


        