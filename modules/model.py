#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marzio Monticelli (1459333)
"""

from tensorflow import keras
from tensorflow.keras.layers import Activation
from tensorflow.keras import utils

def square_activation(x):
    return x**2

utils.get_custom_objects().update({'square_activation': Activation(square_activation)})

class Model:
    
    def getAccuracy(self, model, x_val, y_val, batch_size = None):
        score = model.evaluate(x_val, y_val, verbose=1, batch_size=batch_size)
        score_acc_val = score[1]*100
        return score_acc_val        
    
    def getAccuracyM(self, model, x_val, y_val, batch_size = None):
        score = model.evaluate(x_val, y_val, verbose=1, batch_size=batch_size)
        score_acc_val = score[1]*100
        return score_acc_val, model
    
    def fit(self, x_train, y_train, x_val, y_val, 
            params = {
                'kernel_size': (3,3),
                'strides': (2,2),
                'last_activation': 'softmax',
                'optimizer': 'SGD',
                'loss': 'categorical_crossentropy',
                'batch_size': 200,
                'epochs': 50,
                'dropout': 0.2,
                'learning_rate': 0.01,
                'momentum': 0.9,
                'nesterov': False,
                'use_dropout':True
            }):
        
        """ The mnist neural network: character recogniction task
            
            With respect to the official CryptoNet paper a Dropout in the visibile 
            layer is added so that to increase the network performances by 1.5%
            on this machine. 
        """
        
        print("")
        print("Model params:")
        print("")
        print("Input shape: "+ str(x_train[0].shape))
        print("")
        print(params)
        print("")
        
        #build the model
        model = keras.models.Sequential()
        
        # 1) Convolutional Layer
        model.add(keras.layers.Conv2D(5, kernel_size=params['kernel_size'], strides=params['strides'], input_shape=x_train[0].shape))
            
        # 2) Flatten Layer
        model.add(keras.layers.Flatten())
            
        # 3) Square Activation Layer
        model.add(keras.layers.Dense(100, activation=square_activation))
        
        if params['use_dropout']:
            model.add(keras.layers.Dropout(params['dropout'], input_shape=(100,)))
        
        # 4) Square Activation Layer
        model.add(keras.layers.Dense(10, activation=square_activation))
            
        # 5) Sigmoid / Softmax Activation Function
        model.add(keras.layers.Dense(10, activation=params['last_activation']))
        
        sgd = keras.optimizers.SGD(lr=params['learning_rate'], momentum=params['momentum'], nesterov=params['nesterov'])
        
        model.compile(optimizer=sgd,
                      loss=params['loss'],
                      metrics=['accuracy'])
            
    #    model.summary()
            
        history = model.fit(
            x_train, y_train,
            validation_data=[x_val, y_val],
            epochs=params['epochs'],
            batch_size=params['batch_size']
        )
        
        return history, model