import numpy as np
import scipy
import os
import keras.backend as K
import pandas as pd
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D, BatchNormalization, MaxPooling1D, RepeatVector, TimeDistributed
from keras.layers import LSTM, GRU, ConvLSTM2D
from keras.layers.merge import concatenate
from keras import optimizers, regularizers
from keras.metrics import logcosh
from custom_recurrents import AttentionDecoder
import tensorflow as tf

np.random.seed(1)


class NeuralNetworks:
    def __init__(self, neurons_1, neurons_2, neurons_3, dropout):
        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.neurons_3 = neurons_3
        self.dropout = dropout

    def custom_loss(self, y_pred, y_true, plan):
        plan_error = K.sum(K.abs(y_true - plan))
        pred_error = K.sum(K.abs(y_true - y_pred))
        poymano_loss = (plan_error - pred_error)/plan_error
        return poymano_loss

    def nn_model(self, previous_fact, ar_features, plan, weekdays, errors, fourier, targets):
        input_1 = Input(shape=previous_fact[0].shape)
        input_2 = Input(shape=ar_features[0].shape)
        input_3 = Input(shape=plan[0].shape)
        input_4 = Input(shape=weekdays[0].shape)
        input_5 = Input(shape=errors[0].shape)
        input_6 = Input(shape=fourier[0].shape)

        # block1 for input_1
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(input_1)
        x = Conv1D(filters=16, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=3)(x)
        x = Dropout(0.5)(x)
        output_1 = Flatten()(x)

        # # block2 for input_1
        y = GRU(units=100, activation='relu', return_sequences=True)(input_1)
        y = Dropout(0.5)(y)
        y = GRU(units=100, activation='relu', return_sequences=True)(y)
        y = Dropout(0.5)(y)
        output_2 = Flatten()(y)

        q = LSTM(units=100, activation='relu', return_sequences=True)(input_1)
        q = AttentionDecoder(units=100, activation='relu', output_dim=2)(q)
        # q = Dropout(0.5)(q)
        # q = LSTM(units=100, activation='relu', return_sequences=True)(q)
        # q = Dropout(0.5)(q)
        # q = TimeDistributed(Dense(1))(q)
        output_3 = Flatten()(q)

        w = Dense(units=100, activation='relu')(input_1)
        w = Dropout(0.3)(w)
        w = Dense(units=50, activation='relu')(w)
        w = Dropout(0.3)(w)
        w = Dense(units=10, activation='relu')(w)
        w = Dropout(0.3)(w)
        output_4 = Flatten()(w)

        # concatenation
        input_dense = concatenate([output_1, output_2, output_3, output_4, input_3])

        # dense layers for regression
        z = Dense(units=1000, activation='relu')(input_dense)
        output = Dense(targets.shape[1], activation='linear')(z)

        model = Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6], outputs=[output])
        adam = optimizers.Adam(lr=0.0001)
        model.compile(loss=logcosh, optimizer=adam)
        return model




