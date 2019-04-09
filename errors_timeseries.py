from nn_architecture import NeuralNetworks
from extract_features import ExtractFeatures
from nearest_neighbors_reg import NeighborsRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.stattools import ccf, acf
import matplotlib.pyplot as plt
import numpy as np
import tspy
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
import xlwt


np.random.seed(1)

# define customer's objects
c3 = NeighborsRegression(file_list=['consumer_3/consumer3_2017.xlsx',
                                    'consumer_3/consumer3_2018.xlsx',
                                    'consumer_3/consumer3_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 4, 9])

c2 = NeighborsRegression(file_list=['consumer_2/consumer2_2017.xlsx',
                                    'consumer_2/consumer2_2018.xlsx',
                                    'consumer_2/consumer2_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 4, 9])


def errors_analysis(customer, prev_days):
    # define all input data for customer 3
    fact = customer.parsing_excel('fact')[:-2]
    plan = customer.parsing_excel('predict')
    weekdays = customer.get_weekday_vec()
    print(fact.shape, plan.shape, weekdays.shape)

    # scaling and encoding all input data
    scaler = MinMaxScaler()
    scaler.fit(fact)
    fact, plan = scaler.transform(fact), scaler.transform(plan)

    # encoding weekdays set
    enc = OneHotEncoder(handle_unknown='ignore')
    weekdays = enc.fit_transform(np.expand_dims(weekdays, axis=1)).toarray()

    # count errors for each hour
    errors = np.asarray([fact[i] / plan[i] for i in range(fact.shape[0])])
    print(errors.shape)

    # create train set
    features, targets, prediction_vec = create_dataset(errors, prev_days=prev_days)
    features = np.expand_dims(features, axis=2)
    prediction_vec = np.expand_dims(prediction_vec, axis=2)
    print(features.shape, targets.shape, prediction_vec.shape)

    # train_model
    model = network_arch(features_1=features, features_2=weekdays[prev_days+1:-2], targets=targets)
    model.fit([features, weekdays[prev_days+1:-2]], targets, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

    # predictions
    predictions = model.predict([prediction_vec, weekdays[-2:]])
    full_predictions = plan[-2:]*predictions
    write_predictions(prediction=scaler.inverse_transform(full_predictions), file_to_write='errors_predictions')

    return errors


def create_dataset(seq, prev_days):
    features = [seq[i:i+prev_days].flatten() for i in range(seq.shape[0] - prev_days + 1)]
    targets = seq[prev_days+1:]
    return np.asarray(features)[:-2], np.asarray(targets), np.asarray(features)[-2:]


def network_arch(features_1, features_2, targets):
    input_1 = Input(shape=features_1[0].shape)
    input_2 = Input(shape=features_2[0].shape)

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
    input_dense = concatenate([output_1, output_2, output_3, output_4, input_2])

    # dense layers for regression
    z = Dense(units=1000, activation='relu')(input_dense)
    output = Dense(targets.shape[1], activation='linear')(z)

    model = Model(inputs=[input_1, input_2], outputs=[output])
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=logcosh, optimizer=adam)
    return model


def write_predictions(prediction, file_to_write):

    book = xlwt.Workbook(file_to_write + '.xlsx')
    worksheet = book.add_sheet(file_to_write)
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            worksheet.write(j, i, float(prediction[i][j]))

    book.save(file_to_write + '.xlsx')
    return prediction


errors = errors_analysis(c3, prev_days=2)

