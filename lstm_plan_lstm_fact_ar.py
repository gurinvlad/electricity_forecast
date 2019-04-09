import numpy as np
import scipy
import os
import keras
import pandas as pd
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Flatten, Conv2D, BatchNormalization
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras import optimizers, regularizers
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
import xlwt
import datetime
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from pmdarima.arima import auto_arima
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder

np.random.seed(1)


class MainModel:

    def __init__(self, file_list, look_back_days, date_from, date_till):
        self.file_list = file_list
        self.look_back = int(look_back_days)
        self.date_from = date_from
        self.date_till = date_till

    # one year data each month
    def electricity_each_year(self, year_data, column_name):
        data_electricity = pd.read_excel(year_data, year_data.sheet_names[0])[column_name]
        for name in year_data.sheet_names[1:]:
            data_electricity = pd.concat([data_electricity, pd.read_excel(year_data, name)[column_name]],
                                         ignore_index=True)
        return data_electricity

    def parsing_excel(self, column_name):
        data_list = []
        for file in self.file_list:
            data = pd.ExcelFile(file)
            data = self.electricity_each_year(data, column_name=column_name)
            data_list.append(data)

        data_electricity = pd.concat(tuple(data_list))
        data_electricity = data_electricity.values
        data_electricity = data_electricity.astype('float64')
        data_electricity = np.asarray(data_electricity)
        data_electricity = np.reshape(data_electricity, (int(data_electricity.shape[0] / 24), 24))
        print(data_electricity)

        return data_electricity

    def divide_dataset(self, data):
        part_1 = data[:, :8]
        part_2 = data[:, 8:17]
        part_3 = data[:, 17:]
        return part_1, part_2, part_3

    def get_weekday_vec(self):
        start_date = datetime.date(self.date_from[0], self.date_from[1], self.date_from[2])
        end_date = datetime.date(self.date_till[0], self.date_till[1], self.date_till[2])

        weekdays = []
        dates_range = [start_date + datetime.timedelta(n) for n in range(int((end_date - start_date).days))]

        for i in range(len(dates_range)):
            weekday = datetime.date(dates_range[i].year, dates_range[i].month, dates_range[i].day).weekday()
            weekdays.append(weekday)

        return np.asarray(weekdays)

    def count_error(self, vector_fact, vector_plan, vector_predict):
        s_pl = np.sum(np.abs(vector_plan - vector_fact))
        s_pr = np.sum(np.abs(vector_predict - vector_fact))
        square = ((s_pl - s_pr)/s_pl)*100
        return square

    def AR(self, history, forecast_period):
        model = ARIMA(history, order=(1, 0, 1))
        model_fit = model.fit()
        arima_predict = model_fit.predict(start=len(history), end=len(history) + (forecast_period-1))
        return np.asarray(arima_predict)

    def get_ar_predictions(self, data, n_previous_days, forecast_period):
        ar = np.stack([self.AR(history=data[i:i+n_previous_days].flatten(), forecast_period=forecast_period) for i in range(data.shape[0] - n_previous_days + 1)])
        return ar

    def lstm_model(self, model_file, train_x, train_y):

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        model = Sequential()
        model.add(LSTM(100, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(LSTM(100, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(train_y.shape[1], activation='linear', use_bias=False))
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_x, train_y, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

        model_json = model.to_json()
        with open(model_file + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_file + '.h5')
        print('Saved model' + model_file + 'to disk')

        return model

    def dense_model(self, model_file, train_x, train_y):

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(train_y.shape[1], activation='linear', use_bias=False))
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_x, train_y, batch_size=128, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

        model_json = model.to_json()
        with open(model_file + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_file + '.h5')
        print('Saved model' + model_file + 'to disk')

        return model

    def write_predictions(self, prediction, file_to_write):

        book = xlwt.Workbook(file_to_write + '.xlsx')
        worksheet = book.add_sheet(file_to_write)
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                worksheet.write(j, i, float(prediction[i][j]))

        book.save(file_to_write + '.xlsx')
        return prediction

    def train_model(self, lstm_dim, n_previous_days, file_to_write):

        fact = self.parsing_excel('fact')[:-2]/10000
        plan = self.parsing_excel('predict')/10000
        weekdays = np.expand_dims(self.get_weekday_vec(), axis=1)
        enc = OneHotEncoder(handle_unknown='ignore')
        weekdays = enc.fit_transform(weekdays).toarray()
        print(weekdays, weekdays.shape)
        print(fact.shape, plan.shape, weekdays.shape)
        sum_fact = np.stack([sum(fact[i]) for i in range(fact.shape[0])])
        sum_plan = np.stack([sum(plan[i]) for i in range(plan.shape[0])])

        # AR RPEPROCESSING
        ar_data = self.get_ar_predictions(data=fact[lstm_dim - n_previous_days - 1:], n_previous_days=n_previous_days, forecast_period=48)

        # MAIN MODEL
        features = np.hstack([ar_data[:, 24:], plan[lstm_dim:], weekdays[lstm_dim:]])
        train_features = features[:-2]
        train_targets = fact[lstm_dim:]

        # test_features = features[-12:-2]
        # test_targets = fact[-10:]
        # test_plan = plan[-12:-2]

        # train main model
        main_lstm_model = self.lstm_model('main_lstm_model', train_x=train_features, train_y=train_targets)

        # # test model
        # test_predictions = main_lstm_model.predict(np.expand_dims(test_features, axis=2))
        # error = [self.count_error(vector_fact=test_targets[i], vector_predict=test_predictions[i], vector_plan=test_plan[i]) for i in range(test_features.shape[0])]
        # print('error', error)

        # predictions
        predict_features = np.expand_dims(features[-2:], axis=2)
        predictions = main_lstm_model.predict(predict_features)
        self.write_predictions(features[-2:], file_to_write='features')
        self.write_predictions(prediction=predictions*10000, file_to_write=file_to_write)
        return predictions
