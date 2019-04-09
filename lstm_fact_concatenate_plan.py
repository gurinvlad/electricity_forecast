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
        ar = np.stack([self.AR(history=data[i:i+n_previous_days].flatten(), forecast_period=forecast_period) for i in range(data.shape[0] - n_previous_days - 1)])
        print(ar.shape)
        return ar

    def lstm_model(self, x_lstm, x_plan, x_weekdays, targets):

        print(x_weekdays.shape)
        lstm_input = Input(shape=np.expand_dims(x_lstm, axis=2)[0].shape)
        plan_input = Input(shape=x_plan[0].shape)
        weekdays_input = Input(shape=x_weekdays[0].shape)
        print(weekdays_input.shape)

        x = LSTM(100, return_sequences=True)(lstm_input)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = LSTM(100, return_sequences=False)(x)
        x = BatchNormalization()(x)
        lstm_output = Dropout(0.3)(x)

        input_dense = concatenate([lstm_output, plan_input, weekdays_input])

        x = Dense(100, activation='relu')(input_dense)
        x = Dropout(0.3)(x)
        output = Dense(targets.shape[1], activation='linear', use_bias=False)(x)
        model = Model(inputs=[lstm_input, plan_input, weekdays_input], outputs=[output])
        model.compile(loss='mse', optimizer='adam')
        return model

    def save_model(self, model_file, model):
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

    def previous_fact_set(self, data, n_prev_days):
        features = [data[i:i+n_prev_days].flatten() for i in range(data.shape[0] - n_prev_days - 1)]
        targets = [data[i+n_prev_days+1] for i in range(data.shape[0] - n_prev_days - 1)]
        return np.asarray(features), np.asarray(targets)

    def train_model(self, n_previous_days, file_to_write):

        fact = self.parsing_excel('fact')[:-2]                      # because of NaN
        plan = self.parsing_excel('predict')
        weekdays = np.expand_dims(self.get_weekday_vec(), axis=1)


        # SCALING
        scaler = StandardScaler()
        scaler.fit(fact)

        # get fact set
        fact = scaler.transform(fact)
        features, targets = self.previous_fact_set(data=fact, n_prev_days=n_previous_days)
        print(features.shape, targets.shape)

        # get plan set
        plan = scaler.transform(plan)[int(fact.shape[0] - features.shape[0]):-2]
        print(plan.shape)

        # weekdays one hot encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        weekdays = enc.fit_transform(weekdays).toarray()
        weekdays = weekdays[int(fact.shape[0] - features.shape[0]):-2]
        print(weekdays.shape)

        # AR RPEPROCESSING
        # ar_data = self.get_ar_predictions(data=fact, n_previous_days=n_previous_days, forecast_period=48)
        # print(ar_data[:, 24:].shape)

        print(plan[-1].shape)
        # Train model
        lstm_model = self.lstm_model(x_lstm=features, x_plan=plan, x_weekdays=weekdays, targets=targets)
        lstm_model.fit([np.expand_dims(features, axis=2), plan, weekdays], targets, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

        # predictions
        predict_fact = np.reshape(fact[-n_previous_days:].flatten(), (1, n_previous_days*24, 1))
        print(predict_fact.shape, plan[-1].shape, weekdays[-1].shape)
        predictions = lstm_model.predict([predict_fact, np.expand_dims(plan[-1], axis=0), np.expand_dims(weekdays[-1], axis=0)])
        self.write_predictions(features[-2:], file_to_write='features')
        self.write_predictions(prediction=scaler.inverse_transform(predictions), file_to_write=file_to_write)
        return lstm_model


model_c3 = MainModel(file_list=['consumer3/consumer3_2017.xlsx',
                                'consumer3/consumer3_2018.xlsx',
                                'consumer3/consumer3_2019.xlsx'],
                     look_back_days=4,
                     date_from=[2017, 1, 1], date_till=[2019, 3, 17])

model_c3.train_model(n_previous_days=4, file_to_write='lstm_fact_concatenate_plan_c3')
# fact = model_c3.parsing_excel('fact')[-50:-2]
# print(fact.shape)
# dates_range, weekdays = model_c3.get_weekday_vec()
# time_series = pd.Series(np.reshape(fact, (fact.shape[0]*fact.shape[1])))
# time_series.index = pd.DatetimeIndex(freq='H', start=0, periods=len(time_series))
# print(time_series)
# result = seasonal_decompose(time_series, model='multiplicative')
# resid = result.resid
# print(resid)
# result.plot()
# pyplot.show()


model_c2 = MainModel(file_list=['consumer2/consumer2_2017.xlsx',
                                'consumer2/consumer2_2018.xlsx',
                                'consumer2/consumer2_2019.xlsx'],
                     look_back_days=4,
                     date_from=[2017, 1, 1], date_till=[2019, 3, 17])

model_c2.train_model(n_previous_days=4, file_to_write='lstm_fact_concatenate_plan_c2')
