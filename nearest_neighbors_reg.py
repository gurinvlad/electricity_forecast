import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from keras import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, LSTM, BatchNormalization
import torch
import torch.nn as nn
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import xlwt
from scipy.optimize import fmin


np.random.seed(1)


class NeighborsRegression:

    def __init__(self, file_list, date_from, date_till):
        self.file_list = file_list
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

    def find_nearest_neighbors(self, collection, neighbors_number):
        X = collection
        nbrs = NearestNeighbors(n_neighbors=neighbors_number, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        distance_array = nbrs.kneighbors_graph(X).toarray()
        return distances, indices, distance_array

    def regression_model(self, train_x, train_y, model_file):

        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        model = Sequential()
        model.add(Dense(100, activation='relu', input_shape=train_x[0].shape))
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(train_y.shape[1], activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_x, train_y, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

        model_json = model.to_json()
        with open(model_file + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_file + '.h5')
        print('Saved model' + model_file + 'to disk')

        return model

    def find_peaks(self, x):
        mins = [x[i] for i in range(1, len(x)-1, 1) if (x[i-1]>x[i]) and (x[i+1]>x[i])]
        maxs = [x[i] for i in range(1, len(x)-1, 1) if (x[i-1]<x[i]) and (x[i+1]<x[i])]
        extremas = np.hstack([mins, maxs])
        return extremas

    # create data concatenates previous fact(f), previous plan(p1), next plan(p2) and weekdays(w);
    # fact is for the targets
    def create_collection(self, fact, f, p1, p2, w):
        median_p2 = np.expand_dims(np.asarray([np.median(p2[i]) for i in range(p2.shape[0])]), axis=1)
        median_f = np.expand_dims(np.asarray([np.median(f[i]) for i in range(f.shape[0])]), axis=1)
        median_p1 = np.expand_dims(np.asarray([np.median(p1[i]) for i in range(p1.shape[0])]), axis=1)
        mean_p2 = np.expand_dims(np.asarray([np.mean(p2[i]) for i in range(p2.shape[0])]), axis=1)
        mean_f = np.expand_dims(np.asarray([np.mean(f[i]) for i in range(f.shape[0])]), axis=1)
        mean_p1 = np.expand_dims(np.asarray([np.mean(p1[i]) for i in range(p1.shape[0])]), axis=1)
        sum_p2 = np.expand_dims(np.asarray([np.sum(p2[i]) for i in range(p2.shape[0])]), axis=1)
        sum_p1 = np.expand_dims(np.asarray([np.sum(p1[i]) for i in range(p1.shape[0])]), axis=1)
        sum_f = np.expand_dims(np.asarray([np.sum(f[i]) for i in range(f.shape[0])]), axis=1)
        max_p2 = np.expand_dims(np.asarray([np.max(p2[i]) for i in range(p2.shape[0])]), axis=1)
        min_p2 = np.expand_dims(np.asarray([np.min(p2[i]) for i in range(p2.shape[0])]), axis=1)
        max_f = np.expand_dims(np.asarray([np.max(f[i]) for i in range(f.shape[0])]), axis=1)
        min_f = np.expand_dims(np.asarray([np.min(f[i]) for i in range(f.shape[0])]), axis=1)
        max_p1 = np.expand_dims(np.asarray([np.max(p1[i]) for i in range(p1.shape[0])]), axis=1)
        min_p1 = np.expand_dims(np.asarray([np.min(p1[i]) for i in range(p1.shape[0])]), axis=1)
        error = np.expand_dims(np.asarray([np.mean(np.abs(f[i]-p1[i])/p1[i]) for i in range(f.shape[0])]), axis=1)

        mins_maxs_p1 = np.expand_dims(np.asarray([self.find_peaks(p1[i])[0] for i in range(p1.shape[0])]), axis=1)
        mins_maxs_p2 = np.expand_dims(np.asarray([self.find_peaks(p2[i])[0] for i in range(p2.shape[0])]), axis=1)
        mins_maxs_f = np.expand_dims(np.asarray([self.find_peaks(f[i])[0] for i in range(f.shape[0])]), axis=1)

        collection = np.hstack([f, p2, w])
        collection_features = np.hstack([mean_f, median_f, sum_f, max_f, min_f, mean_p2, median_p2, sum_p2, max_p2, min_p2, w])
        targets = fact[2:]
        return collection, collection_features, targets

    # making set consisted of neighbors previous fact
    def create_neighbors_set(self, fact, dist_array):
        all_neighbors = []
        for i in range(dist_array.shape[0]):
            neighbors_indexes = np.hstack([fact[j] for j in range(dist_array.shape[1]) if (dist_array[i, j] == 1) and (i!=j)])
            all_neighbors.append(neighbors_indexes)
        return np.asarray(all_neighbors)

    def write_predictions(self, prediction, file_to_write):
        book = xlwt.Workbook(file_to_write + '.xlsx')
        worksheet = book.add_sheet(file_to_write)
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                worksheet.write(j, i, float(prediction[i][j]))

        book.save(file_to_write + '.xlsx')
        return prediction

    def count_error(self, vector_fact, vector_plan, vector_predict):
        s_pl = np.sum(np.abs(vector_plan - vector_fact))
        s_pr = np.sum(np.abs(vector_predict - vector_fact))
        square = ((s_pl - s_pr)/s_pl)*100
        return square


customer_2 = NeighborsRegression(file_list=['consumer_2/consumer2_2017.xlsx',
                                            'consumer_2/consumer2_2018.xlsx',
                                            'consumer_2/consumer2_2019.xlsx'],
                                 date_from=[2017, 1, 1], date_till=[2019, 3, 16])

customer_3 = NeighborsRegression(file_list=['consumer_3/consumer3_2017.xlsx',
                                            'consumer_3/consumer3_2018.xlsx',
                                            'consumer_3/consumer3_2019.xlsx'],
                                 date_from=[2017, 1, 1], date_till=[2019, 3, 16])


def train_predict_cutomer(fact, plan, weekdays, customer, model_file, file_to_write, neighbors_number):
    # scaling data and encoding weekdays

    scaler = StandardScaler()
    scaler.fit(fact)
    fact = scaler.transform(fact)
    plan = scaler.transform(plan)

    enc = OneHotEncoder(handle_unknown='ignore')
    weekdays = enc.fit_transform(np.expand_dims(weekdays, axis=1)).toarray()

    # create collection for finding neighbors
    print(fact.shape, plan[:-2].shape, plan[2:].shape, weekdays[2:].shape)
    collection, collection_features, targets = customer.create_collection(f=fact, p1=plan[:-2], p2=plan[2:], w=weekdays[2:], fact=fact)
    distances, indices, distance_array = customer.find_nearest_neighbors(collection=collection, neighbors_number=neighbors_number)

    # return all neighbors vectors for each target
    all_neighbors = customer.create_neighbors_set(dist_array=distance_array, fact=fact)

    # create train and predict sets, adding features for targets
    train_set = np.hstack([collection_features[:-2], all_neighbors[:-2]])
    print(train_set.shape)
    predict_set = np.hstack([collection_features[-2:], all_neighbors[-2:]])

    # train regression model
    reg_model = customer.regression_model(train_x=train_set, train_y=targets, model_file=model_file)
    predictions = reg_model.predict(np.expand_dims(predict_set, axis=2))
    customer.write_predictions(scaler.inverse_transform(predictions), file_to_write=file_to_write)
    return predictions


def test(customer, neighbors, test_number, model_file, file_to_write):

    fact = customer.parsing_excel('fact')[:-2]
    plan = customer.parsing_excel('predict')
    weekdays = customer.get_weekday_vec()
    all_errors = []
    for n in range(2, neighbors, 1):
        mean_error = []
        for i in range(test_number):
            train_fact, train_plan, train_weekdays = fact[:-2-i], plan[:-2-i], weekdays[:-2-i]

            # get prediction and count error
            prediction = train_predict_cutomer(customer=customer,
                                               model_file=model_file,
                                               file_to_write=file_to_write,
                                               neighbors_number=n,
                                               fact=train_fact, plan=train_plan, weekdays=train_weekdays)
            mean_error.append(customer.count_error(vector_fact=fact[-i-1], vector_plan=plan[-2-i-1], vector_predict=prediction[1]))
        all_errors.append(mean_error)
    return all_errors

#
# errors_c2 = test(customer=customer_2, neighbors=10, test_number=10, model_file='c2', file_to_write='c2')
# errors_c3 = test(customer=customer_3, neighbors=10, test_number=10, model_file='c3', file_to_write='c3')
# print('error_c2 = ', errors_c2)
# print('error_c3 = ', errors_c3)

# train_predict_cutomer(customer_3,
#                       model_file='reg_model_c3',
#                       file_to_write='reg_model_results_c3',
#                       neighbors_number=5)





