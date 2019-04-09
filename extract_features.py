import numpy as np
from pmdarima.arima import auto_arima
from sklearn.neighbors import NearestNeighbors
from nearest_neighbors_reg import NeighborsRegression
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from nn_architecture import NeuralNetworks
import xlwt
from numpy import fft
np.random.seed(1)


class ExtractFeatures:
    def __init__(self, fact, plan, weekdays, fact_previous_days=1):
        self.fact = fact
        self.plan = plan
        self.weekdays = weekdays
        self.fact_previous_days = fact_previous_days
        self.forecast_period = 48

    def find_peaks(self, x):
        mins = [x[i] for i in range(1, len(x)-1, 1) if (x[i-1]>x[i]) and (x[i+1]>x[i])]
        maxs = [x[i] for i in range(1, len(x)-1, 1) if (x[i-1]<x[i]) and (x[i+1]<x[i])]
        extremas = np.hstack([mins, maxs])
        return extremas

    # input array size (n, 24)
    def stats_features(self, array):
        max_array = np.expand_dims(np.asarray([np.max(array[i]) for i in range(array.shape[0])]), axis=1)
        min_array = np.expand_dims(np.asarray([np.min(array[i]) for i in range(array.shape[0])]), axis=1)
        median_array = np.expand_dims(np.asarray([np.median(array[i]) for i in range(array.shape[0])]), axis=1)
        mean_array = np.expand_dims(np.asarray([np.mean(array[i]) for i in range(array.shape[0])]), axis=1)
        sum_array = np.expand_dims(np.asarray([np.sum(array[i]) for i in range(array.shape[0])]), axis=1)
        peaks_array = np.expand_dims(np.asarray([self.find_peaks(array[i]).shape[0] for i in range(array.shape[0])]), axis=1)
        return max_array, min_array, median_array, mean_array, sum_array, peaks_array

    def neighbors_features(self, features_collection, neighbors_number):
        X = features_collection
        nbrs = NearestNeighbors(n_neighbors=neighbors_number, algorithm='ball_tree').fit(X)
        distance_array = nbrs.kneighbors_graph(X).toarray()
        all_neighbors = []
        for i in range(distance_array.shape[0]):
            neighbors_indexes = np.hstack(
                [self.fact[j] for j in range(distance_array.shape[1]) if (distance_array[i, j] == 1) and (i != j)])
            all_neighbors.append(neighbors_indexes)
        return np.asarray(all_neighbors)

    def ar(self, history, forecast_period):
        model = auto_arima(history, trace=True)
        model_fit = model.fit(history)
        arima_predict = model_fit.predict(n_periods=forecast_period)
        return np.asarray(arima_predict)

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

    def fourierExtrapolation(self, x, n_predict):
        n = x.size
        n_harm = 10  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        x_notrend = x - p[0] * t  # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)  # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(0, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    def create_train_set(self, scaler, ar_data_file):
        # count all stats features for fact, plan
        max_array_fact, min_array_fact, median_array_fact, mean_array_fact, sum_array_fact, peaks_array_fact = self.stats_features(self.fact)
        stats_features_fact = np.hstack([max_array_fact, min_array_fact, median_array_fact, mean_array_fact, sum_array_fact, peaks_array_fact])
        print('stats features shape', stats_features_fact.shape)

        # get neighbors features (fact.shape == neighbors_fact.shape)
        neighbors_fact = self.neighbors_features(np.hstack([max_array_fact, min_array_fact, median_array_fact, mean_array_fact, peaks_array_fact]), neighbors_number=2)

        # fact previous days set (last two are fro prediction)
        previous_fact = np.asarray([self.fact[i:i+self.fact_previous_days].flatten() for i in range(self.fact.shape[0]-self.fact_previous_days + 1)])

        # previous error
        errors = np.expand_dims(np.asarray([np.mean(np.abs(self.fact[i] - self.plan[i])) for i in range(self.fact.shape[0])]), axis=2)
        previous_errors = np.asarray([errors[i:i+self.fact_previous_days].flatten() for i in range(self.fact.shape[0]-self.fact_previous_days + 1)])

        fourier_prediction = np.asarray([self.fourierExtrapolation(x=self.fact[i:i+self.fact_previous_days].flatten(), n_predict=24)[-24:]
                              for i in range(self.fact.shape[0]-self.fact_previous_days + 1)])

        # create train features and targets for neural network architecture (-2 is because last two are for predictions)
        train_features = [np.expand_dims(previous_fact[:-2], axis=2),
                          stats_features_fact[self.fact_previous_days-1:-2],
                          self.plan[self.fact_previous_days + 1:-2],
                          self.weekdays[self.fact_previous_days + 1:-2],
                          previous_errors[:-2],
                          fourier_prediction[:-2]]

        predict_features = [np.expand_dims(previous_fact[-2:], axis=2),
                            stats_features_fact[-2:],
                            self.plan[-2:],
                            self.weekdays[-2:],
                            previous_errors[-2:],
                            fourier_prediction[-2:]]

        train_targets = self.fact[self.fact_previous_days + 1:]             # +1 for forecasting over the day
        print('targets shape', train_targets.shape)
        return train_features, train_targets, predict_features





