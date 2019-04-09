from nn_architecture import NeuralNetworks
from extract_features import ExtractFeatures
from nearest_neighbors_reg import NeighborsRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import ccf
from statsmodels.tsa.ar_model import AR
import statsmodels.api as sm
import pickle
np.random.seed(1)

# define customer's objects
c3 = NeighborsRegression(file_list=['consumer_3/consumer3_2017.xlsx',
                                    'consumer_3/consumer3_2018.xlsx',
                                    'consumer_3/consumer3_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 3, 29])

c2 = NeighborsRegression(file_list=['consumer_2/consumer2_2017.xlsx',
                                    'consumer_2/consumer2_2018.xlsx',
                                    'consumer_2/consumer2_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 3, 29])


def fourierExtrapolation(x, n_predict):
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


def ar(history, forecast_period, p, d, q):
    model = SARIMAX(history, order=(p, d, q), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    arima_predict = model_fit.predict(start=len(history), end=len(history)+forecast_period-1)
    return arima_predict


def test(customer, file_to_write, prev_days):

    fact = customer.parsing_excel('fact')[:-2]
    plan = customer.parsing_excel('predict')

    coef = ccf(fact.flatten(), plan[:-2].flatten())
    print(coef)

    fourier_prediction = np.asarray([fourierExtrapolation(x=fact[i:i + prev_days].flatten(), n_predict=24)[-24:]
                                     for i in range(fact.shape[0] - prev_days + 1)])

    # resDiff = sm.tsa.arma_order_select_ic(fact.flatten(), max_ar=10, max_ma=10, ic='aic', trend='c')
    # print('ARMA(p,q) =', resDiff['aic_min_order'], 'is the best.')

    ar_data = np.stack([ar(history=fact[:i+prev_days].flatten(), forecast_period=48, p=7, d=1, q=7) for i in range(fact.shape[0] - prev_days - 1)])
    plt.plot(ar_data[-100:].flatten())

    with open('ar_data.txt', 'wb') as f:
        pickle.dump(ar_data, f)

    with open('ar_data.txt', 'rb') as f:
        ar_data = pickle.load(f)

    features = []
    for i in range(fact.shape[0] - prev_days + 1):
        prev_mean = [sum(j)/prev_days for j in zip(*fact[i+prev_days-2:i+prev_days])]
        features.append(prev_mean)

    targets = fact[prev_days + 1:]
    plan_targets = plan[prev_days + 1:-2]
    features = (plan_targets + fourier_prediction[:-2] + ar_data[:, 24:])/3

    return features, targets, plan_targets, fourier_prediction[:-2]


features, targets, plan_targets, fourier = test(customer=c3, file_to_write='', prev_days=14)
c3_error = [c3.count_error(vector_fact=targets[i], vector_plan=plan_targets[i], vector_predict=features[i]) for i in range(targets.shape[0])]
print(c3_error[-10:])

# plt.plot(c3_error[-70:])
# plt.show()
