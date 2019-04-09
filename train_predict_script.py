from nn_architecture import NeuralNetworks
from extract_features import ExtractFeatures
from nearest_neighbors_reg import NeighborsRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.stattools import ccf, acf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import tspy

np.random.seed(1)

# define customer's objects
c3 = NeighborsRegression(file_list=['consumer_3/consumer3_2017.xlsx',
                                    'consumer_3/consumer3_2018.xlsx',
                                    'consumer_3/consumer3_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 4, 11])

c2 = NeighborsRegression(file_list=['consumer_2/consumer2_2017.xlsx',
                                    'consumer_2/consumer2_2018.xlsx',
                                    'consumer_2/consumer2_2019.xlsx'],
                                    date_from=[2017, 1, 1], date_till=[2019, 4, 11])


def train_customer_model(customer, file_to_write, ar_data_file):
    # define all input data for customer 3
    fact = customer.parsing_excel('fact')[:-2]
    plan = customer.parsing_excel('predict')

    # coef = ccf(fact.flatten(), plan[:-2].flatten())
    # plt.plot(coef[-300:])
    # plt.show()
    # print('autocorelation', coef)

    weekdays = customer.get_weekday_vec()

    # scaling and encoding all input data
    scaler = MinMaxScaler()
    scaler.fit(fact)
    fact, plan = scaler.transform(fact), scaler.transform(plan)

    enc = OneHotEncoder(handle_unknown='ignore')
    weekdays = enc.fit_transform(np.expand_dims(weekdays, axis=1)).toarray()
    print(weekdays)

    # extract all features for customer
    extracted_features = ExtractFeatures(fact=fact,
                                         plan=plan,
                                         weekdays=weekdays)

    # get train and predict set for customer
    train_features, train_targets, predict_features = extracted_features.create_train_set(scaler=scaler, ar_data_file=ar_data_file)

    # all_poymano = []
    # for i in range(5):
    #     train_features, train_targets = [features[j][:-i-1] for j in range(len(features))], targets[:-i-1]
    #     for l in train_features:
    #         print(l.shape)
    #     test_vector = [np.expand_dims(features[j][-i-1], axis=0) for j in range(len(features))]
    #     for k in test_vector:
    #         print(k.shape)

    nn_model = NeuralNetworks(neurons_1=200,
                              neurons_2=300,
                              neurons_3=300,
                              dropout=0.5).nn_model(train_features[0],
                                                    train_features[1],
                                                    train_features[2],
                                                    train_features[3],
                                                    train_features[4],
                                                    train_features[5],
                                                    train_targets)
    # train model
    nn_model.fit(train_features, train_targets, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

    # predict and write for the next day
    predictions = nn_model.predict(predict_features)
    extracted_features.write_predictions(prediction=scaler.inverse_transform(predictions), file_to_write=file_to_write)
    #
    # test_predictions = nn_model.predict(test_vector)
    # test_predictions = np.reshape(np.asarray(test_predictions[0]), (1, len(test_predictions[0])))
    # print(test_predictions[0])
    # all_poymano.append(extracted_features.count_error(vector_fact=targets[-i-1],
    #                                                   vector_plan=plan[-i-1],
    #                                                   vector_predict=(test_predictions + plan[-i-1])/2))
    return predictions


def arima_predictions(customer, forecast_period):
    fact = customer.parsing_excel('fact')[:-2].flatten()
    model = ARIMA(fact, order=(48, 1, 0))
    model_fit = model.fit(disp=0)
    arima_predict = model_fit.predict(start=len(fact), end=len(fact) + forecast_period - 1)
    plt.plot(arima_predict)
    plt.show()
    return np.asarray(arima_predict)


def test_tspy(customer, prev_days):

    # hyperparmeters
    n = 48
    HIDDEN_UNITS = [100]
    NUM_EPOCHS = 5000

    fact = customer.parsing_excel('fact')[:-2]
    scaler = MinMaxScaler()
    scaler.fit(fact)
    fact = scaler.transform(fact)

    # make train set for dnn
    features = np.asarray([fact[i:i+prev_days].flatten() for i in range(fact.shape[0] - prev_days - 1)])
    targets = np.asarray(fact[prev_days+1:])

    train_features, test_features, train_targets, test_targets = features[:-10], features[-10:], targets[:-10], targets[-10:]

    # train dnn
    dnn = tspy.model.ar.DNN(hidden_units=HIDDEN_UNITS, window=n).fit(train_features, train_targets, num_epochs=NUM_EPOCHS)
    predictions = dnn.predict(test_features)

    # train sklearn regressor
    # hyperparameters
    REGRESSOR_TYPE = 'ridge'

    # regressor / model
    reg = tspy.model.ar.SKReg(regressor_type=REGRESSOR_TYPE, window=n).fit(train_features, train_targets)
    predictions_reg = reg.predict(test_features)

    return scaler.inverse_transform(predictions), scaler.inverse_transform(predictions_reg)


# # testing new package
# c3_predictions, c3_predictions_reg = test_tspy(customer=c3, prev_days=2)
# plt.plot(c3_predictions)
# plt.show()
#
# plt.plot(c3_predictions_reg)
# plt.show()
# print(c3_predictions, c3_predictions_reg)


# get ARIMA predictions
arima_res = arima_predictions(customer=c3, forecast_period=48)

# # get prediction for each customer
# c3_predictions = train_customer_model(customer=c3, file_to_write='c3_predictions', ar_data_file='ar_features.txt')
# c2_predictions = train_customer_model(customer=c2, file_to_write='c2_predictions', ar_data_file='ar_features.txt')
# print(c3_predictions)
# print(c2_predictions)
