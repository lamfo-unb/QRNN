# coding=utf-8
from toolz.curried import *
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras import backend as K
from qrnn.evaluation import quantile_loss_evaluator


def qrnn_learner(dataset, price_cols, target_col, target_lag, prediction_col="prediction",
                 tau=0.05, neurons=20, lr=1e-4, batch_size=512, epochs=5, dropout=0.1,
                 stocastic_pass=100, return_variance=True):

    def to_3D(dataset):
        all_p_columns = pipe(dataset.columns,
                             filter(lambda col: reduce(lambda acc, p_col: acc or col.find(p_col) >= 0,
                                                       price_cols, False)),
                             filter(lambda col: col != target_col),
                             filter(lambda col: not col.endswith(str(target_lag))),
                             list)

        def p(new_data):
            return new_data[all_p_columns].values.reshape(-1,
                                                          int(len(all_p_columns) / len(price_cols)),
                                                          len(price_cols))

        return p, p(dataset)

    def quantile_loss(y_true, y_pred):
        ro = tau - tf.cast(tf.greater(y_pred, y_true), tf.float32)
        return tf.reduce_mean(ro * (y_true - y_pred))

    _3Dnator, x_train = to_3D(dataset)
    y_train = dataset[[target_col]].values
    n_samples, timesteps, n_vars = x_train.shape

    # build model
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(timesteps, n_vars), dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(1, activation=None))
    model.add(Dropout(dropout))
    opt = Adam(lr=lr)
    model.compile(loss=quantile_loss, optimizer=opt)

    # train model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    def p(new_dataset):
        x_new = _3Dnator(new_dataset)

        if return_variance:
            predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

            y_hat_mc = np.array([predict_stochastic([x_new, 1])
                                 for _ in range(stocastic_pass)]).reshape(-1, x_new.shape[0]).T

            y_hat_test_mean = np.mean(y_hat_mc, axis=1)
            y_hat_test_variance = np.var(y_hat_mc, axis=1)

            return new_dataset.assign(**{prediction_col: y_hat_test_mean,
                                         prediction_col + "_var": y_hat_test_variance})
        else:
            return new_dataset.assign(**{prediction_col: model.predict(x_new)})

    return p, p(dataset)


def adaptive(dataset, price_cols, target_col, prediction_col, tau):
    from scipy.stats import norm
    from scipy.optimize import brent
    # initializing VaR arrays

    VaR_train = np.zeros(len(dataset))

    VaR_train[0] = - norm.ppf(tau) * np.std(dataset[price_cols])

    def adaptative(beta):
        for i in range(1, len(dataset)):
            VaR_train[i] = VaR_train[i - 1] + beta * ((dataset[price_cols].iloc[i - 1] < - VaR_train[i - 1]) - tau)

        VaRdataset = dataset.assign(**{prediction_col: VaR_train})
        return quantile_loss_evaluator(VaRdataset, price_cols, prediction_col, tau)

        # beta that minimizes loss

    beta_opt = brent(adaptative, brack=(-100, 100))

    def p(new_dataset):
        VaR = np.zeros(len(new_dataset))
        VaR[0] = - norm.ppf(tau) * np.std(new_dataset[price_cols])
        for i in range(1, len(new_dataset)):
            VaR[i] = VaR[i - 1] + beta_opt * ((new_dataset[price_cols].iloc[i - 1] < - VaR[i - 1]) - tau)

        return new_dataset.assign(**{prediction_col: VaR})

    return p, p(dataset)
