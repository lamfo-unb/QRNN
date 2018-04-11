# coding=utf-8
from toolz.curried import *
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam


# TODO : Implement recurrent variational dropout

def qrnn_learner(dataset, price_cols, target_col, prediction_col="prediction",
                 tau=0.05, neurons=20, lr=1e-4, batch_size=512, epochs=5):
    def to_3D(dataset):
        all_p_columns = pipe(dataset.columns,
                             filter(lambda col: reduce(lambda acc, p_col: acc or col.find(p_col) >= 0,
                                                       price_cols, False)),
                             filter(lambda col: col != target_col),
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
    model.add(LSTM(neurons, input_shape=(timesteps, n_vars)))
    model.add(Dense(1, activation=None))
    opt = Adam(lr=lr)
    model.compile(loss=quantile_loss, optimizer=opt)

    # train model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

    def p(new_dataset):
        x_new = _3Dnator(new_dataset)
        return new_dataset.assign(**{prediction_col: model.predict(x_new)})

    return p, p(dataset)
