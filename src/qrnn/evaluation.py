import numpy as np
from toolz.curried import *

@curry
def quantile_loss_evaluator(df, predict_col, target_col, tau):
    y_true = df[[target_col]].values
    y_hat = df[[predict_col]].values
    return np.mean((tau - (y_true < y_hat)) * (y_true - y_hat))

@curry
def proportion_of_hits_evaluator(df, predict_col, target_col):
    y_true = df[[target_col]].values
    y_hat = df[[predict_col]].values
    return np.mean(y_hat > y_true)
