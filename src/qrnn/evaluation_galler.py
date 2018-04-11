# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:24:24 2018

@author: Leonardo Galler

Create two assesment functions:
    1 - quantile_loss_evaluator(df: DataFrame, predict_col: str, target_col: str, tau: float):
        return {"quantile_loss": quantile_loss}
    input: 
        df: DataFrame
        predict_col: str
        target_col: str
        tau: float
    output:
        dictionary = {"quantile_loss": quantile_loss}
    
    
    2 - proportion_of_hits_evaluator(df: DataFrame, predict_col: str, target_col: str):
        return {"proportion_of_hits": proportion_of_hits}
    input:
        df: DataFrame
        predict_col: str
        target_col: str
    output:
        dictionary = {"proportion_of_hits": proportion_of_hits}
        
"""
import pandas as pd
import numpy as np

# Structure for quantile loss evaluator
def quantile_loss_evaluator( df , predict_col , target_col , tau ):
    
    # assigning the values
    y_hat = df[predict_col]
    y_tru = df[target_col]
    # Returning a dictionary with the quartile and the loss
    return np.mean((tau - (y_tru < y_hat )) * (y_tru - y_hat))



# Structure for proportion of hits evaluator
def proportion_of_hits_evaluator(df , predict_col , target_col ):
    
    return np.mean(df[predict_col] > df[target_col])