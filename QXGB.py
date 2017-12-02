# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import binom_test

from sklearn.base import BaseEstimator, RegressorMixin
from xgboost.sklearn import XGBRegressor
from functools import partial

class QXGB(BaseEstimator, RegressorMixin):
	def __init__(self, quant_alpha,quant_delta,quant_thres,quant_var, timesteps=5, 
	n_estimators = 100,max_depth = 3,reg_alpha = 5.,reg_lambda=1.0,gamma=0.5):
		self.quant_alpha = quant_alpha
		self.quant_delta = quant_delta 
		self.quant_thres = quant_thres 
		self.quant_var = quant_var 
		self.timesteps = timesteps
		#xgboost parameters 
		self.n_estimators = n_estimators 
		self.max_depth = max_depth 
		self.reg_alpha= reg_alpha 
		self.reg_lambda = reg_lambda 
		self.gamma = gamma 
		#keep xgboost estimator in memory 
		self.clf = None 

	def fit(self, X, y): 
		def quantile_loss(y_true, y_pred,_alpha,_delta,_threshold,_var): 
			x = y_true - y_pred 
			grad = (x<(_alpha-1.0)*_delta)*(1.0-_alpha)- ((x>=(_alpha-1.0)*_delta)&
									(x<_alpha*_delta) )*x/_delta-_alpha*(x>_alpha*_delta) 
			hess = ((x>=(_alpha-1.0)*_delta)& (x<_alpha*_delta) )/_delta 
			_len = np.array([y_true]).size 
			var = (2*np.random.randint(2, size=_len)-1.0)*_var 
			grad = (np.abs(x)<_threshold )*grad - (np.abs(x)>=_threshold )*var 
			hess = (np.abs(x)<_threshold )*hess + (np.abs(x)>=_threshold ) 
			return grad, hess 
		self.clf = XGBRegressor(
		objective=partial( quantile_loss,
							_alpha = self.quant_alpha,
							_delta = self.quant_delta,
							_threshold = self.quant_thres,
							_var = self.quant_var), 
							n_estimators = self.n_estimators,
							max_depth = self.max_depth,
							reg_alpha =self.reg_alpha, 
							reg_lambda = self.reg_lambda,
							gamma = self.gamma )
		self.clf.fit(X,y) 
		return self 

	def predict(self, X): 
		y_pred = self.clf.predict(X) 
		return y_pred 


	def _pre_process(self, stock):
		'''
		stock DF [n, 1] -> Stock DF [n, timesteps]

		Processa uma coluna de ativo uma DF com sequência quasi-redundantes.
		Cada coluna é a anterior um tempo a frente. 
		'''
		assert stock.shape[1] == 1 # garante que os dados são uma coluna
		stock.columns = ['price'] # renomeia a coluna para price
		stock = np.log(stock) # converte em log preço
		stock = (stock.shift(1) - stock) * 100 # dif em log preço

		# reformata para array 2D [n_samples, timesteps]
		for timestep in range(1, self.timesteps+1):
			stock['price'+str(timestep)] = stock[['price']].shift(-timestep).values

		# stock.dropna(inplace=True) # limpa NANs
		stock.fillna(stock.mean(), inplace=True)
		return stock


	def train_test_split(self, stock, n_test=500):
		'''
		Stock DF [n, timesteps] -> X_train, y_train, X_test, y_test
		X e y são Arrays 3D [n_samples, timesteps, var=1] e 
		y é X um tempo a frente
		'''
		
		stocks = self._pre_process(stock)

		# separa em variáveis dependentes e independentes
		X = stocks.iloc[:, :self.timesteps].values
		y = stocks.iloc[:, self.timesteps:].values

		# separa em treino e teste
		X_train, X_test = X[:-n_test, :], X[-n_test:, :]
		y_train, y_test = y[:-n_test, :], y[-n_test:, :]

		return X_train, y_train[:, 0], X_test, y_test[:, 0]
	
	def quantile_loss(self, y_true, y_hat, tau):
		return np.mean((tau-(y_true < y_hat)) * (y_true-y_hat))

	def score(self, X, y):
		var = self.predict(X)
		print("#Hits: ", np.mean(y < var))
		print("Quantile Loss: ", self.quantile_loss(y, var, self.quant_alpha))
