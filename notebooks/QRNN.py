# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell, BasicLSTMCell, DropoutWrapper

class QRNN(object):
	"""docstring for QRNN"""
	def __init__(self, timesteps, tau, lr=0.001, neurons=100, keep_prob=.5, 
				 iterations = 5000, batch_size = 50):
		
		self.timesteps = timesteps
		self.tau = tau
		self.lr = lr
		self.neurons = neurons
		self.keep_prob = keep_prob
		self.iterations = iterations
		self.batch_size = batch_size
		self.trained = False


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


	def train_test_split(self, stock, n_test=100):
		'''
		Stock DF [n, timesteps] -> X_train, y_train, X_test, y_test
		X e y são Arrays 3D [n_samples, timesteps, var=1] e 
		y é X um tempo a frente
		'''
		
		stocks = self._pre_process(stock)

		# separa em variáveis dependentes e independentes
		X = stocks.iloc[:, :self.timesteps].values
		y = stocks.iloc[:, 1:].values # y é X um tempo a frente
		
		# reformata para arrays 3D [n_samples, timesteps, var=1]
		X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # ad dimensão
		y = np.reshape(y, (y.shape[0], y.shape[1], 1)) # ad dimensão

		# separa em treino e teste
		X_train, X_test = X[:-n_test, :, :], X[-n_test:, :, :]
		y_train, y_test = y[:-n_test, :, :], y[-n_test:, :, :]

		return X_train, y_train, X_test, y_test


	def train(self, X_train, y_train):
		'''
		X_train, y_train -> None
		X_train, y_train são Arrays 3D [n, timesteps, var=1]

		Treina uma rede neural recorrente com a função objetivo da regressão quantílica
		'''

		# constroi o grafo tensorflow
		self.graph = tf.Graph()
		with self.graph.as_default():

			# placeholders
			self.tf_X = tf.placeholder(tf.float32, [None, self.timesteps, 1], name='X')
			self.tf_y = tf.placeholder(tf.float32, [None, self.timesteps, 1], name='y')

			# camada recorrente
			cell = GRUCell(num_units=self.neurons, activation=tf.nn.elu)
			cell = DropoutWrapper(cell,
									input_keep_prob=1.0,
									output_keep_prob=self.keep_prob,
									state_keep_prob=1.0,
									variational_recurrent=False,
									input_size=1,
									dtype=tf.float32)

			outputs, states = tf.nn.dynamic_rnn(cell, self.tf_X, dtype=tf.float32)

			# camada de saída
			stacked_outputs = tf.reshape(outputs, [-1, self.neurons]) # empilha os estados recorrentes
			stacked_outputs = tf.layers.dense(stacked_outputs, 1, activation=None) # projeta
			self.net_outputs = tf.reshape(stacked_outputs, [-1, self.timesteps, 1]) # reformata para [n, timesteps, var=1]

			# custo da regressão quantílica
			ro = self.tau - tf.cast(tf.greater(self.net_outputs, self.tf_y), tf.float32)
			loss = tf.reduce_mean( ro * (self.tf_y - self.net_outputs) ) 

			# otimização
			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

			# helpers
			init = tf.global_variables_initializer()
			self.saver = tf.train.Saver() # para salvar o modelo treinado


		# execução do grafo
		with tf.Session(graph=self.graph) as sess:
			init.run()

			# loop de treino
			for step in range(self.iterations+1):
				# monta o mini-lote
				offset = (step * self.batch_size) % (y_train.shape[0] - self.batch_size)
				X_batch = X_train[offset:(offset + self.batch_size)]
				y_batch = y_train[offset:(offset + self.batch_size)]

				# roda uma iteração de treino
				sess.run(optimizer, feed_dict={self.tf_X: X_batch, self.tf_y: y_batch})

				# printa infos de treino
				if step % 1000 == 0:
					shuffle_mask = np.arange(0, X_train.shape[0]) # cria array de 0 a n_train
					np.random.shuffle(shuffle_mask) # embaralha o array acima

					# embaralha X e y consistentemente
					X_train = X_train[shuffle_mask]
					y_train = y_train[shuffle_mask]

					train_mse = loss.eval(feed_dict={self.tf_X: X_train, self.tf_y: y_train})
					print(step, "\tCusto de treinamento:", np.sqrt(train_mse))

			if not os.path.exists('tmp'):
			    os.makedirs('tmp')
			self.saver.save(sess, "./tmp/QRNN.ckpt")
		
		
		self.trained = True # marca o final do treinamento
		return None


	def predict(self, X_test, MC_steps = 20):
		assert self.trained
		
		with tf.Session(graph=self.graph) as sess:
			self.saver.restore(sess, "./tmp/QRNN.ckpt")
			preds = np.array([sess.run(self.net_outputs, feed_dict={self.tf_X: X_test})[:, 0, 0]
					for _ in range(MC_steps)])

			pred_mean = np.mean(preds, axis=0)
			pred_var = np.var(preds, axis=0)

		return pred_mean, pred_var

	def quantile_loss(self, y_true, y_hat, tau):
		return np.mean((tau-(y_true < y_hat)) * (y_true-y_hat))

	def score(self, X, y):
		var_mean, _ = self.predict(X)
		y = y[:, 0, 0]
		print("#Hits: ", np.mean(y < var_mean))
		print("Quantile Loss: ", self.quantile_loss(y, var_mean, self.tau))


def main():
	data  = pd.read_csv('SP500.csv', usecols=['Adjusted Close'])

	model = QRNN(timesteps=15, tau=.05, lr=0.005, neurons=100, iterations=2500)
	
	X_train, y_train, X_test, y_test = model.train_test_split(data, n_test=500)

	model.train(X_train, y_train)
	
	print('\nHits de Treino:')
	model.score(X_train, y_train)

	print('\nHits de Teste:')
	model.score(X_test, y_test)
	
	

if __name__ == '__main__':
	main()