import sys
sys.path.insert(1, '../mlgw_v1')

from GW_helper import *
from ML_routines import *
from DenseMoE import *
from keras.models import Model
from keras.layers import Input, Dense
from EM import *

class cluster_fit(object):
	"""
	This class is to fit a cluster model over data given. After that, for each cluster, a MoE model is fitted for each cluster.
	It has methods for make predictions and test validation error.
	"""

	def __init__(self, K, L, exp_list):
		"""
		Initialize the class with L clusters in a space with K features. The MoE model has a number of experts given in exp_list.
		Input:
			D				dimensionality of space
			K				number of clusters
			exp_list ()/[]	list of L number of experts for each cluster model (if a number it's the same for every cluster)
		Output:
		""" 
		self.K = K
		self.D = D
		if type(exp_list) is not list:
			self.exp_list = [exp_list for i in range(K)]
		else:
			self.exp_list = exp_list
		return

	def fit(self, X, y, N_epochs = 500):
		"""
		It fits the model given a training set X.
		Input:
			X (N,L)		training set
			y (N,D)		labels set
			N_epochs	epochs to be used for fitting MoE
		Output:
		"""
		self.GMM_model = GMM(X, self.K)
		self.GMM_model.fit(tol=1e-3)

			#fitting MoE models
		self.MoE_list = []
		for i in range(self.K):
				#looking for data in considered cluster
			indices = self.GMM_model.get_cluster_indices(y,i)
			inputs = Input(shape=(X.shape[1],))
			hidden = DenseMoE(self.D, self.exp_list[i], expert_activation='linear', gating_activation='softmax')(inputs)
			model = Model(inputs=inputs, outputs=hidden)
			model.compile(optimizer = 'rmsprop', loss = 'mse')
			history = model.fit(x=X[indices,:], y=y[indices,:], batch_size=64, epochs=N_epochs, shuffle=True, verbose=0)
			print("Train model loss for cluster "+str(i), model.evaluate(X[indices,:], y[indices,:]))
			self.MoE_list.append(model.copy()) #can result in troubles...

		return
	










