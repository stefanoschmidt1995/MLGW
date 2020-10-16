#Class for dealing with a list of MoE models for dealing with many PCA components fit

import os
from EM_MoE import *		#MoE model

class MoE_list(object):
	"""
	This class contains a list of MoE models and deals with them easily. This might be useful for multidimensional regression where each target dimension is fitted separately by a MoE model.
	All models must have same input space dimensionality; they shall have same gating function model but might have different experts number.
	It takes care also of boring part of saving model to file.
	"""
	def __init__(self, K, folder = None, D = 1, N_exp_list = 10):
		"""
		Initialise class. If folder is given, model is loaded from folder. Otherwise a softmax gating function model is built with the required dimensionality and number of experts.
		Input:
			K				number of MoE models in MoE_list
			folder			folder at which the model should be loaded from. (If None, nothing is loaded)
			The following not required if folder is None:
				D				dimensionality of input space
				N_exp_list		list holding the number of expert to use for each model (if int all N_exp are the same) 
		"""
		self.K = K
		if folder is not None:
			self.MoE_list = [None for k in range(K)]
			self.load(folder)
		else:
				#default model (unfitted)
			if type(N_exp_list) is int:
				N_exp_list = [N_exp_list for k in range(self.K)]
			if len(N_exp_list) != self.K:
				raise TypeError("Lenght of number of expters list doesn't match number of models!")
			self.D = D
			for k in range(self.K):
				self.MoE_list.append(MoE_model(self.D, N_exp_list[k]))
		return

	def load(self, folder, load_function = None):
		"""
		Load models from file. Folder must contain files of the following type:
			exp_#
			gat_#
		where # is in {0,..,K-1}
		Each model is the built from files given.
		Input:
			folder			path to folder in which model is stored
			load_function	function that shall be used to load the gating model. (if None default softmax is used)
		"""
		file_list = os.listdir(folder)
		D_list = []
		for k in range(self.K):
			if "gat_"+str(k) not in file_list or "exp_"+str(k) not in file_list:
				raise RuntimeError("Couldn't find proper files for MoE model "+str(k))
			
			self.MoE_list[k] = MoE_model(self.D,1)
			self.MoE_list[k].load(folder+"exp_"+str(k), folder+"gat_"+str(k), load_function = load_function)
			(D,N_exp) = MoE_list[k].get_iperparams()
			D_list.append(D)
			self.N_exp_list[k] = N_exp
			
		assert len(set(D_list)) == 1 #every value of input space dimensionality must be the same
		self.D = D_list[0]

		return

	def save(self, folder):
		"""
		Saves all the models to the same folder. Files are:
			exp_#
			gat_#
		where # is in {0,..,K-1}.
		Input:
			folder	name of the folder to save files to
		"""
		for k in range(self.K):
			self.MoE_list[k].save(folder+"exp_"+str(k), folder+"gat_"+str(k))
		return

	def fit(self, X_train, y_train, args_list = None):
		"""
		Fit for the k-th model the regression X -> y[k] k = 0,1,.. K.
		Each fit can have its own arguments [N_iter=None, threshold = 1e-2, args= [], verbose = False]
		Input:
			X_train (N,D)	training data
			y_train (N,K)	training targets
			args_list []	list of tuples of arguments to be given to each call of function MoE_model.fit
							Each tuple must be of form (N_iter, threshold, args, verbose)
							If only one tuple is given, it is used for each fitting procedure
		"""
		if type(arg_list) is tuple:
			arg_list = [arg_list for k in range(self.K)]

			#some useful checks
		assert y_train.shape[1] == self.K
		assert X_train.shape[0] == y_train.shape[0]
		assert X.shape[1] == self.D

		for k in range(self.K):
			print("Fitting component ",k)
			#useless variables for sake of clariness
			y_train = PCA_train_ph[:,k]
			MoE_models[k].fit(train_theta, y_train, *arg_list[k])

		return

	def models(self, k_list=None):
		"""
		Returns the MoE model(s).
		Input:
			k_list []	index(indices) of the model to be returned (if None all models are returned)
		Output:
			models []	list of models to be returned
		"""
		if k_list is None:
			k_list = range(self.K)
		return self.MoE_list[k]

	def predict(self, X):
		"""
		Makes predictions for all the models.
		Input:
			X (N,D)	points to make predictions at
		Ouput:
			y (N,K)	prediction for each MoE model
		"""
		assert X.shape[1] == self.D

		y = np.zeros((X_test.shape[0],self.K))
		for k in range(self.K):
			y[:,k] = self.MoE_list[k].predict(X)

		return y







		
