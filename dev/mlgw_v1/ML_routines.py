"""
Module ML_routines.py
=====================
	Definition of the following ML routines:
		PCA model
			class PCA_model: implements a PCA model with methods for fitting and doing data reduction
		Gaussian Discriminant Analysis
			class GDA: implements a model for a Gaussian discriminant Analysis classifiers. It might be useful for MoE.
		Data augmentation helper
			function add_extra_features: adds to a dataset some extra polynomial features
"""
#################

import scipy.stats
import numpy as np
import warnings

################# PCA class
class PCA_model:
	"""
PCA_model
=========
	Class aimed to deal with a PCA model.
	It fits a PCA model and is able to reduce a dataset (dimension D) to a lower dimensional (dimension K) one and to reconstruct low dimensional data to high dimensional one.
	It stores the following parameters (get them with get_PCA_params()):
		V (D,K)			matrix for dimensional reduction
		mu (D,)			the average value for each feature of dataset
		max_PC (K,)		maximum value of PC projection used to redurn scaled low dimensional data (activate it with scale_PC=True in fit_model methods())
		E (K,)			Eigenvalues of the PCs
	"""
	def __init__(self, filename = None):
		"""
	__init__
	========
		Constructor for PCA model. If filename is given, loads the model from file.
		Input:
			filename	file to load the model from
		"""
		self.PCA_params = []
		if filename is not None:
			self.load_model(filename)
		return None

	def save_model(self, filename):
		"""
	save_model
	==========
		Save the PCA model parameters to file in the matrix
			[[V (D,K), mu (D,)], [max_PC (K(+1),)], [E (K(+1),)] ]
		with shape (D+2,K+1)
		Input:
			filename	file to save the model in
		Output:
		"""
		if self.PCA_params == []:
			print("Model is not fitted yet! There is nothing to save")
			return None
		(D, K) = self.PCA_params[0].shape
		V = self.PCA_params[0] #(D,K)
		mu = (self.PCA_params[1])[:,np.newaxis]#(D,1)
		max_PC = (self.PCA_params[2]) #(K,)
		E = (self.PCA_params[3]) #(K,)
		first_row = np.concatenate((V,mu), axis = 1) #(D, K+1)

		to_save = np.zeros((D+2, K+1))
		to_save[:D,:] = first_row
		to_save[D,:K] = max_PC
		to_save[D+1,:K] = E.real
		to_save[D:,-1] = np.NAN		

		np.savetxt(filename, to_save)
		return None 

	def load_model(self, filename):
		"""
	load_model
	==========
		Load the PCA parameters from file. The format is the same as save_model
		Input:
			filename	file to load the model from
		Output:
		"""
		data = np.loadtxt(filename) #loading data

		if not np.any(np.isnan(data)): #if there is no NaN, the old format is employed. This is to ensure code portability :(
			warnings.warn("Old PCA model type given. The model is loaded correctly but it is better to save the model to the new format.")
			V = data[:,:data.shape[1]-2] #(D,K)
			mu = data[:,data.shape[1]-2] #(D,)
			max_PC = data[:V.shape[1],data.shape[1]-1] #(K,)
			E = np.ones((V.shape[1],)) #(K,)
		else:
			V = data[:data.shape[0]-2,:data.shape[1]-1]
			mu = data[:data.shape[0]-2,data.shape[1]-1]
			max_PC = data[data.shape[0]-2,:data.shape[1]-1]
			E = data[data.shape[0]-1,:data.shape[1]-1]

		self.PCA_params= [V,mu,max_PC, E]
		return None

	def reconstruct_data(self, red_data):
		"""
	reconstruct_data
	================
		Gives the best estimate of high dimensional data given the low dimensional PCA approximation.
		Data are rescaled back to the original training measure inverting the preprocessing procedure.
		Input:
			red_data (N,K)	low dimensional representation of data
		Output:
			data (N,D)		high dimensional reconstruction of data (after inversion of preprocessing)
		"""
		red_data = np.multiply(red_data, self.PCA_params[2])
		data = np.matmul(red_data, self.PCA_params[0].T)
		data = data+self.PCA_params[1]
		return data.real


	def reduce_data(self, data):
		"""
	reduce_data
	===========
		Reduce data by applying a PCA dimensionality reduction. Data are preprocessed by a scaling and a mean shift according to parameters give in PCA_params (in the fashion of fit_PCA). The reduced version of preprocessd data is returned (X_red = X*V_PCA).
		Input:
			data (N,D)		data to reduce
		Output:
			red_data (N,K)	dimensional reduction of preprocessed data
		"""
		data = data - self.PCA_params[1]
		red_data = np.matmul(data, self.PCA_params[0])
		red_data = np.divide(red_data, self.PCA_params[2]) #scaling PC to make them o(1)
		return red_data.real

	def fit_model(self, X, K = None, scale_PC=True):
		"""
	fit_model
	=========
		Fit the PCA model for the given dataset. Data are done zero mean for each feature and rescaled s.t. are O(1) if scale_data is True.
		A parameter set is returned holding fitted PCA parameters (projection matrix, data mean and scale factor)
		Input:
			X (N,D)		training set
			K ()		number of principal components
			scale_PC	whether PC should be scaled by their maximum value to make them all O(1)
		Output:
			E (K,)	eigenvalues of the first K principal components
		"""
		X = X.real
		mu = np.mean(X,0) #(D,)
		X = X - mu

			#doing actual PCA
		if K is None:
			K = X.shape[1]
		#E, V = np.linalg.eig(np.dot(X.T, X))
		E, V = np.linalg.eig(np.cov(X.T))
		idx = np.argsort(E)[::-1]
		V = V[:, idx[:K]] # (D,K)
		E = E[idx[:K]].real #(K,)
		self.PCA_params = [V.real, mu, np.ones((K,)),  E]

		if scale_PC:
			red_data = np.matmul(X, self.PCA_params[0]) #(N,K)
			self.PCA_params[2] = np.max(np.abs(red_data), axis = 0) #(K,)

		return E[:K].real

	def get_V_matrix(self):
		"""
	get_V_matrix
	============
		Returns the projection matrix of the model
		Input:
		Output:
			V (D,K) matrix of eigenvector used for projection and reconstruction of data
		"""
		return self.PCA_params[0]

	def get_dimensions(self):
		"""
	get_dimensions
	==============
		Returns dimension of high- and low-dimensional space.
		Input:
		Output:
			(D,K) (tuple)	dimensions in the format (high-dim, low-dim)
		"""
		return self.PCA_params[0].shape

	def get_PCA_params(self):
		"""
	get_PCA_params
	==============
		Returns the parameters of the model
		Input:
		Output:
			PCA_params [V (D,K), mu (D,), max_PC (K,), E (K,)]	paramters for preprocessing and PCA
		"""
		return self.PCA_params

################# Gaussian Discriminant Analysis
class GDA(object):
	"""
GDA
===
	This class implements a model for Gaussian Discriminant Analysis. The model is a classifier with form:
		p(y=k|x,params) ~ p(y=k) * p(x | y=k, params) =  pi_k * N(x | mu_k, sigma_k)
	"""
	def __init__(self, D, K, naive = True, hard_clustering = False, same_weights = False):
		"""
	__init__
	========
		Initialize the model with K classes for regressions.
		Input:
			D 				dimensionality of input space
			K				number or classes for regression
			naive			whether the covariance matrix for each class should be diagonal (naive assumption)
			hard_clustering	whether hard clustering predictions should be done
			same_weights	whethet all p(y=k) should be the same
		"""
		self.D = D 
		self.K = K 
		self.naive = naive
		self.same_weights = same_weights
		self.hard_clustering = hard_clustering
		if not naive: self.models = []
		self.model_params = []
			#initializing with dummy things
		for k in range(self.K):	
			mu = np.zeros((D,)) #(D,)
			sigma = np.ones((D,))/D
			if not naive:
				self.models.append( scipy.stats.multivariate_normal(mu, np.diag(sigma)) )
				self.model_params.append((mu,np.diag(sigma)))
			else:
				self.model_params.append((mu,sigma))
		self.pi_k = np.ones((K,))/K #initialization
		return

	def init_centroids(self, centroids, sigma = None):
		"""
	init_centroids
	==============
		Initialize each of cluster mean and variance with a custom value..
		Input:
			centroids (D,K)		array with K centroids (each of D dimension)
			sigma (D,K)			diagonal of covariance matrix
		"""
		if centroids.shape != (self.D,self.K):
			raise TypeError("Centroids matrix not suitable! Shape should be ("+str(self.D)+","+str(self.K)+")")
		if sigma is None:
			sigma = np.ones((self.D,self.K))
		if sigma.shape != (self.D,self.K):
			raise TypeError("Sigma matrix not suitable!")
		for k in range(self.K):
			self.model_params[k] = (centroids[:,k],sigma[:,k])
		return 

	def predict(self, X_test, LL = False):
		"""
	predict
	=======
		Makes predictions with probability: p(y = k | x).
		Log probability can be returned
		Input:
			X_test (N,D)	test points
			LL				whether log probability is required
		Ouput:
			y_pred (N,K)	predictions
		"""
		if len(self.model_params) != self.K:
			raise RuntimeWarning("Model must be fitted before making predictions! No predictions have been done")
			return None
		res = np.zeros((X_test.shape[0],self.K))		
		for k in range(self.K):
			if self.naive:
				mu = self.model_params[k][0]	#(D,)
				sigma_sq = self.model_params[k][1] #(D,) diagonal elements
				non_zero = np.where(sigma_sq !=0)[0] #(D',) only non-zero variance components enter the valuation of argument
				arg = -0.5*np.divide(np.square(X_test[:,non_zero]- mu[non_zero]), sigma_sq[non_zero]) #shape (N,D')
				arg = np.sum(arg, axis =1) #(N,)
				#arg = arg - .5 * np.sum(np.log(sigma_sq[non_zero])) #it's better not to normalize things
				res[:,k] = arg + np.log(self.pi_k[k]) #adding p(y=k) as in Bayes' rule
			if not self.naive:
				res[:,k] = self.pi_k[k]*self.models[k].pdf(X_test) #(N,) #really old version of the model

		if self.hard_clustering: #returning hard clustering predictions
			to_return = np.zeros(res.shape)
			to_return[np.arange(len(res)), res.argmax(1)] = 1
			return to_return

		if not self.naive:
			res = np.divide(res.T, np.sum(res, axis =1)).T
			if LL: res = np.log(res)
			return res #(N,K)	
		else:
			if LL:
				return res #result is not normalized... I should find a way to do it...
			res = (res.T - np.mean(res, axis =1)).T
			res = np.exp(res)
			res[np.where(res==0)[0]] = 1e-5
			return np.divide(res.T, np.sum(res, axis =1)).T

	def fit(self, X_train, y_train):
		"""
	fit
	===
		Fit the model with MLE.
		Input:
			X_train (N,D)	train data
			y_train (N,)	train labels
		"""
		self.models = []
		self.model_params = []
		for k in range(self.K):	
			mu = np.sum(np.multiply(X_train.T,y_train[:,k]).T, axis = 0) #(D,)
			mu = np.divide(mu, np.sum(y_train[:,k]))
			#print("mu:", mu.shape)
			temp_var = X_train-mu
			#print("temp_var: ", temp_var.shape)
			sigma_sq = np.matmul(np.multiply((X_train-mu).T, y_train[:,k]), X_train-mu) #(D,D)
			sigma_sq = np.divide(sigma_sq, np.sum(y_train[:,k]))
			if not self.naive:
				self.models.append( scipy.stats.multivariate_normal(mu, sigma_sq) )
			if self.naive:
				#only elements in diagonal are taken (naive assumption)
				sigma_sq = np.diag(sigma_sq) #(D,) #can be used to prevent overfitting
			self.model_params.append((mu,sigma_sq))
		if not self.same_weights:
			self.pi_k = np.sum(y_train, axis = 0)/np.sum(y_train)
		return
	
	def get_weights(self):
		"""
	get_weights
	===========
		Return the weights of the model.
		Output:
			model_params []		parameters of generative gaussians [(mu_0,sigma_0), ... , (mu_K-1,sigma_K-1)]
			pi_k (k,)			probabilities for each class p(y = k) 
		"""
		return self.model_params, self.pi_k

	def accuracy(self, X_test, y_test, LL = False):
		"""
	accuracy
	========
		Computes the accuracy of the model (i.e. the fraction of misclassified points).
		This measure is meaningful only in the case of hard clustering where there is only one label for each data point.
		Input:
			X_test (N,D)	test points
			y_test (N,K)	true labels of test points
			LL				whether to use log prob for predictions
		Output:
			accuracy	accuracy of the predictions made at test points
		"""
		if X_test.ndim == 1:
			X_test = np.reshape(X_test, (X_test.shape[0],1))
		y_pred = self.predict(X_test, LL)
		return np.sum(np.argmax(y_pred,axis=1)==np.argmax(y_test,axis=1))/float(y_test.shape[0])


################# Extra features routine
def add_extra_features(data, feature_list, log_list = None):
	"""
add_extra_features
==================
	Given a dataset, it enlarge its feature number to make a basis function regression.
	Features to add must be specified with feature list. Each element in the list is a string in form "ijk" where ijk are feature indices as in numpy array data (repetitions allowed); this represents the new feauture x_new = x_i*x_j*x_k
	Features can be log preprocessed.
	Input:
		data (N,D)/(N,)			data to augment
		feature_list (len L)	list of features to add
		log_list []				list of indices in data, to which apply a log preprocessing (None is the same as [])
	Output:
		new_data (N,D+L)	data with new feature
	"""
	data = np.array(data) #this is required to manipulate freely the data...
	if data.ndim == 1: data = data[:,np.newaxis]
	if len(feature_list)==0:
		return data

	if log_list is not None:
		data[:,log_list] = np.log(data[:,log_list]) #probably a good idea...

	D = data.shape[1]
	new_features = np.zeros((data.shape[0],len(feature_list)))
	for i in range(len(feature_list)):
		temp = np.ones((data.shape[0],)) #(N,)
		exps = np.zeros((D,)) #(D,)
		for j in range(D):
			exps[j] = feature_list[i].count(str(j))
		temp = np.power(data,exps) #(N,D)
		temp = np.prod(temp, axis =1) #(N,)
		new_features[:,i] = temp #assign new feature

	new_data = np.concatenate((data, new_features), axis = 1)
	return new_data

def jac_extra_features(data, feature_list, log_list = None):
	"""	
jac_extra_features
==================
	Given a dataset, it computes the jacobian of the augmented data. Data are augmented as in add_extra_features.
	Features to add must be specified with feature list. Each element in the list is a string in form "ijk" where ijk are feature indices as in numpy array data (repetitions allowed); this represents the new feauture x_new = x_i*x_j*x_k
	Features can be log preprocessed.
	Gradients are computed as follows:
		grad_ijk = D_k (xi_i)_j
	where (xi_i)_j is the j-th augmented feature for the i-th data point
	Input:
		data (N,D)/(N,)			data to augment
		feature_list (len L)	list of features to add
		log_list []				list of indices in data, to which apply a log preprocessing (None is the same as [])
	Output:
		grad (N,D+L,D)	gradient of the new features
	"""
	data = np.array(data) #this is required to manipulate freely the data...
	if data.ndim == 1: data = data[:,np.newaxis]
	if len(feature_list)==0:
		return data
	D = data.shape[1]

	if log_list is not None:
		data[:,log_list] = np.log(data[:,log_list]) #probably a good idea...

	jac = np.zeros((data.shape[0], len(feature_list)+D,D)) #(N,D+L,D)
	jac[:,:D,:] = np.identity(D) #setting easy gradients
	for i in range(len(feature_list)):
		exps = np.zeros((D,)) #(D,)
		for j in range(D):
			exps[j] = feature_list[i].count(str(j))
		for j in range(D):
			if exps[j] !=0:
				temp_exps = np.array(exps)
				temp_exps[j] = temp_exps[j] -1
				#print(exps[j],temp_exps)
				jac[:,i+D,j] = exps[j]*np.prod(np.power(data, temp_exps), axis =1)

	if log_list is not None:
		jac[:,:,log_list] = np.divide(jac[:,:,log_list],np.exp(data[:,None,log_list]))

	return jac























