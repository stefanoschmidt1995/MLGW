###################
#	Some ML routines useful for logistic regression and PCA in a GW dataset
#	(even though procedure are slightly more general than that)
###################

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

################# LOGREG class
class logreg_model:
	"""
	Class aimed to deal with a modified logistic regression model.
	It fits a logistic regression model by gradient descent and it's able to do prediction on test data.
	The basic model is
		y = (a + <h,x>) * sigma(Wx + b)
	with y (K,) and x(D,)
	You can decide to fit the amplitude or not with the parameter fit_amplitude
	"""
	def __init__(self, D, K, fit_amplitude = False):
		"""
		Creates the model and initialize everything.
		Model is:
			y = A * sigma(Wx + b)
		with A = 1 			if fit_amplitude is false
			 A = <h,x> + a	if fit_amplitude is true
		Input:
			D ()			number of dimension for the x space
			K ()			number of dimension for the y space
			fit_amplitude	whether amplitude shall be fitted or not
		Output:
		"""
		self.fit_amplitude = fit_amplitude
		self.W = np.zeros((K,D))
		self.b = np.zeros((K,))
		if self.fit_amplitude:
			self.a = np.array([0])
			self.h = np.zeros((D,))
		self.D = D
		self.K = K
		self.prep_constants = []
		return None

	def __params_unwrap(self, param_vec, shapes, sizes):
		"""Helper routine for minimize_list"""
		args = []
		pos = 0
		for i in range(len(shapes)):
			sz = sizes[i]
			args.append(param_vec[pos:pos+sz].reshape(shapes[i]))
			pos += sz
		return args

	def __params_wrap(self, param_list):
		"""Helper routine for minimize_list"""
		param_list = [np.array(x) for x in param_list]
		shapes = [x.shape for x in param_list]
		sizes = [x.size for x in param_list]
		param_vec = np.zeros(sum(sizes))
		pos = 0
		for param in param_list:
			sz = param.size
			param_vec[pos:pos+sz] = param.ravel()
			pos += sz
		unwrap = lambda pvec: self.__params_unwrap(pvec, shapes, sizes)
		return param_vec, unwrap

	def __minimize_list(self, cost, init_list, args):
		"""Optimize a list of arrays (wrapper of scipy.optimize.minimize)

		The input function "cost" should take a list of parameters,
		followed by any extra arguments:
			cost(init_list, *args)
		should return the cost of the initial condition, and a list in the same
		format as init_list giving gradients of the cost wrt the parameters.
		"""
		opt = {'maxiter': 500, 'disp': False}
		init, unwrap = self.__params_wrap(init_list)
		def wrap_cost(vec, *args):
			E, params_bar = cost(unwrap(vec), *args)
			vec_bar, _ = self.__params_wrap(params_bar)
			return E, vec_bar
		res = minimize(wrap_cost, init, args, 'L-BFGS-B', jac=True, options=opt)
		return unwrap(res.x)

	def preprocess_data(self, y):
		"""
		Preprocess labels y for logistic regression by making every data point in [0,1]. The scaling is done feature-wise to ensure that every feature is O(1).
		Input:
			y (N,K)	labels features
		Output:
			y_prep (N,K)		preprocessed labels features
			prep_constants []	constants used for preprocessing [max_y (1,K), min_y (1,K)]
		"""
			#getting scale factors
		if len(self.prep_constants) == 0: #getting constants if it's the first time they're computed
			self.prep_constants = [np.reshape(np.max(y, axis =0), (1,y.shape[1])), np.reshape(np.min(y, axis =0), (1,y.shape[1]))]
		max_y = np.repeat(self.prep_constants[0], y.shape[0], axis=0) #(N,K)
		min_y = np.repeat(self.prep_constants[1], y.shape[0], axis=0) #(N,K)
	
		y_prep = np.divide(y - min_y, np.abs(max_y-min_y))

		return y_prep, self.prep_constants

	def un_preprocess_data(self, y_prep, prep_constants = None):
		"""
		Preprocess data by making every label point in [0,1]. The scaling is done feature-wise to ensure that every feature is O(1).
		Input:
			y_prep (N,K)		preprocessed labels features
			prep_constants []	constants used for preprocessing [max_y (1,K), min_y (1,K)] (if None the last prep_constants are used)			
		Output:
			y (N,K)	labels features
		"""
		if prep_constants is None:
			prep_constants = self.prep_constants
		if len(prep_constants) != 2: #list is not the right size
			raise TypeError("List prep_constants of preprocessing constants has not the right lenght (len ="+str(len(prep_constants))+" instead of 2)")
			return None

			#getting scale factors
		max_y = np.repeat(prep_constants[0], y_prep.shape[0], axis=0)
		min_y = np.repeat(prep_constants[1], y_prep.shape[0], axis=0)
			#scaling back data to original size
		y = np.multiply(y_prep, np.abs(max_y-min_y)) + min_y

		return y

	def logreg_cost(self, params, X, yy, alpha=0.):
		"""
		Regularized "logistic regression" square error cost function and gradients
		Can be optimized with minimize_list.
		Inputs:
			params		(W, b): W (K,D), b (K,)
						(W, b, h, a): W (K,D), b (K,), h (D,), a scalar
			X(N,D)		design matrix of input features
			yy (N,K)	real-valued labels
			alpha		regularization constant
		Outputs:
			(E, [W_bar, b_bar]/[W_bar, b_bar, h_bar, a_bar])	cost and gradients
		"""
		N = X.shape[0]
			#Unpack parameters from list
		if self.fit_amplitude:
			ww, bb, hh, aa = params
			amp = aa + np.matmul(X, hh) #(N,) 
			amp = np.repeat(np.reshape(amp,(N,1)),self.K,axis = 1) #(N,K)
			reg_cost = alpha*(np.sum(np.square(ww))+np.sum(np.square(bb))+np.sum(np.square(hh)))
					#a is not regularized since amplitude can be any scale
		else:
			ww, bb = params
			reg_cost = alpha*(np.sum(np.square(ww))+np.sum(np.square(bb)))
			amp = np.ones((N,self.K)) #(N,k)

		sigma = expit(np.matmul(X,ww.T)+bb)	#(N,K)
		sigma2 = np.multiply(sigma , (1- sigma)) #(N,K)
		delta = yy - np.multiply(amp,sigma) #(N,K)	

			#computation of error
		E = np.sum(np.square(delta)) + reg_cost

			#computation of gradients
		bb_bar = - 2 * np.sum(np.multiply(np.multiply(delta,sigma2),amp),axis=0) + 2*alpha*bb	#(K,N)*(N,)
		ww_bar = -2 * np.matmul(np.multiply(np.multiply(delta,sigma2),amp).T, X) +2*alpha*ww #(K,N)*(N,D) -> (K,D)
		if self.fit_amplitude is False:		
			return E, (ww_bar, bb_bar)

		hh_bar = - 2 * np.sum(np.matmul(np.multiply(delta,sigma).T, X), axis = 0 ) + 2*alpha*hh #(K,N)*(N,D) -> (K,D) -> (D,)
		aa_bar = - 2 * np.sum(np.matmul(delta.T,sigma)) #(K,N)*(N,K) -> ()
		if self.fit_amplitude is True:		
			return E, (ww_bar, bb_bar, hh_bar, aa_bar)

	def fit_gradopt(self, X, yy, alpha):
		'''
		Fits the logistic regression
			f(theta;w) = 1./(1 + np.exp(-w*x + b))
		by mimizing with steepest gradient descent the square error provided in the helper function logreg_cost.
		It preprocess data (if required) and save the preproccessing constants to a local variable ready to use for predictions.
		Input:
			X(N,D)			design matrix of input features
			yy (N,K)/(N,)	real-valued targets
			alpha			regularization constant
		Output:
		'''
		if yy.ndim == 1:
			K = 1
			yy = np.reshape(yy, (yy.shape[0],1))
		else:
			K = yy.shape[1]
		D = X.shape[1] #numbers of features (i.e. size of w)

		if self.K != K or self.D != D or X.shape[0] != yy.shape[0]:
			raise TypeError("Invalid size for input data!")

			#checking if data are to preprocess
		self.preprocessed = False
		if np.max(yy) > 1 or np.min(yy)<0:
			yy, prep_constants = self.preprocess_data(yy)
			self.preprocessed = True

		args = (X, yy, alpha)
			#random initialization & fit
		if self.fit_amplitude:
			init = (np.random.rand(K,D)/np.sqrt(K*D), np.random.rand(K)/np.sqrt(K), np.random.rand(D)/np.sqrt(D), np.array(np.random.rand())) #W,b,h,a
			self.W, self.b, self.h, self.a = self.__minimize_list(self.logreg_cost, init, args)
		else:
			init = (np.random.rand(K,D)/np.sqrt(K*D), np.random.rand(K)/np.sqrt(K)) #W,b
			self.W, self.b = self.__minimize_list(self.logreg_cost, init, args)
		return None

	def get_predictions(self,X_test):
		'''
		Given a list of weights return the prediction of logistic regression on a test set X for each of the weight in W_list. If require it make the un-preprocessing operation.
		Input:
			X_test (N,D)	test set to evaluate
		Output:
			y_pred (N,K)	predictions for the logistic regression

	Evaluates the sigmoid function for the np array X (N*D) given the list (with K elements) of weights W_list = [(ww,bb)...].
		Returns a N*K matrix'''
		res = expit(np.matmul(X_test,self.W.T)+self.b)
		if self.fit_amplitude:
			amp = self.a + np.matmul(X_test, self.h)
			amp = np.repeat(np.reshape(amp,(amp.shape[0],1)),self.K,axis = 1)
		else:
			amp = 1
		y_pred = np.multiply(amp,res)
		if self.preprocessed:
			y_pred = self.un_preprocess_data(y_pred)
		return y_pred
	
	def get_reconstruction_error(self, X_test, y_test):
		"""
		Return reconstruction errror evaluated for the values X_test using the formula: E = ||y_pred - y_test|| / N
		Input:
			X_test (N,D)	test set 
			y_test (N,K)	test labels
		Output:
			erorr ()	square loss of the model
		"""
		y_pred = self.get_predictions(X_test)
		return np.linalg.norm(y_pred - y_test, ord= 'fro')/(X_test.shape[0])

	def get_weights(self):
		"""
		Returns the weights of the model.
		Input:
		Output:
			weights []	[W, b]: W (K,D), b (K,)
						[W, b, h, a]: W (K,D), b (K,), h (D,), a scalar
		"""
		if self.fit_amplitude:
			return [self.W, self.b, self.h, self.a]
		if not self.fit_amplitude:
			return [self.W, self.b]

	def get_prep_constants(self):
		"""
		Returns the preprocessing constants used for the model
		Input:
		Output:
			prep_constants []	constants used for preprocessing [max_y (1,K), min_y (1,K)]
		"""
		return self.prep_constants

################# PCA class
class PCA_model:
	"""
	Class aimed to deal with a PCA model.
	It fits a PCA model and is able to reduce a dataset to a lower dimensional one and to reconstruct low dimensional data to high dimensional one.
	"""
	def __init__(self):
		"Constructor"
		self.PCA_params = []
		return None

	def save_model(self, filename):
		"""
		Save the PCA model parameters to file in the matrix [V (D,K), mu (D,), scale_factor ()]' with shape (D,K+2)
		Input:
			filename	file to save the model in
		Output:
		"""
			#doesn't work properly... :(
		if self.PCA_params == []:
			print("Model is not fitted yet! There is nothing to save")
			return None
		V = self.PCA_params[0] #(D,K)
		mu = (self.PCA_params[1])[:,np.newaxis]#(D,1)
		max_PC = np.zeros(mu.shape) #(D,1)
		max_PC[:V.shape[1],0] = self.PCA_params[2] #(K,)
		to_save = np.concatenate((V,mu,max_PC), axis = 1) #(D, K+2)
		np.savetxt(filename, to_save)
		return None 

	def load_model(self, filename):
		"""
		Load the PCA parameters from file. The format is the same as save_model
		Input:
			filename	file to load the model from
		Output:
		"""
		data = np.loadtxt(filename)
		V = data[:,:data.shape[1]-2]
		mu = data[:,data.shape[1]-2]
		max_PC = data[:V.shape[1],data.shape[1]-1]
		self.PCA_params= [V,mu,max_PC]
		return None

	def reconstruct_data(self, red_data):
		"""
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
		E, V = np.linalg.eig(np.dot(X.T, X))
		idx = np.argsort(E)[::-1]
		V = V[:, idx[:K]] # (D,K)
		self.PCA_params = [V.real, mu, np.ones((K,))]

		if scale_PC:
			red_data = np.matmul(X, self.PCA_params[0]) #(N,K)
			self.PCA_params[2] = np.max(np.abs(red_data), axis = 0) #(K,)

		return E[:K].real

	def get_V_matrix(self):
		"""
		Returns the projection matrix of the model
		Input:
		Output:
			V (D,K) matrix of eigenvector used for projection and reconstruction of data
		"""
		return self.PCA_params[0]

	def get_PCA_params(self):
		"""
		Returns the parameters of the model
		Input:
		Output:
			PCA_params [V (D,K), mu (D,), max_PC (K,)]	paramters for preprocessing and PCA
		"""
		return self.PCA_params

	def change_grid(self, old_grid, new_grid):
		"""
		Changes the dimensionality D of the visibile data by interpolating each of PC from old grid (dimension D) to new grid (dimension D').
		Input:
			old_grid (D,)	grid of point which old features are evaluated in
			new_grid (D',)	grid of points for the new features
		Output:
		"""
		old_V = self.PCA_params[0]
		new_V = np.zeros((new_grid.shape[0], old_V.shape[1]))
		new_mu = np.interp(new_grid, old_grid,self.PCA_params[1]) #changing mu
		for i in range(old_V.shape[1]):
			new_V[:,i] = np.interp(new_grid, old_grid, old_V[:,i])
		self.PCA_params = [new_V, new_mu, self.PCA_params[2]]
		return None

#A small helper to test logistic regression...
def try_fit_logreg(N_data = 1000):
	theta_vector = np.zeros((N_data,3))
	for i in range(N_data): #loop on data to be created
		theta_vector[i,:] = [np.random.uniform(1,20), np.random.uniform(-0.99,0.99), np.random.uniform(-0.99,0.99)]

	W = np.array([[1,2,3],[-2,-1,4]])
	b = np.array([1,-2])
	a = 3.
	h = np.array([-1,3,2])
	#h = np.array([0,0,0])
	
	amp = a + np.matmul(theta_vector, h)
	amp = 3*np.repeat(np.reshape(amp,(amp.shape[0],1)), b.shape[0],axis = 1)
	y_dataset = np.multiply(amp, expit(np.matmul(theta_vector,W.T)+b))
	train_theta, test_theta, y_train, y_test = make_set_split(theta_vector, y_dataset, .85)

	try_logreg = logreg_model(W.shape[1],W.shape[0], True)
	try_logreg.fit_gradopt(train_theta, y_train, .0)
	y_fit = try_logreg.get_predictions(test_theta)

	error_logreg = np.linalg.norm(y_fit - y_test, ord= 'fro')/y_fit.shape[0]
	print("Error logreg fit ", error_logreg)
	print(try_logreg.get_weights()[0],try_logreg.get_weights()[1],try_logreg.get_weights()[2],try_logreg.get_weights()[3])
	quit()
	return None

################# Extra feature routine
def add_extra_features(data, feature_list):
	"""
	Given a dataset, it enlarge its feature number to make a basis function regression.
	Features to add must be specified with feature list. Each element in the list is a string in form "ijk" where ijk are feature indices as in numpy array data (repetitions allowed); this represents the new feauture x_new = x_i*x_j*x_k
	Input:
		data (N,D)/(N,)			data to augment
		feature_list (len L)	list of features to add
	Output:
		new_data (N,D+L)	data with new feature
	"""
	if data.ndim == 1: data = data[:,np.newaxis]
	if len(feature_list)==0:
		return data

	new_features = np.zeros((data.shape[0],len(feature_list)))
	for i in range(len(feature_list)):
		feat_str = feature_list[i]
		temp = np.ones((data.shape[0],)) #(N,)
		if len(feat_str) ==1: continue #single feature is already in the data
		for k in feat_str:
			try:
				temp = np.multiply(temp, data[:,int(k)]) #(N,)
			except:
				raise RuntimeWarning("Unproper feature code for feature \""+k+"\": unable to add any the feature. Constant feature is placed instead")
				temp = np.ones((data.shape[0],))
				break
		new_features[:,i] = temp #assign new feature

	new_data = np.concatenate((data, new_features), axis = 1)
	return new_data


################# tf loss function
def mismatch_function(logreg_weights, PCA_weights, test_amp):
	"""
	Keras version of mismatch function.
	Input:
		logreg_weights	[max (K,), min (K,)]
		PCA_weights		[V (D,K), mu (D,), beta ()]
		test_amp 		(N,D)
	Output:
		loss(y_true, y_pred)	function of two tf tensors returning a tensor with mismatches
	"""
	import tensorflow as tf
	import keras.backend as K

	def loss(y_true, y_pred):
		if not K.is_tensor(y_pred):
			y_pred = K.constant(y_pred)
		y_true = K.cast(y_true, y_pred.dtype)

			#initializing np tensors to tf tensors
		max_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[0]))
		min_y = K.tf.Variable(K.cast_to_floatx(logreg_weights[1]))
		V_PCA = K.tf.Variable(K.cast_to_floatx(PCA_weights[0].T))
		mu = K.tf.Variable(K.cast_to_floatx(PCA_weights[1]))
		alpha = K.tf.Variable(K.cast_to_floatx(PCA_weights[2]))
		amp = K.tf.Variable(K.cast_to_floatx(test_amp[0,:]))

			#un pre-processing with logreg
		y_true = K.tf.multiply(y_true, K.abs(max_y-min_y)) + min_y #multiply does authomatically the right thing when (N,D)*(D,) !!!!!!
		y_pred = K.tf.multiply(y_pred, K.abs(max_y-min_y)) + min_y
			#inverting PCA
		y_true = K.dot(y_true, V_PCA)
		y_true = (y_true+mu)*alpha
		y_pred = K.dot(y_pred, V_PCA)
		y_pred = (y_pred+mu)*alpha
			#mismatch
		w_pred = K.tf.complex(K.tf.multiply(amp, K.cos(y_pred)), K.tf.multiply(amp, K.sin(y_pred)))
		w_true = K.tf.complex(K.tf.multiply(amp, K.cos(y_true)), K.tf.multiply(amp, K.sin(y_true)))
		overlap = K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_true)))
		norm = K.sqrt(K.tf.multiply(K.tf.real(K.sum(np.multiply(K.tf.conj(w_pred),w_pred))), K.tf.real(K.sum(np.multiply(K.tf.conj(w_true),w_true)))) )
		overlap = 1-K.tf.divide(overlap, norm)
		
		return overlap
		#return K.mean(K.square(y_pred - y_true), axis=-1)
	return loss





