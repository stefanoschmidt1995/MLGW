############
#	class for EM algorithm for fitting a MoE model
#	The expert model and the gating functions can be freely specified by the user
############
import scipy.stats
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

class MoE_model(object):
	"""
	It represents and fits a MoE model of the form:
		p(y_i | x_i, params) = sum_{k=1}^K S(x_i)_k * N(y_i|<w_k,x_i>+b_k, sigma_k)
	where S(x_i) is any model. A default for S(x_i) is the softmax model S(x_i)= softmax(V*x_i).
	The model is fitted trough EM algorithm.
	"""
	def __init__(self, D, K, gating_function = None, bias = True):
		"""
		Initialize the model.
		The gating function must be specified through an object with methods:
			fit(x_train(N,D)		labels_train (N,K)) returning
			predict((x_test(N,D)))	returning labels_predicted (N,K)
			save					for saving to file the entire model
			load					to load from file the entire model
		Gating function must use cross entropy loss function.
		If None a default softmax regression model is used built from class softmax_regression.
		The gating function obkect must be already initialized properly.
		Input:
			D						dimensionality of input space
			K						number of experts for the model
			gating_function (obj)	an object which represents the gating function
			bias					whether to use a bias in the expert model
		"""
		self.D = D 
		self.K = K 
		if gating_function is None:
			self.gating = softmax_regression(D,K)
		else:
			self.gating = gating_function
		self.W = np.zeros((self.D, self.K)) #[w_1, ..., w_K]
		self.sigma = np.ones((self.K,))
		self.bias = bias
		self.b = np.zeros((self.K,))
		self.initialized = False

	def get_iperparams(self):
		"""
		Returns values of D and K.
		Outpu:
			(D,K)	(dimensionality of input space, number of experts)
		"""
		return (self.D, self.K)

	def save(self, exp_file, gat_file):
		"""
		Saves the model to file.
		Input:
			exp_file	file to save the expert model to
			gat_file	file to save the model for gating function
		"""
		to_save = np.stack((self.b, self.sigma)) #(2, K)
		to_save = np.concatenate((self.W,to_save) , axis = 0)
		np.savetxt(exp_file, to_save)
		self.gating.save(gat_file)
		return
	
	def load(self, exp_file, gat_file, load_function = None):
		"""
		Load the model from file. It changes parameters D and K if required.
		Input:
			exp_file		file to load the expert model from
			gat_file		file to load the model for gating function from (using function load_function)
			gating_function	function for loading the gating function model from file
		"""
		weights = np.loadtxt(exp_file)
		self.W = weights[:weights.shape[0]-2,:]
		self.b = weights[weights.shape[0]-2,:]
		self.sigma = weights[weights.shape[0]-1,:]
		if np.all(self.b==0):
			self.bias = False
		else:
			self.bias = True
		self.D = self.W.shape[0]
		self.K = self.W.shape[1]
		
		if load_function is None:
			temp_load = softmax_regression(1,1)
			load_function = temp_load.load

		del self.gating
		self.gating = load_function(gat_file)
		self.initialized =  True
		return

	def experts_predictions(self, X):
		"""
		Returns the predictions of the experts.
		Input:
			X_test (N,D)	test points
		Output:
			y_test (N,K)	experts predictions
		"""
		if X.ndim ==1:
			X = np.reshape(X, (1,X.shape[0]))
		return np.matmul(X,self.W) + self.b

	def get_gating_probs(self,X):
		"""
		Returns the probability p(z=k|x, params)
		Input:
			X (N,D)	data points
		Output:
			p_gating (N,K)	probabilities of each gating function
		"""
		pi = self.gating.predict(X) #p(z_i = k|x_i) (N,K)
		#pi = np.divide(pi.T, np.sum(pi, axis = 1)).T
		return pi

	def predict(self, X):
		"""
		Return the predictions of the model
		Input:
			X (N,D)	test points
		Output:
			y (N,)	model value at test points
		"""
		if X.ndim ==1:
			X = np.reshape(X, (1,X.shape[0]))
		
		pi = self.gating.predict(X) #p(z_i = k|x_i) (N,K)

		#indices = np.argmax(pi, axis =1)
		#for i in range(pi.shape[0]):
		#	pi[i,indices[i]] = 1.

		pi = np.divide(pi.T, np.sum(pi, axis = 1)).T
		res = np.multiply(pi, self.experts_predictions(X))
		return np.sum(res, axis = 1)

	def expert_likelihood(self, X, y): #give to it a proper name!!!
		"""
		Computes the quantity p(y_i|x_i, z_i=k) = N(y_i| <w_k,x_i>). This corresponds to the likelihood of each expert for having generated the data.
		Input:
			X (N,D)	data
			y (N,)	targets
		Output:
			pi_k (N,K)	N(y_i| <w_k,x_i>)
		"""
		np.set_printoptions(threshold=sys.maxsize)

		gaussians_mean = self.experts_predictions(X) #(N,K) X*W + b
		y = np.repeat( np.reshape(y, (len(y),1)), self.K, axis = 1) #(N,K)

		#print('sigma: ', self.sigma)
		res = scipy.stats.norm.pdf( np.divide((y - gaussians_mean), self.sigma) ) #(N,K)
		return np.divide(res, self.sigma) #normalizing result

	def log_likelihood(self, X, y):
		"""
		Computes the log_likelihood for the data given with formula: (Are you sure of the formula??)
			LL 	= sum_{i=1}^N log sum_{k=1}^K p(y_i|x_i, z_i=k) * p(z_i=k |x_i) =
				= sum_{i=1}^N log sum_{k=1}^K N(y_i | <w_k,x_i>) S(x_i)_k
		Input:
			X (N,D)	data
			y (N,)	targets for regression
		Output:
			LL	log-likelihood for the model
		"""
		if X.ndim == 1:
			X = np.reshape(X, (X.shape[0],1))
		exp_likelihood = self.expert_likelihood(X, y)
		res = np.multiply(self.gating.predict(X), exp_likelihood) #(N,K)
		res[np.where(np.abs(res)<1e-30)] = 1e-30 #small regularizer for LL
		res = np.log(np.sum(res,axis=1)) #(N,)
		return np.sum(res) / X.shape[0]

	def __initialise_smart__(self, X, args):
		"""
		Having seen the data makes a smart first guess (with farhtest point clustering) for responsibilities and fit gating function model with those responbility.
		Input:
			X (N,D)		train data
			args		arguments to be given to fit method of gating function
		"""
		centroids = np.zeros((self.K,self.D))
		if X.shape[0] > 10*self.K:
			data = X[:10*self.K,:]
		else:
			data = X_train
		N = data.shape[0]

			#choosing centroids
			#points are chosen from dataset with farhtest point clustering
		ran_index = np.random.choice(N)
		centroids[0,:] = data[ran_index]

		for k in range(1,self.K):
			distances = np.zeros((N,k)) #(N,K)
			for k_prime in range(k):
				distances[:,k_prime] = np.sum(np.square(data - centroids[k_prime,:]), axis =1) #(N,K')
			distances = np.min(distances, axis = 1) #(N,)
			distances /= np.sum(distances) #normalizing distances to make it a prob vector
			next_cl_arg = np.random.choice(range(data.shape[0]), p = distances) #chosen argument for the next cluster center
			centroids[k,:] = data[next_cl_arg,:]

		var = np.var(X, axis = 0) #(D,)

			#computing initial responsibilities
		r_0 = np.zeros((X.shape[0],self.K))
		for k in range(self.K):
			r_0[:,k] = np.divide(np.square(X - centroids[k,:]), var)[:,0]
		r_0 = np.divide(r_0.T, np.sum(r_0,axis=1)).T

		self.gating.fit(X,r_0, *args)

		return r_0



	def __initialise__(self, X, args):
		"""
		Having seen the data makes a first guess for responsibilities and fit gating function model with those responbility.
		Input:
			X (N,D)		train data
			args		arguments to be given to fit method of gating function
		"""
			#getting centroids
		indices = np.random.choice(range(X.shape[0]), size = self.K, replace = False)
		centroids = X[indices,:] #(K,D) #K centroids are chosen
			#getting variances
		var = np.var(X, axis = 0) #(D,)

			#computing initial responsibilities
		r_0 = np.zeros((X.shape[0],self.K))
		for k in range(self.K):
			r_0[:,k] = np.divide(np.square(X - centroids[k,:]), var)[:,0]
		r_0 = np.divide(r_0.T, np.sum(r_0,axis=1)).T

		self.gating.fit(X,r_0, *args)

		return r_0

	def fit(self, X_train, y_train, N_iter=None, threshold = 1e-2, args= [], verbose = False, val_set = None, pick_best = True):
		"""
		Fit the model using EM algorithm.
		Input:
			X_train (N,D)	train data
			y_train (N,)	train targets for regression
			N_iter			Maximum number of iteration (if None only threshold is applied)
			threshold		Minimum change in LL below which algorithm is terminated
			args			arguments to be given to fit method of gating function
			verbose 		whether to print values during fit
			val_set			tuple (X_val, y_val) with a validation set to test performances
			pick_best		if True the model with best validation mse is chosen as best model (doesn't apply if val_set is None)
		Output:
			history		list of value for the LL of the model at every epoch
		"""
		if X_train.ndim == 1:
			X_train = np.reshape(X_train, (X_train.shape[0],1))
		if X_train.shape[1] != self.D:
			raise TypeError("Wrong shape for X_train matrix "+str(X_train.shape)+". Second dimension should have lenght "+str(self.D))

		if threshold is None:
			threshold = 0

			#initialization
		if not self.initialized:
			r_0 = self.__initialise_smart__(X_train,args)
			self.EM_step(X_train,y_train, r_0, args)

		i = 0
		old_LL = (-1e5,)
		LL = (-1e4,)
		if pick_best: best_mse = 1e6 #best mse so far (only if val_set is not None)
		history=[]
		while(LL[0] - old_LL[0] > threshold ): #exit condition is decided by train LL (even if val_set is not None): don't know why...
				#do batch update!!!
			old_LL = LL
			gat_history = self.EM_step(X_train,y_train, args = args)
			LL=(self.log_likelihood(X_train, y_train),)
			if isinstance(val_set,tuple) :
				LL = (LL[0], self.log_likelihood(val_set[0], val_set[1]))
			history.append(LL)
			i += 1
			if verbose:
				print("LL at iter "+str(i)+"= ",LL)
				try:
					print("   Gating loss: ", gat_history[0], gat_history[-1])
				except TypeError:
					pass
				if isinstance(val_set,tuple) :
					mse = np.sum(np.square( self.predict(val_set[0])-val_set[1]))/val_set[0].shape[0]
					print("   Val loss: ", mse)
					if mse < best_mse and pick_best:
						best_mse = mse
						print("Chosen the best!")
						try: #saving best model so far
							self.save("temp_exp", "temp_gat") 
						except:
							os.system("rm -f temp_exp temp_gat")
			if N_iter is not None:
				if i>= N_iter:
					break
			try:
				assert LL[0] - old_LL[0] >=0 #train LL must always increase in EM algorithm. Useful check
			except:
				break #if LL increase (and it shouldn't) EM should terminate

		self.initialized =  True

		if isinstance(val_set,tuple) and pick_best: #loading best model so far
			files = os.listdir(".")
			if "temp_exp" in files and  "temp_gat" in files:
				print("loaded the best")
				self.load = ("temp_exp", "temp_gat")
				os.system("rm -f temp_exp temp_gat")

		return history

	def EM_step(self, X, y, r = None, args = []):
		"""
		Does one EM update.
		Input:
			X (N,D)	train data
			y (N,)	train targets for regression
			r (N,K)	responsibilities for the E step (if None they are computed with method get_responsibilities)
			args	some arguments to pass to fit method of gating function
		Output:
			gat_history		history for the gating function fit
		"""
			#E step
		if r is None:
			r= self.get_responsibilities(X,y) #(N,K)
	
		#print("r: ",r)#,"\npi: ", self.expert_likelihood(X, y))
		#plt.plot(X[:,0],r, 'o', ms = 1, label = "true")
		#plt.plot(X[:,0], self.get_gating_probs(X), 'o', ms = 1, label = "pred")
		#plt.legend()
		#plt.show()

			#M step
			#M step for experts
				#weights is updated by solving a linear fit with weights r_{ik} in loss function
		if self.bias:
			X_temp = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1)
		else:
			X_temp = X
		for k in range(self.K):
			R = np.diag(r[:,k]) #(N,N)
			temp = np.linalg.inv(np.matmul(X_temp.T,np.matmul(R,X_temp))) #(D,D)/(D+1,D+1)
			temp = np.matmul(np.matmul(temp, X_temp.T),R) #(D,N)/(D+1,N)
			temp = np.matmul(temp, y) #(D,)/(D+1,)
			if self.bias:
				self.b[k] = temp[0] #()
				self.W[:,k] = temp[1:] #(D,)
			else:
				self.W[:,k] = temp #(D,)
			sigma_square = np.sum(np.multiply(r[:,k], np.square(y-self.experts_predictions(X)[:,k])) ) / np.sum(r[:,k])
			self.sigma[k] = np.sqrt(sigma_square)

			#M step for gating functions
		gat_history = self.gating.fit(X,r, *args)
		return gat_history

	def get_responsibilities(self,X,y):
		"""
		Computes responsibilities for the given input data:
			r_k = p(y=k|x)
		Input:
			X (N,D)	data
			y (N,)	data labels
		"""
		pi = self.gating.predict(X) #p(z_i = k|x_i) (N,K)
		pi = np.divide(pi.T, np.sum(pi, axis = 1)).T
		exp_term = self.expert_likelihood(X, y)

		r =  np.multiply(pi,  exp_term) # (N,K) responsibilities matrix
		#r = np.square(r) #debug
		r = np.divide(r.T, np.sum(r, axis =1)).T
		assert np.all(r>=0) 								#checking that all r_{ik} are more than zero
		assert np.all(np.abs(np.sum(r,axis=1) - 1)<1e-5) 	#checking proper normalization

		#r_new = np.zeros(r.shape)
		#print("r",r)
		#indices = np.argmax(r, axis =1)
		#print(indices)
		#for i in range(r.shape[0]):
		#	r_new[i,indices[i]] = 1.
		#print("r_new", r_new)
		return r

class softmax_regression(object):
	"""
	Implements a class for softmax regression with K labels. It has the form:
		p(y= k|x,V) = exp( V_k*x ) / sum_{k=1}^K exp( V_k*x + b_k )
	It has methods for getting predictions from the model and to fit the model.
	"""
	def __init__(self, D, K):
		"""
		Initialize the model with K classes for regressions.
		Input:
			D 	dimensionality of input space
			K	number or classes for regression
		"""
		self.D = D 
		self.K = K 
		self.V = np.zeros((D+1,K))
		return

	def save(self, filename):
		"""
		Save the model to file.
		Input:
			filename	name of the file to save the model to
		"""
		np.savetxt(filename, self.V)
		return

	def load(self, filename):
		"""
		Load the model from file.
		Input:
			filename	name of the file to load the model from
		"""
		self.V = np.loadtxt(filename)
		self.D = self.V.shape[0]-1
		self.K = self.V.shape[1]
		return self

	def predict(self, X_test, V = None):
		"""
		Makes predictions for the softmax regression model. Weights can be freely specified by the user.
		Input:
			X_test (N,D)	test points
			V (D,K)			weight of the model that gives predictions (if None, internal weights are used)
		Output:
			y_test (N,K)	model prediction
		"""
		if X_test.ndim == 1:
			X_test = np.reshape(X_test, (X_test.shape[0],1))
		if X_test.shape[1] == self.D:
			X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis = 1) #adding dummy variable for the bias
		if V is None:
			V = self.V
		res = np.matmul(X_test,V) #(N,K)
		res = np.exp(res) #(N,K)
		return np.divide(res.T, np.sum(res, axis = 1)).T

	def fit_single_loop(self, X_train, y_train):
		"""		
		Fit the model using the closed form of LL of the problem.
		See: https://link.springer.com/content/pdf/10.1007%2F978-3-642-01510-6_109.pdf 
		Input:
			X_train (N,D)	train data
			y_train (N,)	train targets for regression
		"""
		if X_train.shape[1] == self.D:
			X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis = 1) #adding dummy variable for the bias

		for k in range(self.K-1):
			#print(np.where(y_train[:,-1]+2e-5 ==0))
			div = np.divide(y_train[:,k],y_train[:,-1]+2e-10)
			div[np.where(div ==0)] = 1e-20
			H = np.log(div) # (N,) "new targets for linear regression"
			fitted_V = np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train)), X_train.T) #(D+1,N)
			fitted_V = np.matmul(fitted_V, H)
			self.V[:,k] = fitted_V

		self.V[:,-1] = np.zeros((self.D+1,)) #last is zero by default! Remember it!

		non_zero = np.where(self.predict(X_train)<0)
		print("<0: ",non_zero)
		return 

	def accuracy(self, X_test, y_test):
		"""
		Computes the accuracy of the model (i.e. the fraction of misclassified points).
		This measure is meaningful only in the case of hard clustering where there is only one label for each data point.
		Input:
			X_test (N,D)	test points
			y_test (N,K)	true labels of test points
		Output:
			accuracy	accuracy of the predictions made at test points
		"""
		if X_test.ndim == 1:
			X_test = np.reshape(X_test, (X_test.shape[0],1))
		y_pred = self.predict(X_test)
		return np.sum(np.argmax(y_pred,axis=1)==np.argmax(y_test,axis=1))/float(y_test.shape[0])


	def LL(self, X_test, y_test):
		"""
		Evaluate the log-likelihood for the model given X and their labels.
		Input:
			X_test (N,D)	test points
			y_test (N,K)	labels of test points
		Output:
			LL	log-likelihood for the model
		"""
		if X_test.ndim == 1:
			X_test = np.reshape(X_test, (X_test.shape[0],1))
		return self.loss(self.V, [X_test, y_test, 0.]) * X_test.shape[0]
		

	def loss(self, V, data):
		"""
		Loss function to minimize wrt. V. It is the function:
			NLL(V) = - log[p(D|V)]
		Input:
			V ((D+1)*K,)	weights of logreg
			data 			list [X_train (N,D), y_train (N,K), lambda ()]
		Output:
			loss	value for the loss function evaluated at V	
		"""
		V = np.reshape(V, (self.D+1,self.K))
		X = data[0]
		y = data[1]
		reg_constant = data[2]

		mu = self.predict(X, V) #(N,K)
			#mu must be regularized in logarithm. Otherwise it might give Nan if a label prob is 0
		LL = -(np.sum(np.multiply(y, np.log(mu+1e-40))) / X.shape[0]) + reg_constant * np.sum(np.square(V))
		return LL

	def grad(self, V, data):
		"""
		Gradient of the loss function to minimize wrt. V (see function loss())
		Input:
			V ((D+1)*K,)	weights of logreg
			data 			list [X_train (N,D), y_train (N,K), lambda ()]
		Output:
			grad	value for the gradient of loss function evaluated at V	
		"""
		to_reshape = False
		if V.ndim == 1:
			V = np.reshape(V, (self.D+1,self.K))
			to_reshape = True
		X = data[0]
		y = data[1]
		reg_constant = data[2] #regularizer
		
		mu = self.predict(X, V) #(N,K)
		delta = (mu - y) + 1e-40
		grad = np.matmul(X.T,delta) / X.shape[0] + reg_constant*V #(N,D).T (N,K) = (D,K)
		if to_reshape:
			return np.reshape(grad, ((self.D+1)*self.K,))
		else:
			return grad
	
	def get_weights(self):
		"""
		Returns the weights of the model.
		Input:
		Output:
			V (D+1,K)	weights for the model
		"""
		return self.V

	def fit(self, X_train, y_train, opt = "adam", val_set = None, reg_constant = 1e-4, verbose = False, threshold = 1e-2, N_iter = 30, learning_rate = 1e-3):
		"""
		Fit the model using gradient descent.
		Can use adam for adaptive step: https://arxiv.org/abs/1412.6980v8
		Can use bfgs method provided by scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
		Using a small regularizer can help convergence (especially for bfgs).
		Input:
			X_train (N,D)	train data
			y_train (N,)	train targets for regression
			opt				which optimizer to use ("adam" or "bsfg")
			val_set			tuple (X_val, y_val) with a validation set to test performances
			reg_constant	regularization constants
			verbose			whether to print values of loss function at every train step
			threshold		minimun improvement of validation erorr on 10 iteration before stopping (train error if val_set =None)
			N_iter			number of iteration to be performed (doesn't apply to bfgs or if threshold is not None)
			learning_rate	learning rate used for gradient update (doesn't apply to bfgs)
		Output:
			history		list of value for the loss function
		"""
		if X_train.ndim == 1:
			X_train = np.reshape(X_train, (X_train.shape[0],1))
		if X_train.shape[1] == self.D:
			X_train = np.concatenate((np.ones((X_train.shape[0],1)), X_train), axis = 1) #not necessary but might be useful to speed up the code

		args = (X_train, y_train, reg_constant) #arguments for loss and gradients
		
		if opt == "adam":
			return self.__optimize_adam__(args, threshold, N_iter, learning_rate, verbose, val_set)
		if opt == "bfgs":
			return self.__optimise_bfgs__(args, verbose, val_set)

	def __optimise_bfgs__(self, args, verbose, val_set = None):
		"""
		Wrapper to scipy.optimize.fmin_bfgs (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html) to minimise loss function.
		Input:
			args		tuple of arguments to be passed to loss and gradient [X_train (N,D), y_train (N,K), lambda ()]
			verbose		whether to print scipy convergence message
			val_set		tuple (X_val, y_val) with a validation set to test performances
		Output:
			history		initial and final value of loss function
		"""
		loss_0 = (self.loss(self.V, args),)
		if isinstance(val_set,tuple):
			args_val = (val_set[0], val_set[1], args[2])
			loss_val = self.loss(self.V, args_val)
			loss_0 = (loss_0, loss_val)

			#wrapper to self.loss and self.grad to make them suitable for scipy
		loss = lambda V, a,b,c: self.loss(V,(a,b,c))
		grad = lambda V, a,b,c: self.grad(V,(a,b,c))

		res = scipy.optimize.fmin_bfgs(loss, self.V.reshape(((self.D+1)*self.K,)), grad, args , disp = verbose)
		self.V = res.reshape((self.D+1,self.K))

		if isinstance(val_set,tuple):
			loss_fin = (self.loss(self.V, args), self.loss(self.V, args_val))
		else:
			loss_fin = (self.loss(self.V, args), )
		return [loss_0, loss_fin]


	def __optimize_adam__(self, args, threshold, N_iter, learning_rate, verbose, val_set = None):
		"""
		Implements optimizer with to perform adaptive step gradient descent.
		The implementation follows: https://arxiv.org/abs/1412.6980v8
		Input:
			args			tuple of arguments to be passed to loss and gradient [X_train (N,D), y_train (N,K), lambda ()]
			threshold		minimun improvement of train erorr before stopping fitting procedure
			N_iter			number of iteration to be performed
			learning_rate	learning rate to be set for adam
			verbose			whether to print loss at each step
			val_set			tuple (X_val, y_val) with a validation set to test performances
		Output:
			history		list of loss function value (train, )/(train,val) at each iteration step
		"""
			#setting parameters for learning rate
		beta1 = .9		#forgetting factor for first moment
		beta2 = .999	#forgetting factor for second moment
		epsilon = 1e-8
		m = np.zeros(self.V.shape) #first moment
		v = np.zeros(self.V.shape) #second moment
		history = []
		if threshold is not None:
			N_iter = 1000000000 # if threshold, no maximum iteration should be used
		for i in range(0,N_iter):
			g = self.grad(self.V, args)
			m = beta1*m + (1-beta1)*g
			v = beta2*v + (1-beta2)*np.square(g)
			m_corr = m / (1-beta1)
			v_corr = v / (1-beta2)

			self.V = self.V - learning_rate * np.divide(m_corr, np.sqrt(v_corr)+epsilon)
			self.V[:,-1] = np.zeros((self.D+1,))

			if isinstance(val_set,tuple):
				args_val = (val_set[0], val_set[1], args[2])
				history.append((self.loss(self.V, args), self.loss(self.V, args_val)) ) #(train_err, val_err)
			else:
				history.append((self.loss(self.V, args),))

			if verbose:
				print("Loss at iter= ",i, history[i])

			if threshold is not None and i>10:
				if history[-10][-1] - history[-1][-1] < threshold:
					break
		return history

class GDA(object):
	"""
	This class implements a model for Gaussian Discriminant Analysis. The model is a classifier with form:
		p(y=k|x,params) ~ p(y=k) * p(x | y=k, params) =  pi_k * N(x | mu_k, sigma_k)
	"""
	def __init__(self, D, K, naive = True, hard_clustering = False, same_weights = False):
		"""
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
		Return the weights of the model.
		Output:
			model_params []		parameters of generative gaussians [(mu_0,sigma_0), ... , (mu_K-1,sigma_K-1)]
			pi_k (k,)			probabilities for each class p(y = k) 
		"""
		return self.model_params, self.pi_k

	def accuracy(self, X_test, y_test, LL = False):
		"""
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






