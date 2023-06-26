############
#	class for the EM algorithm for fitting a K means clustering algorithm
############
import scipy.stats
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

class K_means_model(object):
	"""
	This class represents a "modified" K means algorithm, i.e. a model in the form:
		p(x | params) = 1/K sum_{k=1}^K p(x|z=k) = 1/K sum_{k=1}^K N(x| mu_k; diag(sigma_k))
	The model is fitted with EM algorithm.
	The model can make clustering with hard-clustering procedure:
		z = argmax_k p(x|z=k)
	or with soft-clustering:
		z_k ~ p(x|z=k)
	A cluster is eliminated if it is not able to be responsible for any data.
	"""
	def __init__(self, D, K, centroids = None, sigma = None, fit_sigma = None):
		"""
		Initialize the model.
		Input:
			D				dimensionality of space
			K				number of "experts" for the model
			centroids (D,K)	initial guess for centroids mu_k (if None mu are spread randomly in the domain)
			sigma (D,)		initial guess of covariance matrix (if None sigma = np.ones)
			fit_sigma []	list of indices in sigma matrix to be fitted (if None, nothing is fitted)
		"""
		self.D = D 
		self.K = K
		self.init = not (centroids is None) #saving whether data must be initialized later of not...
		if centroids is None:
			self.mu = np.zeros((self.D, self.K)) #[mu_1, ..., mu_K]
		else:
			self.mu = centroids
			assert self.mu.shape == (self.D, self.K)
		if sigma is None:
			self.sigma = np.ones((self.D, self.K))
		else:
			sigma = np.repeat(np.reshape(sigma, (self.D,1)),self.K, axis = 1)
			self.sigma = sigma
			assert self.sigma.shape == (self.D, self.K)
		self.fit_sigma = fit_sigma
		return

	def predict(self, X, hard_clustering = True, log_prob = False, dim_list = None):
		"""
		Makes predictions for labels of each data.
		Input:
			X (N,D')			data to make predictions for (D'<=D)
			hard_clustering		whether to do hard clustering
			log_prob 			whether to return log probabilities (only if soft-clustering)
			dim_list			list of dimensions to consider for distances. If None all dimensions are used
		Output:
			labels (N,)/(N,K)	labels/probability of clustering
		"""
			#checking for shape
		if X.ndim == 1:
			X = X.reshape((X.shape[0],1))

		if dim_list is None:
			dim_list = range(X.shape[1])
		else:
			assert len(dim_list) <= X.shape[1]

		dist = np.zeros((X.shape[0],self.K)) #keeps distance of points i to centroids k
		for k in range(self.K):
			temp = np.divide((X[:,dim_list] - self.mu[dim_list,k]), self.sigma[dim_list,k]) #(N,D')
			dist[:,k] = np.sum(np.square(temp),axis = 1) #(N,)
		if hard_clustering:
			to_return = np.argmin(dist, axis = 1) #(N,)
			return to_return
		else:
			if log_prob:
				return -0.5*dist
			probs = np.exp(-0.5*dist) #(N,K)
			return probs

	def LL(self,X):
		"""
		Returns the LL of the model for the given data.
		Input:
			X (N,D)	data
		Ouput:
			LL	log-ikelihood for the data
		"""
		log_prob = self.predict(X, hard_clustering = False, log_prob = True) # (N,K) log_prob (distances)
		log_prob = log_prob[range(log_prob.shape[0]),np.argmax(log_prob, axis = 1)] #(N,) picking the smallest value

		return np.sum(log_prob) / X.shape[0]
		
	def fit(self, X_train, N_iter=None, threshold = 1e-2):
		"""
		Fit the model EM algorithm for minimizing the LL.
		Input:
			X_train (N,D)	train data to cluster
			N_iter			Maximum number of iteration (if None only threshold is applied)
			threshold		Minimum change in LL below which algorithm is terminated
		Output:
			history		list of value for the LL of the model at every epoch
		"""
		if X_train.ndim == 1:
			X_train = np.reshape(X_train, (X_train.shape[0],1))
		if X_train.shape[1] != self.D:
			raise TypeError("Wrong shape for X_train matrix "+str(X_train.shape)+". Second dimension should have lenght "+str(self.D))

		if not self.init: #initialising
			init_guess = np.linspace(np.min(X_train,axis =0),np.max(X_train, axis =0), num = self.K)
			print(init_guess.shape)
			self.mu = init_guess.T

		i = 0
		old_LL = -1e5
		LL = -1e4
		history=[]
		while (LL - old_LL > threshold ): #debug
				#do batch update!!!
			old_LL = LL
			self.EM_step(X_train)
			LL=self.LL(X_train)
			history.append(LL)
			i += 1
			print("LL at iter "+str(i)+"= ",LL)
			if N_iter is not None:
				if i>= N_iter:
					break
			assert LL - old_LL >=0 #LL must always increase in EM algorithm. Useful check
		return history

	def EM_step(self, X):
		"""
		Does one EM update.
		Input:
			X (N,D)	train data
		"""
			#E step
		r = self.predict(X, hard_clustering = True) #(N,)

			#M step
		to_save = []
		for k in range(self.K):
			#print(len(np.where(r==k)[0]))
			X_k = X[np.where(r==k)[0],:] #(N_k, D)
			if len(X_k) ==0:
				print("Cluster "+str(k)+" empty")
				continue
			else:
				to_save.append(k)
			#self.mu[:,k] = np.mean(X_k, axis =0) #(D,)
			self.mu[:,k] = (np.max(X_k, axis = 0)+np.min(X_k, axis = 0))/2. #more robust than average

				#dealing with sigma
			if self.fit_sigma is not None:
				#self.sigma[self.fit_sigma,k] = np.std(X_k - self.mu[:,k], axis = 0)[self.fit_sigma]
				self.sigma[self.fit_sigma,k] = (np.max(X_k, axis = 0)-np.min(X_k, axis=0))[self.fit_sigma]/2.

			#print(self.mu[:,k],self.sigma[:,k]) #debug

			#sigma_0 = (np.var(X_k, axis = 0)) /np.power(self.K, 1./self.D) #(D,D) for regularization
			#sigma_0 = np.diag(np.full((self.D,), 1e-3)) #(2,2)
			#sigma_k = np.matmul((X_k - self.mu[:,k]).T, (X_k - self.mu[:,k])) / X_k.shape[0] #(D,D)
			#sigma_k = np.sum(np.square(X_k - self.mu[:,k]), axis = 0) / X_k.shape[0] #(D,)
			#self.sigma[:,k] = (sigma_k + sigma_0)/(X_k.shape[0]+2*(self.D+1)) #np.ones((self.D,))
			#self.sigma[:,k] = sigma_k + sigma_0


		self.mu = self.mu[:,to_save]
		self.sigma = self.sigma[:,to_save]
		assert self.mu.shape == self.sigma.shape
		self.K = int(self.mu.shape[1])
		
		return

	def get_params(self):
		"""
		Returns parameters of the model.
		Output:
			params	list of model params [D,K, centroids (D,K), sigma(D,)]
		"""
		return [self.D,self.K, self.mu, self.sigma]

	def accuracy(self, X, true_labels):
		"""
		Computes the accuracy of the model (i.e. the fraction of misclassified points).
		Input:
			X (N,D)				test points
			true_labels (N,)	true labels of test points
		Output:
			accuracy	accuracy of the predictions made at test points
		"""
		if X.ndim == 1:
			X = np.reshape(X, (X.shape[0],1))
		y_pred = self.predict(X)
		return np.sum(y_pred==true_labels)/float(true_labels.shape[0])
		

class predictor_lin_fit_cluster(object):
	"""
	Given a predictor of hard clustering classes, it makes a linear fit of data in each cluster class.
	The model has the form:
		p(y|x) = (w_k*x+b_k) delta_{k,z*}
	with z*= cluster with highest responsibility for x; z* = argmax_k p(x|z=k).
	Cluster object must have a method predict() which return the cluster class (as positive integer) of the test point.

	OSS: You should try to do a fit more robust to outliers... Check how to do it but probably the way to do is L1 regularization!!
	"""
	def __init__(self,D, K, cluster_object):
		"""
		Initialize the object.
		Input:
			D				dimensionality of the indipendent variable space
			K				number of cluster classes
			cluster_object	any object representing a clustering algorithm (must be already fitted)
		"""
		self.D = D
		self.K = K
		self.cluster_object = cluster_object
			#weights
		self.w = np.zeros((self.D+1,self.K))
		return

	def predict(self,X):
		"""
		Makes model predictions.
		Input:
			X (N,D)	data
		Ouput:
			y (N,)	predicted labels 
		"""
			#checking for shape
		if X.ndim == 1:
			X = np.reshape(X, (X.shape[0],1))
		if X.shape[1] == self.D:
			X = np.concatenate((np.ones((X.shape[0],1)),X), axis = 1) #(N,D+1)
		if X.shape[1] != self.D+1:
			raise TypeError("Wrong shape for X matrix "+str(X.shape)+". Second dimension should have lenght "+str(self.D))

		data_labels = self.cluster_object.predict(X[:,1:]) #leaving out bias...
		indices = range(X.shape[0])
		y = np.zeros((X.shape[0],))
		for k in range(self.K):
			indices = np.where(data_labels==k)[0] #(D',)
			y[indices] = np.matmul(X[indices,:], self.w[:,k]) #(N,)

		return y

	def fit(self, X_train, y_train, data_cluster = None, regularizer = 1e-3, loss = "L1"):
		"""
		Fits the model.
		Input:
			X_train (N,D)		train indipendent variables
			y_train (N,)		train targets
			data_cluster (N,D')	any dataset that can be useful to label train data (if None, X_train is used)
			regularizer			regularization constant (only applies to L2 loss function)
			loss				loss function to be used for linear fit ("L2", "L1" or "L1_linprog" are implemented)
		"""
					#checking for shape
		if X_train.ndim == 1:
			X_train = np.reshape(X_train, (X_train.shape[0],1))
		if X_train.shape[1] == self.D:
			X_train = np.concatenate((np.ones((X_train.shape[0],1)),X_train), axis= 1) #(N,D+1)
		if X_train.shape[1] != self.D+1:
			raise TypeError("Wrong shape for X_train_train matrix "+str(X_train.shape)+". Second dimension should have lenght "+str(self.D))

		if data_cluster is None:
			data_cluster = X_train[:,1:] #leaving out bias...
		else:
			assert data_cluster.shape[0] == X_train.shape[0]

			#starting proper fitting
		data_labels = self.cluster_object.predict(data_cluster)
		for k in range(self.K):
			X_k = X_train[np.where(data_labels==k)[0],:] #(N_k, D)
			y_k = y_train[np.where(data_labels==k)[0]]
			if len(X_k) ==0:
				continue

			if loss == "L2":
					#L2 loss function (less robust to outliers)
				reg_matrix = np.diag([0]+[1 for i in range(self.D)]) #bias is not regularized
				temp = np.matmul(np.linalg.inv(regularizer*reg_matrix + np.matmul(X_k.T,X_k)), X_k.T) #(D+1,N)
				temp = np.matmul(temp, y_k) #(D+1,)
				self.w[:,k] = temp

			elif loss == "L1":
					#L1 loss function (more robust to outliers)
					#using brute force with BFGS algorithm for minimizing L1 loss
				L1_loss = lambda w: np.sum(np.abs(y_k-np.matmul(X_k,w))) #w must have shape (D+1,)

					#intial guess is L2 solution
				init = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_k.T,X_k)), X_k.T),y_k) #(D+1,N)

				res = scipy.optimize.fmin_bfgs(L1_loss, x0 = init, disp = False) 
				self.w[:,k] = res[:]

			elif loss == "L1_linprog": #Doesn't work!!
					#L1 loss function (more robust to outliers)
					#setting up a linear program for L1 regression... 
				c = np.concatenate((np.zeros(X_k.shape[1],), np.ones(X_k.shape[0]*2,))) #(D+N'+N',) [w, r+, r-]
				A_eq = np.concatenate((X_k, np.identity(X_k.shape[0]), -np.identity(X_k.shape[0])), axis = 1) #(N, D+N'+N')
				b_eq = y_k
				bounds = [(None,None) for i in range(X_k.shape[1])] + [(0., None) for i in range(X_k.shape[0]*2)]
				opt_dict = {"maxiter":10}

				res = scipy.optimize.linprog(c, A_eq = A_eq, b_eq = b_eq, bounds = bounds, options = opt_dict)#, method = 'simplex')
				continue
				if res["success"] == True:
					self.w[:,k] = res["x"][0:X_k.shape[1]]
				else:
					print("Optimization failed with message: ")
					print("\t",res['message'])
			else:
				print("Chosen loss function "+loss+" not implemented! Function has not been fitted")

		return

	def evaluate_loss(self, X, y):
		"""
		Evaluate fit residuals with MSE.
		Input:
			X (N,D)		indipendent variables
			y (N,)		targets
		Output:
			mse		mean square error
		"""
		y_pred = self.predict(X)

		return np.mean(np.square(y_pred-y))

	def get_weights(self):
		"""
		Return the weights of the model.
		Output:
			weights (D+1,K)	weights of the model
		"""
		return self.w

	def get_params(self,):
		"""
		Return parameters D and K.
		Output:
			D	dimensionality of input space
			K	number of clusters
		"""
		return [self.D,self.K]

###########
#	K means linear fit
###########

class K_means_linfit(object):
	"""
	This class represents a K means + linear fit model:
		p((y,x) | params) = 1/K sum_{k=1}^K p((y,x)|z=k) = 1/K sum_{k=1}^K N((x,y)| (y_0,x_0) ; SIGMA_INV_k)
	with SIGMA_INV_k a linear regression matrix + variance in x coordinates
		[[1, -w],[-w.T, w*w.T + sigma**2 * lambda_k]]/sigma**2
	with w_k (D,) and lambda_k (D,D)
	The model is fitted with EM algorithm which also sets K (number of clusters)
	The model can make clustering with hard-clustering procedure:
		z* = argmax_k p((y,x)|z=k)
	or with soft-clustering:
		z_k ~ p((y,x)|z=k)
	Inference can be performed with standard formulas:
		p(y|x, z=z*) = N(y|w*(x-x_0)+y_0, sigma**2)

	OSS: The model is useful since it's a compact way to write a non trivial covariance matrix. Might help?
	OSS2: To implement... Try a model in which you delete cluster with too few examples in it and split cluster with low likelihood
	"""
	def __init__(self, D, sigma = 1e-3):
		"""
		Initialize the model.
		Input:
			D				dimensionality of x space (thus (x,y) has dimension D+1)
			sigma			std dev for the univariate guassian of the residuals
		"""
		self.D = D
		self.sigma = sigma #std of univariate residuals
		self.K = 1 #dummy intialization
		self.mu = np.zeros((D+1,self.K)) #centroids
		self.w = np.zeros((D,self.K)) #initialization
		self.lambda_k = []
		for i in range(self.K):
			self.lambda_k.append(np.identity(self.D)/sigma**2)
		return

	def get_covariance(self, weights = None):
		"""
		Returns the inverse covariance matrix given some weights.
		Input
			weights (D,K)	weights of each cluster
		Output
			lambda_list []		list of K covariance matrices (with shape (D+1,D+1)) for each cluster
		"""
		lambda_list = []
		if weights is None:
			weights = self.w

		for k in range(weights):
			w_k = np.reshape(weights[:,k], (D,1))
			first_row = np.concatenate((np.ones((1,1)), -w_k.T), axis = 1) #(1,D+1)
			second_row = np.matmul(w,w.T) + self.sigma**2 * (self.lambda_k[k])
			second_row = np.concatenate((w,second_row), axis = 1) #(D,D+1)
			sigma_inv_k = np.concatenate((first_row, second_row), axis =0)
			lambda_list.append(lambda_k)
		return lambda_k

	def predict(self, X, y = None, hard_clustering = True, log_prob = True):
		"""
		Makes predictions for labels of each data.
		If y is not given the predicted cluster label is computed based only on X (only hard clustering).
		Input:
			X (N,D)(N,D+1)		data to make predictions for
			y (N,)				targets for data (if None y must be last column of X or predicted y label is returned)
			hard_clustering		whether to do hard clustering
			log_prob 			whether to return log probabilities (only if soft-clustering)
		Output:
			labels (N,)/(N,K)	labels/probability of clustering
		"""
		if y is not None:
			X,y = self.__check_adjust__(X,y)
		else:
			if X.shape[1] == self.D:
				return self.predict_y(X, get_labels = True)
			if X.shape[1] == self.D+1:
				y = X[:,X.shape[1]-1]
				X = X[:,:X.shape[1]-1]
			else:
				raise TypeError("X type not understood: if y is not given, shape[1] must be either "+str(self.D)+" or "+str(self.D+1))

		dist = np.zeros((X.shape[0],self.K)) #keeps distance of points i to centroids k

		for k in range(self.K):
			y_ = y - self.mu[0,k]
			X_ = X - self.mu[1:,k]
			temp = ( y_ - np.matmul(X_,self.w[:,k]) ) / self.sigma#(N,)
			dist[:,k] = (np.square(temp)) #(N,)
			dist[:,k] += np.sum(np.multiply(np.matmul(X_,self.lambda_k[k]),X_), axis = 1) #(N,) 
		if hard_clustering:
			to_return = np.argmin(dist, axis = 1) #(N,)
			return to_return
		else:
			if log_prob:
				return -0.5*dist
			probs = np.exp(-0.5*dist) #(N,K)
			return probs

	def predict_y(self,X, get_labels = False):
		"""
		Makes the best prediction according to the model:
			p(y|x) = N(y| (w_z*)*(x-x_0)+y_0, sigma**2)
		with z* = argmax_k p_k(x) = argmax_k int_R dy p(x,y) = argmax_k exp(-.5*(x-x_0_k)*lambda_k[k]*(x-x_0_k))
		Input:
			X (N,D)		points to make prediction at
			get_labels	whether to get cluster prediction instead of y
		Output:
			y (N,)		predictions (or labels depending on get_labels)
		"""
		X = self.__check_adjust__(X)
		
			#making prediction
		dist = np.zeros((X.shape[0], self.K))
		for k in range(self.K):
			X_k = X - self.mu[1:,k]
			dist[:,k] = np.sum(np.multiply(np.matmul(X_k,self.lambda_k[k]),X_k), axis = 1)
		labels = np.argmin(dist, axis = 1) #(N,)
		if get_labels:
			return labels

			#making regression
		X_ = X - self.mu[1:,labels].T #(N,D)
		weights = self.w[:,labels].T
		y_0 = self.mu[0,labels]
		y = np.sum(np.multiply(X_,weights), axis = 1) + y_0 #(N,)

		return y

	def LL(self,X, y):
		"""
		Returns the LL of the model for the given data.
		Input:
			X (N,D)	data
			y (N,)	targets
		Ouput:
			LL	log-ikelihood for the data
		"""
		X,y = self.__check_adjust__(X,y)
		log_prob = self.predict(X,y, hard_clustering = False, log_prob = True) # (N,K) log_prob (distances)
		log_prob = log_prob[range(log_prob.shape[0]),np.argmax(log_prob, axis = 1)] #(N,) picking the highest value

		return np.sum(log_prob) / X.shape[0]

	def __initialise_weights__(self, X_train, y_train, K_0):
		"""
		Makes a nice guess for the initial weights of the model.
		The guesses are stored in local variables self.w, self.mu
		Input:
			X_train (N,D)	train data
			y_train (N,)	train targets
			K_0				initial guess for the number of clusters
		"""
		self.K = int(K_0)
		self.mu = np.zeros((self.D+1,self.K))
		self.w = np.zeros((self.D,self.K))

		if X_train.shape[0] > 4*K_0:
			X = X_train[:4*K_0,:]
			y = y_train[:4*K_0]
		else:
			X = X_train
			y = y_train
		N = X.shape[0]
		data = np.vstack((y,X.T)).T

			#choosing centroids
			#points are chosen from dataset with farhtest point clustering
		ran_index = np.random.choice(N)
		self.mu[:,0] = data[ran_index]

		for k in range(1,K_0):
			distances = np.zeros((N,k)) #(N,K_0)
			for k_prime in range(k):
				distances[:,k_prime] = np.sum(np.square(data - self.mu[:,k_prime]), axis =1) #(N,K')
			distances = np.min(distances, axis = 1) #(N,)
			distances /= np.sum(distances) #normalizing distances to make it a prob vector
			next_cl_arg = np.random.choice(range(X.shape[0]), p = distances) #chosen argument for the next cluster center
			self.mu[:,k] = data[next_cl_arg,:]

			#choosing initial w
		for k in range(K_0):
			distances = np.sum(np.square(data - self.mu[:,k]), axis=1) #(N,)
			indices = np.argsort(distances)
			indices = [indices[0],indices[1], indices[2]]
			X_k = data[indices,1:] #(3,D)
			y_k = data[indices,0] #(3,)
			self.w[:,k] = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_k.T,X_k)), X_k.T),y_k) #initial guess for k cluster weights

		self.lambda_k = []
		for i in range(self.K):
			self.lambda_k.append(np.identity(self.D)/self.sigma**2)

		return

	def __merge_clusters__(self, X, y, threshold = 1e-2):
		"""
		Find and merge clusters which can be explained well by a single cluster. This can help to prevent overfitting.
		New clusters parameters are NOT fitted!
		Input:
			X (N,D)		train data
			y (N,)		train targets
			threshold	maximum tolerance for a LL increase when merging two clusters
		"""
		changed = True

		while changed:
			to_remove = []
			changed = False
			for k in range(self.K):
				if k in to_remove:
					continue
					#for each cluster finding the most similar cluster
				mu_ = (self.mu.T - self.mu[:,k]).T
				dist = ( mu_[0,:] - np.matmul(mu_[1:,:].T,self.w[:,k]) ) / self.sigma #(K,)
				dist = (np.square(dist))
				#dist += np.sum(np.multiply(np.matmul(mu_[1:,:].T,self.lambda_k[k]),mu_[1:,:].T), axis = 1) #(N,) 
				#dist = np.sum(np.square(self.mu.T - self.mu[:,k]), axis = 1) #(K,)
				k_near = np.argsort(dist)[1]
				#print(k_near, dist[k_near])
				
					#cheking if LL doesn't grow
				labels = self.predict(X, y, hard_clustering = True)
				k_list = np.where(labels == k)[0]
				k_near_list = np.where(labels == k_near)[0]
				if len(k_near_list) ==0: continue

				X_k_near = X[k_near_list,:]
				y_k_near = y[k_near_list]
				X_k = X[k_list,:]
				y_k = y[k_list]

				res_old = np.mean(np.square( y_k_near - np.matmul(X_k_near,self.w[:,k_near]) )) #(N'',)
					#checking if k_new can fit in cluster k
				res_new = np.mean(np.square( y_k_near - np.matmul(X_k_near,self.w[:,k]) ))  #(N,)

				#print(res_old, res_new, (res_new - res_old)/res_old,self.mu[1,k], self.mu[1,k_near])
				#if np.abs(res_new - res_old)/res_old <= threshold:
				if dist[k_near] < threshold:
					#pass
						#removing k_near from possible clusters
					to_remove.append(k_near)
					#print("cluster killed", k, k_near, self.mu[1,k_near])
						#DO M STEP TO MERGE NEW CLUSTERS!!!
					y_k = np.concatenate((y_k,y_k_near))
					X_k = np.concatenate((X_k, X_k_near), axis= 0)
					if X_k.shape[0] >= 2:
						self.__M_step__(X_k,y_k,k)
					else:
						to_remove.append(k)
					changed = True
			
				#removing items weights
			to_save = [i for i in range(self.K) if not (i in to_remove)] #list of clusters to keep
			self.mu = self.mu[:,to_save]
			self.w = self.w[:,to_save]
			assert self.mu.shape[1] == self.w.shape[1]
			self.K = int(self.mu.shape[1])
			self.lambda_k = [self.lambda_k[i] for i in range(len(self.lambda_k)) if i in to_save]
			assert len(self.lambda_k) == self.K
			if len(to_remove) !=0:
				print("\tRemoved ", len(to_remove), " clusters")
		self.EM_step(X,y)

		return


	def fit(self, X_train, y_train, K_0, N_iter=None, threshold = 1e-2):
		"""
		Fit the model EM algorithm for minimizing the LL.
		Input:
			X_train (N,D)	train data
			y_train (N,)	train targets
			K_0				initial guess for the number of clusters
			N_iter			Maximum number of iteration (if None only threshold is applied)
			threshold		Minimum change in LL below which algorithm is terminated
		Output:
			history		list of value for the LL of the model at every epoch
		"""
		X_train,y_train = self.__check_adjust__(X_train,y_train)

		self.__initialise_weights__(X_train, y_train, K_0) #smart way to initialise weights

		i = 0
		old_LL = -1e5
		LL = -1e4
		history=[]
		while True:#(LL - old_LL > threshold ): #debug
				#do batch update!!!
			old_LL = LL
			self.EM_step(X_train, y_train)
			if self.K > 2:
				self.__merge_clusters__(X_train, y_train)
			LL=self.LL(X_train,y_train)
			history.append(LL)
			i += 1
			print("LL at iter "+str(i)+"= ",LL)
			if N_iter is not None:
				if i>= N_iter:
					break
			#assert LL - old_LL >=0 #LL must always increase in EM algorithm. Useful check but wrong


		return history


	def EM_step(self, X, y):
		"""
		Does one EM update.
		Input:
			X (N,D)			train data
			y (N,)			train targets
		"""
		X,y = self.__check_adjust__(X,y)
		LL = np.zeros((X.shape[0]))
			#here some cluster cleaning procedure should be tried to merge similar cluster...

			#E step
		r = self.predict(X, y, hard_clustering = True) #(N,)

			#M step
			#doing linear fit with L1 loss
		to_save = []
		for k in range(self.K):
			X_k = X[np.where(r==k)[0],:] #(N_k, D)
			y_k = y[np.where(r==k)[0]]

			res_M_step = self.__M_step__(X_k,y_k,k)
			if res_M_step == False: #checking whether M step was successful
				print("\tCluster "+str(k)+" empty!")
				continue
			else:
				to_save.append(k)
				LL[k] = self.LL(X_k,y_k)
			#print("\tLL cluster "+str(k)+" with center ", self.mu[1,k],": ", self.LL(X_k,y_k))
		
		self.mu = self.mu[:,to_save]
		self.w = self.w[:,to_save]
		assert self.mu.shape[1] == self.w.shape[1]
		self.K = int(self.mu.shape[1])
		self.lambda_k = [self.lambda_k[i] for i in range(len(self.lambda_k)) if i in to_save]
		assert len(self.lambda_k) == self.K

		return

	def __M_step__(self, X_k, y_k, k):
		"""
		Computes cluster parameters (mu, w, lambda) for the k-th cluster and stores them into the suitable variables.
		Input:
			X_k (N,D)	data within k-th cluster
			y_k (N,D)	targets within k-th cluster
			k			number of cluster to consider
		Output:
			res (bool)	whether M step was successful
		"""
		if X_k.shape[0] <= 2: #too few data in the cluster to perform M step
			return False

		#updating centroid k
		data = np.vstack((y_k,X_k.T)).T
		mu_index = np.argmin(np.linalg.norm(data - np.mean(data, axis =0)))
		self.mu[:,k] = data[mu_index,:]

		self.mu[1:,k] = np.mean(X_k, axis =0) #(D,)
		self.mu[0,k] = np.mean(y_k) #()

				#updating lambda_k (with a prior to regularize things)
		if X_k.shape[1] == 1:
			cov_mat = np.var(X_k).reshape((1,1)) * X_k.shape[0]
		else:
			cov_mat = np.cov(X_k, rowvar = False) * X_k.shape[0] #(D,D)
		s_0 = np.diag(np.var(X_k, axis =0)) / np.power(self.K,1./self.D) #(D,D) #prior on variance to prevent overfitting
		cov_mat = (s_0 + cov_mat)/(X_k.shape[0] + 2*(self.D+2))
		#print(X_k, cov_mat)
		if np.linalg.matrix_rank(cov_mat) < self.D:
			return False #matrix not invertible: M step cannot be performed

		self.lambda_k[k] = np.linalg.inv(cov_mat)

				#doing linear fit for w (fitting differences between cluster centers)
		X_k = X_k - self.mu[1:,k]
		y_k = y_k - self.mu[0,k]
				#probably L2 is clearer from the theoretical point of view. L1 works better in practise: advisable to use that
		L1_loss = lambda w: np.sum(np.abs(y_k-np.matmul(X_k,w))) #w must have shape (D,)
				#intial guess is L2 solution
		init = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_k.T,X_k)), X_k.T),y_k) #(D+1,N)
				#finding minimum of loss function
		res = scipy.optimize.fmin_bfgs(L1_loss, x0 = init, disp = False) 
		self.w[:,k] = res

		return True

	def __split_cluster__(self, k, X, y):
		"""
		Split cluster k in two clusters.
		Input:
			k	index of cluster to be splitted
		"""
		r = self.predict(X, y, hard_clustering = True) #(N,)
		X_k = X[np.where(r==k)[0],:] #(N_k, D)
		y_k = y[np.where(r==k)[0]]
		data = np.vstack((y_k,X_k.T)).T
		indices = np.random.choice(data.shape[0],2,replace =False) 

		mu_1 = data[indices[0],:]
		mu_2 = data[indices[1],:]
		w = self.w[:,k]

		self.mu[:,k] = mu_1
		print(self.mu.shape, mu_2.shape)
		self.mu = np.vstack([self.mu.T, mu_2]).T
		self.w = np.vstack([self.w.T, w]).T
		self.lambda_k.append(self.lambda_k[k])
		self.K += 1
		print(self.K)

		return


	def __check_adjust__(self,X,y = None):
		"""
		Add an extra dimension if X is (N,), check if X.shape[0]==y.shape[0] and check if X.shape[1] == D.
		Input:
			X (N,D)			data
			y (N,)			targets (if None no check on y is done)
		Output:
			X (N,D)			adjusted data
			y (N,)			adjusted targets
		"""
		if X.ndim == 1:
			X = np.reshape(X, (X.shape[0],1))
		if X.shape[1] != self.D:
			raise TypeError("Wrong shape for X_train matrix "+str(X.shape)+". Second dimension should have lenght "+str(self.D))
		if y is not None:		
			if X.shape[0]!=y.shape[0]:
	 			raise TypeError("X matrix and y matrix have different number of data points "+str(X.shape[0])+" and "+ str(y.shape[0]))
		if y is None:
			return X
		else:
			return X,y

	def get_params(self):
		"""
		Return the weights of the model.
		Output:
			params [] 	list of params [D,K, means (D,K), weights (D,K), lambda_k [(D,D)]*K]
		"""
		return [self.D,self.K, self.mu, self.w, self.lambda_k]





