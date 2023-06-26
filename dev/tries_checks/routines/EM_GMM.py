############
#	class for EM algorithm in GMM downloaded from: https://zhiyzuo.github.io/EM/#applying-em-on-gaussian-mixtures
############
import scipy as sp
import scipy.stats
import numpy as np

class GMM(object):
	def __init__(self, X, k=2):
		# dimension
		X = np.asarray(X)
		self.m, self.n = X.shape
		self.all_data = X.copy()
		self.data = self.all_data
		# number of mixtures
		self.k = k

	def initialize(self, X):
		"Small function used for reinitialization of data with a public method. k and n remains the same!"
		# dimension
		X = np.asarray(X)
		if self.n != X.shape[1]:
			print("Things are not the same as before!!")
			return
		self.m, self.n = X.shape
		self.all_data = X.copy()
		self.data = self.all_data
		self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
		
	def _init(self):
		# init mixture means/sigmas
		self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
		self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
		self.phi = np.ones(self.k)/self.k
		self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
		#print(self.mean_arr)
		#print(self.sigma_arr)
	
	def fit(self, tol=1e-4, batches = False):
		self._init()
		num_iters = 0
		ll = 1
		previous_ll = 0
		while(ll-previous_ll > tol):
			if batches:
				np.random.shuffle(self.all_data)
				N_batch = int(self.all_data.shape[0]/5)
				self.data = self.all_data[0:N_batch,:]
				self.m = N_batch
				self.w = np.asmatrix(np.empty((self.m, self.k), dtype=float))
			previous_ll = self.loglikelihood()
			self._fit()
			num_iters += 1
			ll = self.loglikelihood()
			print('Iteration %d: log-likelihood is %.6f'%(num_iters, ll))
		print('Terminate at %d-th iteration:log-likelihood is %.6f'%(num_iters, ll))
	
	def loglikelihood(self):
		ll = 0
		for i in range(self.m):
			tmp = 0
			for j in range(self.k):
				#print(self.sigma_arr[j])
				tmp += sp.stats.multivariate_normal.pdf(self.data[i, :], 
														self.mean_arr[j, :].A1, 
														self.sigma_arr[j, :]) *\
					   self.phi[j]
			#if tmp == 0:
			#	tmp = 1e-100
			ll += np.log(tmp) 
		return ll
	
	def _fit(self):
		self.e_step()
		self.m_step()
		
	def e_step(self):
		# calculate w_j^{(i)}
		for i in range(self.m):
			den = 0
			for j in range(self.k):
				num = sp.stats.multivariate_normal.pdf(self.data[i, :], 
													   self.mean_arr[j].A1, 
													   self.sigma_arr[j]) *\
					  self.phi[j]
				den += num
				self.w[i, j] = num
			self.w[i, :] /= den
			assert self.w[i, :].sum() - 1 < 1e-4
			
	def m_step(self):
		sigma_0 = np.diagflat(np.var(self.data, axis=0)) / 1.
		nu_0 = sigma_0.shape[0] + 2
		for j in range(self.k):
			const = self.w[:, j].sum() #responsibility of j-th cluster
			self.phi[j] = 1/self.m * const #PUT A PRIOR HERE!!!!
			_mu_j = np.zeros(self.n)
			_sigma_j = np.zeros((self.n, self.n))
			for i in range(self.m):
				_mu_j += (self.data[i, :] * self.w[i, j])
				_sigma_j += self.w[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
				#print((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
			self.mean_arr[j] = _mu_j / const
			#self.sigma_arr[j] = _sigma_j / const #no regularization
			self.sigma_arr[j] = (_sigma_j + sigma_0) / (const + sigma_0.shape[0] + 2 + nu_0) #regularized term
		#print(self.sigma_arr)


	def get_cluster_indices(self,dataset, cluster_number):
		"""
		Return indices of data in dataset belonging to cluster_number according to model GMM_model
		"""
		self.__init__(dataset, self.k)
		self.e_step() #got responsibilities
		max_cluster = np.argmax(self.w, axis = 1)
		indices = []
		for i in range(dataset.shape[0]):
			if max_cluster[i] == cluster_number:
				indices.append(i)
		return indices

####old get_cluster_indices
def get_cluster_indices(dataset, cluster_number, GMM_model):
	"""
	Return indices of data in dataset belonging to cluster_number according to model GMM_model
	"""
	GMM_model.data = dataset.copy()
	GMM_model.m, GMM_model.n = dataset.shape
	GMM_model.w = np.asmatrix(np.empty((GMM_model.m, GMM_model.k), dtype=float))
	GMM_model.e_step() #got responsibilities
	max_cluster = np.argmax(GMM_model.w, axis = 1)
	indices = []
	for i in range(dataset.shape[0]):
		if max_cluster[i] == cluster_number:
			indices.append(i)
	return indices



