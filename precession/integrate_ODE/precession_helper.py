"""
Module precession_helper.py
===========================
	Module for training a ML model for fitting the precessing angles alpha, beta as a function of (theta1, theta2, deltaphi, chi1, chi2, q).
	Requires precession module (pip install precession) and tensorflow (pip install tensorflow)
"""

#Issues:
#	set r_0 properly

import numpy as np
import precession
import os
import sys
try:
	import silence_tensorflow.auto #awsome!!!! :)
except:
	pass
import tensorflow as tf

def get_alpha_beta(q, chi1, chi2, theta1, theta2, delta_phi, r_0, times, verbose = False):
	"""
get_alpha_beta
==============
	Returns angles alpha and beta by solving PN equations for spins. Uses module precession.
	Angles are evaluated on a user-given time grid (units: s/M_sun) s.t. the 0 of time is at separation r = M_tot.
	Inputs:
		q (N,)				mass ratio (>1)
		chi1 (N,)			dimensionless spin magnitude of BH 1 (in [0,1])
		chi1 (N,)			dimensionless spin magnitude of BH 2 (in [0,1])
		theta1 (N,)			angle between spin 1 and L
		theta2 (N,)			angle between spin 2 and L
		delta_phi (N,)		angle between in plane projection of the spins
		r_0					initial separation (in natural units)
		times (D,)			times at which alpha, beta are evaluated (units s/M_sun)
		verbose 			whether to suppress the output of precession package
	Outputs:
		alpha (N,D)		alpha angle evaluated at times
		beta (N,D)		beta angle evaluated at times
	"""
	M_sun = 4.93e-6
	if isinstance(q,float):
		q = np.array(q)
		chi1 = np.array(chi1)
		chi2 = np.array(chi2)
		theta1 = np.array(theta1)
		theta2 = np.array(theta2)
		delta_phi = np.array(delta_phi)

	if len(set([q.shape, chi1.shape, chi2.shape, theta1.shape, theta2.shape, delta_phi.shape])) != 1:
		raise RuntimeError("Inputs are not of the same shape (N,). Unable to continue")

	if q.ndim == 0:
		q = q[None]
		chi1 = chi1[None]; chi2 = chi2[None]
		theta1 = theta1[None]; theta2 = theta2[None]; delta_phi = delta_phi[None]
		squeeze = True
	else:
		squeeze = False

		#initializing vectors
	alpha = np.zeros((q.shape[0],times.shape[0]))
	beta = np.zeros((q.shape[0],times.shape[0]))
	
	if not verbose:
		devnull = open(os.devnull, "w")
		old_stdout = sys.stdout
		sys.stdout = devnull
	else:
		old_stdout = sys.stdout

		#computing alpha, beta
	for i in range(q.shape[0]):
			#computing initial conditions for the time evolution
		q_ = 1./q[i] #using conventions of precession package
		M,m1,m2,S1,S2=precession.get_fixed(q_,chi1[i],chi2[i]) #M_tot is always set to 1

		#print(q_, chi1[i], chi2[i], theta1[i],theta2[i], delta_phi[i], S1, S2, M)
			#nice low level os thing
		print("Generated angle "+str(i)+"\n")
		#old_stdout.write("Generated angle "+str(i)+"\n")
		#old_stdout.flush()

		xi,J, S = precession.from_the_angles(theta1[i],theta2[i], delta_phi[i], q_, S1,S2, r_0) 

		J_vec,L_vec,S1_vec,S2_vec,S_vec = precession.Jframe_projection(xi, S, J, q_, S1, S2, r_0) #initial conditions given angles

		r_f = 1.*M
		sep = np.linspace(r_0, r_f, 10000)

		Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, t = precession.orbit_vectors(*L_vec, *S1_vec, *S2_vec, sep, q_, time = True) #time evolution of L, S1, S2
		L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
		
		temp_alpha = np.unwrap(np.arctan2(Ly,Lx))
		temp_beta = np.arccos(Lz/L)
		
		alpha[i,:] = np.interp(times, (t-t[-1])*M_sun, temp_alpha)
		beta[i,:] = np.interp(times, (t-t[-1])*M_sun, temp_beta)
	
	if not verbose:
		devnull.close()
		sys.stdout = old_stdout

	if squeeze:
		return np.squeeze(alpha), np.squeeze(beta)

	return alpha, beta


class angle_generator():
	"This class provides a generator of angles for the training of the NN."
	def __init__(self, t_min, N_times, ranges, N_batch = 100, replace_step = 1):
		"Input the size of the time grid and the starting time from the angle generation. Ranges for the 6 dimensional inputs shall be provided by a (6,2) array"
		self.t_min = np.abs(t_min)
		self.N_times = N_times
		self.N_batch = N_batch
		self.ranges = ranges #(6,2)
		self.replace_step = replace_step #number of iteration before computing a new element
		self.dataset = np.zeros((N_batch*N_times, 9 )) #allocating memory for the dataset
		return

	def replace_angle(self, i):
		"Updates the angles corresponding to the i-th point of the batch. They are inserted in the dataset."
		M_sun = 4.93e-6
		params = np.random.uniform(self.ranges[:,0], self.ranges[:,1], size = (6,))
		times = np.random.uniform(-self.t_min, 0., (self.N_times,))
		r_0 = 2.5 * np.power(self.t_min/M_sun, .25) #starting point for the r integration #look eq. 4.26 Maggiore
		alpha, beta = get_alpha_beta(*params, r_0, times, verbose = False)

		new_data = np.repeat(params[None,:], self.N_times, axis = 0) #(N,6)
		new_data = np.concatenate([times[:,None], new_data, alpha[:,None], beta[:,None]], axis =1) #(N,9)

		id_start = i*(self.N_times)
		self.dataset[id_start:id_start + self.N_times,:] = new_data
		return

	def __call__(self):
		"Return a dataset of angles: each row is [t, q, chi1, chi2, theta1, theta2, deltaphi, alfa, beta]"
		for i in range(self.N_batch):
			self.replace_angle(i) #changing the i-th angle in the datset
		print("Generated starting dataset")

		i = -1
		j = 0
		while True:
			yield self.dataset
			if j % self.replace_step == 0 and j !=0:
				if i== (self.N_batch-1):
					i =0
				else:
					i +=1
				self.replace_angle(i)
			j+=1

###############################################################################################################
###############################################################################################################
###############################################################################################################

class NN_precession(tf.keras.Model):

	def __init__(self, name = "NN_precession_model"):
		super(NN_precession, self).__init__(name = name)
		print("Initializing model ",self.name)
		self.history = []
		self.metric = []
		self.epoch = 0

		self._l_list = []
		self._l_list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid) )
		self._l_list.append(tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)) #outputs: alpha, beta

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3) #default optimizer
		self.build(input_shape = (None, 7)) #This is required to specify the input shape of the model and to state which are the trainable paramters

	def call(self, inputs):
		"Inputs: [t, X_0 (n_vars,), Omega (n_params,)]"
		output = inputs
		for l in self._l_list:
			output = l(output)
		return output #(N,n_vars)

	def __ok_inputs(self, inputs):
		if not isinstance(inputs, tf.Tensor):
			inputs = tf.convert_to_tensor(inputs, dtype=tf.float32) #(N,D)
			if inputs.ndim == 1:
				inputs = inputs[None,:]
		return inputs

	def loss(self, X):
		"""
		Loss function: takes an input array X (N,9) with values to test the model at and the angles at those points.
		Input should be tensorflow only.
		"""
		loss = tf.math.square(self.__call__(X[:,:7]) - X[:,7:]) #(N,2)
		loss = tf.reduce_sum(loss, axis = 1) /X.shape[1] #(N,)
		return loss

	@tf.function#(jit_compile=True) #very useful for speed up
	def grad_update(self, X):
		"Input should be tensorflow only."
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			loss = tf.reduce_sum(self.loss(X))/X.shape[0]

		gradients = g.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		return loss

	def fit(self, generator, N_epochs, learning_rate = 5e-4, save_output = True, plot_function = None, save_step = 20000, print_step = 10):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) #default optimizer
		epoch_0 = self.epoch

		tf_dataset = tf.data.Dataset.from_generator(
     			generator,
    			output_signature = tf.TensorSpec(shape=(None,9), dtype=tf.float32)
					).prefetch(tf.data.experimental.AUTOTUNE)
		
		n_epoch = -1
		print(N_epochs)
		for X in tf_dataset:
			n_epoch +=1
			if n_epoch >= N_epochs:
				print("Ciao")
				break
			loss = tf.constant(0.)#self.grad_update(X)

			if n_epoch % print_step == 0: #saving history
				self.epoch = epoch_0 + n_epoch
				self.history.append((self.epoch, loss.numpy()))
				print(self.epoch, loss.numpy())
				if save_output:
					self.save_weights("{}/{}".format(self.name, self.name)) #overwriting the newest
					np.savetxt(self.name+"/"+self.name+".loss", np.array(self.history))
					np.savetxt(self.name+"/"+self.name+".metric", np.array(self.metric))

			if n_epoch == 0: continue

			if save_output:
				if n_epoch%save_step ==0: #computing metric loss
					metric = 0.
					self.metric.append((self.epoch, 0.))
					print("\tMetric: {} {}".format(self.metric[-1][0],self.metric[-1][1]))

					self.save_weights("{}/{}/{}".format(self.name, str(self.epoch), self.name)) #saving to arxiv
					if plot_function is not None:
						plot_function(self, "{}/{}".format(self.name, str(self.epoch)))
						
					
		return self.history

	def load_everything(self, path):
		"Loads model and tries to read metric and loss"
		print("Loading model from: ",path)
		self.load_weights(path)
		try:
			self.history = np.loadtxt(path+".loss").tolist()
			self.epoch = int(self.history[-1][0])
		except:
			self.epoch = 0
			pass

		try:
			self.metric = np.loadtxt(path+".metric").tolist()
		except:
			pass

		return






###############################################################################################################
###############################################################################################################
###############################################################################################################
def create_dataset_alpha_beta(N_angles, filename, N_grid, tau_min, q_range, chi1_range= (0.,1.), chi2_range = (0.,1.), theta1_range = (0., np.pi), theta2_range = (0., np.pi), delta_phi_range = (-np.pi, np.pi) ):
	"""
create_dataset_alpha_beta
=========================
	Creates a dataset for the angles alpha and beta.
	The dataset consist in parameter vector (q, chi1, chi2, theta1, theta2, delta_phi) associated to two vectors alpha and beta.
	User must specify a time grid at which the angles are evaluated at.
	More specifically, data are stored in 3 vectors:
		param_vector	vector holding source parameters (q, chi1, chi2, theta1, theta2, delta_phi)
		alpha_vector	vector holding alpha angle for each source evaluated at some N_grid equally spaced points
		beta_vector		vector holding beta angle for each source evaluated at some N_grid equally spaced points
	The values of parameters are randomly drawn within the user given constraints.
	Dataset is saved to file, given in filename and can be loaded with load_angle_dataset.
	Inputs:
		N_angles			Number of angles to include in the dataset
		filename			Name of the file to save the dataset at
		N_grid				Number of grid points
		tau_min				Starting time at which the angles are computed (in s/M_sun)
		q_range				Tuple of values for the range in which to draw the q values. If a single value, q is fixed
		chi1_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 1. If a single value, chi1 is fixed
		chi2_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 2. If a single value, chi2 is fixed
		theta1_range		Tuple of values for the range in which to draw the angles between spin 1 and L. If a single value, theta1 is fixed
		theta2_range		Tuple of values for the range in which to draw the angles between spin 2 and L. If a single value, theta2 is fixed
		delta_phi_range		Tuple of values for the range in which to draw the angles between the in-plane components of the spins. If a single value, delta_phi_range is fixed
	"""
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")
	if not isinstance(filename, str):
		raise TypeError("filename is "+str(type(filename))+"! Expected to be a string.")

	range_list = [q_range, chi1_range, chi2_range, theta1_range, theta2_range, delta_phi_range]

	time_grid = np.linspace(-np.abs(tau_min), 0., N_grid)
		#initializing file. If file is full, it is assumed to have the proper time grid
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file ", filename, " created")
		time_header = np.concatenate((np.zeros((6,)), time_grid, time_grid) )[None,:]
		np.savetxt(filebuff, time_header, header = "#Alpha, Beta dataset" +"\n# row: params (None,6) | alpha (None,"+str(N_grid)+")| beta (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | tau_min ="+str(tau_min)+" | q_range = "+str(q_range)+" | chi1_range = "+str(chi1_range)+" | chi2_range = "+str(chi2_range)+" | theta1_range = "+str(theta1_range)+" | theta2_range = "+str(theta2_range)+" | delta_phi_range = "+str(delta_phi_range), newline = '\n')
	else:
		filebuff = open(filename,'a')
	#computing an approximate r_0 as a function of tau_min
	M_sun = 4.93e-6
	r_0 = 2.5 * np.power(tau_min/M_sun, .25) #look eq. 4.26 Maggiore
	
	#deal with the case in which ranges are not tuples
	for i, r in enumerate(range_list):
		if not isinstance(r,tuple):
			if isinstance(r, float):
				range_list[i] = (r,r)
			else:
				raise RuntimeError("Wrong type of limit given: expected tuple or float!")

	#creating limits for random draws
	lower_limits = [r[0] for r in range_list]	
	upper_limits = [r[1] for r in range_list]	
	
	b_size = 2 #batch size at which angles are stored before being saved
	count = 0 #keep track of how many angles were generated
	while True:
		if N_angles- count > b_size:
			N = b_size
		elif N_angles - count > 0:
			N = N_angles -count
		else:
			break

		params = np.random.uniform(lower_limits, upper_limits, (N, len(range_list))) #(N,6) #parameters to generate the angles at
		count += N

		alpha, beta = get_alpha_beta(*params.T, r_0, time_grid, False)
		to_save = np.concatenate([params, alpha, beta], axis = 1)
		np.savetxt(filebuff, to_save) #saving the batch to file
		print("Generated angle: ", count)

	return





















	


