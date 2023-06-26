"""
Module precession_helper.py
===========================
	Module for training a ML model for fitting the precessing angles alpha, beta as a function of (theta1, theta2, deltaphi, chi1, chi2, q).
	Requires precession module (pip install precession) and tensorflow (pip install tensorflow)
"""

import numpy as np
import precession
import os
import sys
import warnings
import matplotlib.pyplot as plt
import scipy.signal
sys.path.insert(0,'../mlgw_v2') #this should be removed eventually
from GW_helper import *
from tqdm import tqdm
from PyEMD import EMD

sys.path.insert(0,'./IMRPhenomTPHM')
from run_IMR import get_IMRPhenomTPHM_angles

try:
	import silence_tensorflow.auto #awsome!!!! :)
except:
	pass
import tensorflow as tf

###############################################################################################################
###############################################################################################################
###############################################################################################################
def to_polar(s):
	"Given the 3 dimensionless components of a spin, it computes the spherical coordinates representation"
	s_norm = np.linalg.norm(s, axis =1) + 1e-10
	theta = np.arccos(s[:,2]/s_norm)
	phi = np.arctan2(s[:,1], s[:,0])

	return np.column_stack([s_norm, theta, phi]) #(N,3)
	
class angle_generator():
	"This class provides a generator of angles for the training of the NN."
	def __init__(self, N_batch, load_file, polar_coordinates = True):
		"Input the size of the time grid and the starting time from the angle generation. Ranges for the 6 dimensional inputs shall be provided by a (6,2) array"
		self.N_batch = N_batch	#how many different points are considered in each batch
		self.polar_coordinates = polar_coordinates

		self._initialise_dataset(load_file)
		return

	def get_output_dim(self):
		return 11
	
	def get_n_data(self):
		return self.params.shape[0]

	def _initialise_dataset(self, load_file):
		help(load_dataset)
		self.params, self.alpha, self.beta, _, self.times = load_dataset(load_file, N_data=None,
									N_entries = 3, N_grid = None, shuffle = False, n_params = 9)
		self.params = self.params[:,:8]
		
			#transforming parameters
		if self.polar_coordinates:
			s1 = self.params[:,[2,3,4]]
			s2 = self.params[:,[5,6,7]]
			#print(self.params[0,[2,3,4]])#DEBUG
			
			ids_s1 = np.where(np.sum(np.abs(s1), axis = 1) < 1e-9)[0]
			ids_s2 = np.where(np.sum(np.abs(s2), axis = 1) < 1e-9)[0]
			
			self.params[:,2:5] = to_polar(s1)
			self.params[:,5:8] = to_polar(s2)
			
				#regularizing the zeros
			if len(ids_s1)>0:
				self.params[ids_s1,2:5] = 0.
			if len(ids_s2)>0:
				self.params[ids_s2,5:8] = 0.
			
			#print(self.params[0,2]*np.sin(self.params[0,3])*np.cos(self.params[0,4]),
			#	 self.params[0,2]*np.sin(self.params[0,3])*np.sin(self.params[0,4]),
			#	 self.params[0,2]*np.cos(self.params[0,3])) #DEBUG
			#quit()
			#print(self.params[0,[2,3,4]]) #DEBUG
			
			#TODO: check it is fine
		
			######### BH1 dataset
			#removing data with spin 2
		#ids = np.where(np.all(self.params[:,[2,3,4]] != 0., axis = 1))[0]
		#self.params, self.alpha, self.beta = self.params[ids,:], self.alpha[ids,:], self.beta[ids,:]; print("removing BH2 spins")
		
		self.alpha = np.zeros(self.alpha.shape); print("##########Zeroing alpha")
		#self.beta = np.zeros(self.beta.shape); print("##########Zeroing beta")
		
		print("Dataset initialized N = ", self.params.shape[0])

		return

	def __call__(self):
		"Return a dataset of angles: each row is [t, f, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, alpha, beta]"
		while True:
			ids_WFs = np.random.choice(self.params.shape[0], size = (self.N_batch,), replace = True)
			ids_times = np.random.choice(self.times.shape[0], size = (self.N_batch,), replace = True)
			
			yield np.concatenate([self.times[ids_times][:,None], self.params[ids_WFs,:], 
						self.alpha[ids_WFs, ids_times][:,None], self.beta[ids_WFs, ids_times][:,None]], axis = 1) #(N_batch, 11)

	def get_alpha_beta(self, N, shuffle = False):
		"Returns N angles"
		if shuffle:
			ids = np.random.choice(self.params.shape[0], N, replace = False)
		else:
			ids = range(N)

		return self.params[ids,:], self.alpha[ids,:], self.beta[ids,:], self.times

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
	
		self.scaling_consts = tf.constant([1., 1.], dtype = tf.float32) #scaling constants for the loss function (set by hand, kind of)

		self.optimizer = None
		
			#building the network
		self._l_list = []
		self._l_list.append(tf.keras.layers.Dense(128*4, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)) #outputs: alpha, beta
		
		self.build(input_shape = (None, 9)) #This is required to specify the input shape of the model and to state which are the trainable paramters

	def call(self, inputs):
		"Inputs: [t, params (8,)]"
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
		Loss function: takes an input array X (N, 11) with values to test the model at and the angles at those points.
		Input should be tensorflow only.
		"""
		loss = tf.math.square(self.call(X[:,:9]) - X[:,9:]) #(N,2)
		loss = tf.math.divide(loss, self.scaling_consts) #(N,2)
		loss = tf.reduce_sum(loss, axis = 1) /X.shape[1] #(N,) 
		return loss

		#for jit_compil you must first install: pip install tf-nightly
	@tf.function(jit_compile=True) #very useful for speed up
	def grad_update(self, X):
		"Input should be tensorflow only."
		with tf.GradientTape() as g:
			g.watch(self.trainable_weights)
			loss = tf.reduce_sum(self.loss(X))/X.shape[0]

		gradients = g.gradient(loss, self.trainable_weights)
		self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

		return loss

	def fit(self, generator, N_epochs, learning_rate = 5e-4, out_path = './', plot_function = None, checkpoint_step = 20000, print_step = 10, validation_gen = None):
		if out_path is not None:
			if not out_path.endswith('/'):
				out_path = out_path +'/'
	
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) #custom optimizer
		epoch_0 = self.epoch

			#initializing the validation file
		if isinstance(validation_gen, angle_generator):
			val_params, val_alpha, val_beta, val_times = validation_gen.get_alpha_beta(validation_gen.get_n_data())

		tf_dataset = tf.data.Dataset.from_generator(
     			generator,
    			output_signature = tf.TensorSpec(shape=(None,generator.get_output_dim()), dtype=tf.float32)
					)#.prefetch(tf.data.experimental.AUTOTUNE) #good idea?? Probably yes
		
		n_epoch = -1
		for X in tf_dataset:
			#each row in X in dataset is as follows:
				#[t, f, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, alpha, beta]
			#X.shape = (N,11)
			n_epoch +=1
			if n_epoch >= N_epochs:
				break
			
				#gradient update
			loss = self.grad_update(X)

				#user communication, checkpoints and metric
			if n_epoch % print_step == 0: #saving history
				self.epoch = epoch_0 + n_epoch
				self.history.append((self.epoch, loss.numpy()))
				print(self.epoch, loss.numpy())
				if out_path is not None:
					self.save_everything(out_path)

			if (n_epoch)%checkpoint_step ==0 and n_epoch !=0:
				if out_path is not None:
					self.save_weights(out_path+"{}/{}".format(str(self.epoch), self.name)) #saving to an archive

				if plot_function is not None:
					plot_function(self, out_path+"{}".format(str(self.epoch)))

				if isinstance(validation_gen, angle_generator): #computing validation metric
					val_alpha_NN, val_beta_NN = self.get_alpha_beta(val_params, val_times)
					#TODO: compute the loss on the validation set
					
						#validation loss is abs(val-true/true)
					#loss_alpha = np.mean(np.abs(np.divide(val_alpha_NN- val_alpha, val_alpha)))
					#loss_beta = np.mean(np.abs(np.divide(val_beta_NN- val_beta, val_beta)))
					
						#validation loss is mse
					loss_alpha = np.mean(np.square(val_alpha_NN- val_alpha))
					loss_beta = np.mean(np.square(val_beta_NN- val_beta))
					
					self.metric.append((self.epoch, loss_alpha, loss_beta))
					print("\tMetric: {} {} {} ".format(self.metric[-1][0],self.metric[-1][1], self.metric[-1][2]))
					
		return self.history

	def load_everything(self, path):
		"Loads model and tries to read metric and loss"
		if not path.endswith('/'):
			path = path +'/'
		print("Loading from: ", path)
		self.load_weights(path+"{}".format(self.name))
		try:
			self.history = np.loadtxt(path+self.name+".loss").tolist()
			self.epoch = int(self.history[-1][0])
		except:
			self.epoch = 0
			pass

		try:
			self.metric = np.loadtxt(path+self.name+".metric").tolist()
		except:
			pass

		return

	def save_everything(self, path):
		"Save the model as well as the loss and the metric"
		if not path.endswith('/'):
			path = path +'/'
		#print("Saving to: ",path+"{}".format(self.name))
		self.save_weights(path+"{}".format(self.name)) #overwriting the newest
		np.savetxt(path+"{}.loss".format(self.name), np.array(self.history))
		np.savetxt(path+"{}.metric".format(self.name), np.array(self.metric))

		return

	def get_alpha_beta(self, X, times):
		"Computes alpha and beta with the NN on a custom time grid, given X = [[f, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z]]"
		if X.ndim == 1:
			X = X[None,:]

		N = X.shape[0]
		alpha = np.zeros((N , len(times)))
		beta = np.zeros((N, len(times)))

		for i in range(len(times)):
			t = np.repeat([times[i]], N)[:,None]
			X_tf = np.concatenate([t,X],axis =1)
			X_tf = tf.convert_to_tensor(X_tf, dtype=tf.float32)
			alpha_beta = self.call(X_tf) #(N,2)
			alpha[:,i] = alpha_beta[:,0] 
			beta[:,i] = alpha_beta[:,1] 

		return alpha, beta

def plot_validation_set(model, generator, N_sol, folder = ".", show = False):
	'Helper to plot the validation set with the predictions of the NN'
	#params, alpha, beta, _,  times = load_dataset(validation_file, N_data=N_sol, N_entries = 3, N_grid = None, shuffle = False, n_params = 9)
	params, alpha, beta,  times = generator.get_alpha_beta(N_sol, False)
	NN_alpha, NN_beta = model.get_alpha_beta(params[:,:8],times) #(N,D)

		#plotting
	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\alpha$")
	plt.plot(times, NN_alpha.T, c = 'r')
	plt.plot(times, alpha.T, c= 'b')
	plt.plot([],[], 'r', label='NN')
	plt.plot([],[], 'b', label='val')
	plt.legend(loc = 'upper left')
	if isinstance(folder, str):
		plt.savefig(folder+"/alpha.pdf", transparent =True)

	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\beta$")
	plt.plot(times, NN_beta.T, c= 'r')
	plt.plot(times, beta.T, c= 'b')
	plt.plot([],[], 'r', label='NN')
	plt.plot([],[], 'b', label='val')
	plt.legend(loc = 'upper left')
	if isinstance(folder, str):
		plt.savefig(folder+"/beta.pdf", transparent =True)


	if show:
		plt.show()
	else:
		plt.close('all')


###############################################################################################################
###############################################################################################################
###############################################################################################################
def get_random_chi(N, chi_range = (0.,0.8)):
	"""
get_random_chi
==============
	Extract a random chi value
	Inputs:
		N				Number of spins to extract
		chi_range		Range (min, max) for the magnitude of the spin
	Output:
		chi (N,3)	Extracted spins
	"""
	chi = np.random.uniform(chi_range[0], chi_range[1], (N,))
	chi_vec = np.random.normal(0,1, (N,3))
	chi_vec = (chi_vec / np.linalg.norm(chi_vec, axis = 1)) * chi
	return chi_vec

def set_effective_spins(m1, m2, chi1, chi2):
	"""
set_effective_spins
===================
	Given a generic spin configuration, it assigns the spins to a single BH. The inplane spin is assigned according to the spin parameter (https://arxiv.org/pdf/2012.02209.pdf), while the z component is assigned as chi_eff to the same BH that has an in-plane spin.
	Inputs:
		m1 ()/(N,)				mass of the first BH
		m2 ()/(N,)				mass of the second BH
		chi1 (3,)/(N,3)			dimensionless spin of the BH 1
		chi2 (3,)/(N,3)			dimensionless spin of the BH 2
	Outputs:
		chi1_eff (3,)/(N,3)		in-plane component of BH 1 spin after the spin approx is performed
		chi2_eff (3,)/(N,3)		in-plane component of BH 2 spin after the spin approx is performed
	"""
	#TODO: I should check the accuracy of this function
	if isinstance(m1,float):
		m1 = np.array([m1]) #(1,)
		m2 = np.array([m2]) #(1,)
		chi1 = np.array([chi1])#(1,3)
		chi2 = np.array([chi2])#(1,3)
		squeeze = True
	else:
		squeeze = False
	
	chi1_perp_eff, chi2_perp_eff = compute_S_effective(m1,m2, chi1[:,:2], chi2[:,:2]) #(N,2)
	
	chi_eff = (chi2[:,2]*m2 + chi1[:,2]*m1)/(m1+m2) #(N,)

	chi1_eff = np.column_stack([chi1_perp_eff[:,0], chi1_perp_eff[:,1], chi1[:,2]]) #(N,3)
	chi2_eff = np.column_stack([chi2_perp_eff[:,0], chi2_perp_eff[:,1], chi2[:,2]]) #(N,3)
	
		#taking care of chi_eff
	ids_ = np.where(np.sum(np.abs(chi2_perp_eff), axis =1) == 0) #(N,)
	ids_bool = np.zeros((len(chi_eff), ), dtype = bool) #all False
	ids_bool[ids_] = True #indices in which chi_eff is given to BH1
	
	#print("non std chieff")
	#TODO: think about chi_eff and a better way of setting it...
	chi1_eff[ids_bool,2] = chi_eff[ids_bool]#*((m1+m2)/m1)
	chi2_eff[~ids_bool,2] = chi_eff[~ids_bool]#*((m1+m2)/m2)
	chi1_eff[~ids_bool,2] = 0.
	chi2_eff[ids_bool,2] = 0.

	if squeeze:
		return np.squeeze(chi1_eff), np.squeeze(chi2_eff)
	return chi1_eff, chi2_eff


def compute_S_effective(m1,m2, chi1_perp, chi2_perp):
	"""
compute_S_effective
===================
	It computes the 2D effective spin parameter, given masses and in plane dimensionless components of the spins.
	The spin parameter is defined in https://arxiv.org/pdf/2012.02209.pdf
	Inputs:
		m1 ()/(N,)					mass of the first BH
		m2 ()/(N,)					mass of the second BH
		chi1_perp (2,)/(N,2)		in-plane dimensionless spin of the first BH
		chi2_perp (2,)/(N,2)		in-plane dimensionless spin of the second BH
	Outputs:
		chi1_perp_eff (2,)/(N,2)		in-plane component of BH 1 dimensionless spin after the spin approx is performed
		chi2_perp_eff (2,)/(N,2)		in-plane component of BH 2 dimensionless spin after the spin approx is performed
	"""
	#TODO: I should check the accuracy of this function
	if isinstance(m1,float):
		m1 = np.array([m1])
		m2 = np.array([m2])
		S1_perp = (m1**2*np.array([chi1_perp]).T).T #(1,3)
		S2_perp = (m2**2*np.array([chi2_perp]).T).T #(1,3)
		squeeze = True
	else:
		S1_perp = (m1**2 * chi1_perp.T).T #(1,3)
		S2_perp = (m2**2 * chi2_perp.T).T #(1,3)
		squeeze = False
	
	ids_to_invert = np.where(m2>m1)
	m1[ids_to_invert], m2[ids_to_invert] = m2[ids_to_invert], m1[ids_to_invert]
	S1_perp[ids_to_invert,:], S2_perp[ids_to_invert,:] = S2_perp[ids_to_invert,:], S1_perp[ids_to_invert,:] #(N,2)

	S_perp = S1_perp + S2_perp
	
	S1_perp_norm= np.linalg.norm(S1_perp, axis =1) #(N,)
	S2_perp_norm= np.linalg.norm(S2_perp, axis =1) #(N,)
	
	ids_S1 = np.where(S1_perp_norm >= S2_perp_norm)[0]
	ids_S2 = np.where(S1_perp_norm < S2_perp_norm)[0]
	
	chi1_perp_eff = np.zeros(S1_perp.shape) #(N,2)
	chi2_perp_eff = np.zeros(S1_perp.shape) #(N,2)
	
	if ids_S1.shape != (0,):
		chi1_perp_eff[ids_S1,:] = (S_perp[ids_S1,:].T / (np.square(m1[ids_S1])+S2_perp_norm[ids_S1]) ).T
	if ids_S2.shape != (0,):
		chi2_perp_eff[ids_S2,:] = (S_perp[ids_S2,:].T / (np.square(m2[ids_S2])+S1_perp_norm[ids_S2]) ).T
	
	if squeeze:
		return np.squeeze(chi1_perp_eff), np.squeeze(chi2_perp_eff)
	return chi1_perp_eff, chi2_perp_eff

def create_dataset_alpha_beta_gamma(N_angles, filename, N_grid, f_range, q_range, chi1_range= (0.,1.), chi2_range = (0.,1.), single_spin = False, delta_T = 1e-4, alpha =0.95, verbose = False ):
	"""
create_dataset_alpha_beta
=========================
	Creates a dataset for the Euler angles alpha, beta, gamma.
	The dataset consist in parameter vector (q, chi1, chi2, theta1, theta2, delta_phi) associated to two vectors alpha and beta.
	User must specify a time grid at which the angles are evaluated at.
	More specifically, data are stored in 3 vectors:
		param_vector	vector holding source parameters (f_ref, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
		t_0				scaling factor for the grid (i.e. time to merger from f_ref)
		alpha_vector	vector holding alpha angle for each source evaluated at some N_grid equally spaced points
		beta_vector		vector holding beta angle for each source evaluated at some N_grid equally spaced points
		gamma_vector	vector holding gamam angle for each source evaluated at some N_grid equally spaced points
	The values of parameters are randomly drawn within the user given constraints.
	Dataset is saved to file, given in filename and can be loaded with load_angle_dataset.
	Inputs:
		N_angles			Number of angles to include in the dataset
		filename			Name of the file to save the dataset at
		N_grid				Number of grid points
		f_range				Tuple of values for the range in which to draw the f_start = f_ref values. If a single value, f_ref is fixed
		q_range				Tuple of values for the range in which to draw the q values. If a single value, q is fixed
		chi1_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 1. If a single value, chi1 is fixed
		chi2_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 2. If a single value, chi2 is fixed
		single_spin			Whether to use the single_spin approx. In this case, all the spin is assigned to a single BH
		delta_T				Integration step for the angles
		alpha				distorsion parameter (for accumulating more grid points around the merger)
		verbose				Whether to print the output to screen
	"""
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")
	if not isinstance(filename, str):
		raise TypeError("filename is "+str(type(filename))+"! Expected to be a string.")

	range_list = [f_range, q_range, chi1_range, chi2_range]

	M = 20. #total mass (standard)

	time_grid = np.linspace(np.power(1.,alpha), 1e-6, N_grid) #dimensionless t/t[0]
	time_grid = -np.power(time_grid,1./alpha)
	time_grid[-1] = 0.
	
		#initializing file. If file is not empty, it is assumed to have the proper time grid
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file ", filename, " created")
		time_header = np.concatenate((np.zeros((9,)), time_grid, time_grid, time_grid) )[None,:]
		np.savetxt(filebuff, time_header,
			header = "#Alpha, Beta , Gamma dataset" +"\n# row: params (None,8) | t_0 (None,1)| alpha (None,"+str(N_grid)+")| beta (None,"+str(N_grid)+")| gamma (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | f_range ="+str(f_range)+" | q_range = "+str(q_range)+" | chi1_range = "+str(chi1_range)+" | chi2_range = "+str(chi2_range),
			newline = '\n')
	else:
		filebuff = open(filename, 'a')
	
	#deal with the case in which ranges are not tuples
	for i, r in enumerate(range_list):
		if not isinstance(r,tuple):
			if isinstance(r, float):
				range_list[i] = (r,r)
			else:
				raise RuntimeError("Wrong type of limit given: expected tuple or float!")
	
	b_size = 10 #batch size at which angles are stored before being saved
	count = 0 #keep track of how many angles were generated

	M_tot = 20.

	for i in tqdm(range(N_angles)):
		f = np.random.uniform(range_list[0][0], range_list[0][1])
		q = np.random.uniform(range_list[1][0], range_list[1][1])
		chi1 = np.random.uniform(range_list[2][0], range_list[2][1])
		chi2 = np.random.uniform(range_list[3][0], range_list[3][1])
		
		m1, m2 = q*M/(1+q), M/(1+q)
		
			#extracting random components
		chi1_vec = np.random.normal(0,1, (3,))
		chi1_vec = (chi1_vec / np.linalg.norm(chi1_vec)) * chi1 

		chi2_vec = np.random.normal(0,1, (3,))
		chi2_vec = (chi2_vec / np.linalg.norm(chi2_vec)) * chi2
		
		#print("fixed spins and q!!"); chi1_vec = [0.3,-0.1,-0.4]; chi2_vec = [0.4,-0.5,-0.2] #DEBUG
		if single_spin:
			chi1_vec, chi2_vec = set_effective_spins(m1, m2, chi1_vec, chi2_vec)
			if np.all(chi2_vec!=0.): #BH1 dataset
				continue

		t, alpha_, beta_, gamma_ = get_IMRPhenomTPHM_angles(m1, m2, *chi1_vec, *chi2_vec, f, delta_T)

		t_0 = -t[0]
		
		alpha = np.interp(time_grid, t/t_0, alpha_)
		beta = np.interp(time_grid, t/t_0, beta_)
		gamma = np.interp(time_grid, t/t_0, gamma_)

		to_save = np.concatenate([[f], [q], chi1_vec, chi2_vec, [t_0], alpha, beta, gamma])[None,:]
		
		np.savetxt(filebuff, to_save)
		
		if verbose and (i+1)%100 == 0:
			print("Generated angle ", i+1)

	return

def create_dataset_alphagamma_emdbeta(N_angles, filename, N_grid, f_range, q_range, chi1_range= (0.,1.), chi2_range = (0.,1.), single_spin = False, delta_T = 1e-4, alpha =0.95, verbose = False ):
	"""
create_dataset_alpha_beta
=========================
	Creates a dataset for the quantities:
		alpha+gamma, trend beta, envelope beta and angle beta
	The dataset consist in parameter vector (q, chi1, chi2, theta1, theta2, delta_phi) associated to the interesting quantities.
	User must specify a time grid at which the angles are evaluated at.
	More specifically, data are stored in 3 vectors:
		param_vector	vector holding source parameters (f_ref, q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
		t_0				scaling factor for the grid (i.e. time to merger from f_ref)
		alphagamma_vector		vector holding alpha+gamma for each source evaluated at some N_grid equally spaced points
		betatrend_vector		vector holding trend beta for each source evaluated at some N_grid equally spaced points
		betaenvelope_vector		vector holding envelope beta for each source evaluated at some N_grid equally spaced points
		betaangle_vector		vector holding angle beta for each source evaluated at some N_grid equally spaced points
	The values of parameters are randomly drawn within the user given constraints.
	Dataset is saved to file, given in filename and can be loaded with load_angle_dataset.
	Inputs:
		N_angles			Number of angles to include in the dataset
		filename			Name of the file to save the dataset at
		N_grid				Number of grid points
		f_range				Tuple of values for the range in which to draw the f_start = f_ref values. If a single value, f_ref is fixed
		q_range				Tuple of values for the range in which to draw the q values. If a single value, q is fixed
		chi1_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 1. If a single value, chi1 is fixed
		chi2_range			Tuple of values for the range in which to draw the dimensionless spin values of BH 2. If a single value, chi2 is fixed
		single_spin			Whether to use the single_spin approx. In this case, all the spin is assigned to a single BH
		delta_T				Integration step for the angles
		alpha				distorsion parameter (for accumulating more grid points around the merger)
		verbose				Whether to print the output to screen
	"""
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")
	if not isinstance(filename, str):
		raise TypeError("filename is "+str(type(filename))+"! Expected to be a string.")

	range_list = [f_range, q_range, chi1_range, chi2_range]

	M = 20. #total mass (standard)

	time_grid = np.linspace(np.power(1.,alpha), 1e-6, N_grid) #dimensionless t/t[0]
	time_grid = -np.power(time_grid,1./alpha)
	time_grid[-1] = 0.
	
		#initializing file. If file is not empty, it is assumed to have the proper time grid
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file ", filename, " created")
		time_header = np.concatenate((np.zeros((9,)), time_grid, time_grid, time_grid, time_grid) )[None,:]
		np.savetxt(filebuff, time_header,
			header = "#AlphaGamma, BetaTrend , BetaEnvelope, BetaAngle dataset" +"\n# row: params (None,8) | t_0 (None,1)| alpha (None,"+str(N_grid)+")| betatrend (None,"+str(N_grid)+")| betaenvelope (None,"+str(N_grid)+")| betaangle (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | f_range ="+str(f_range)+" | q_range = "+str(q_range)+" | chi1_range = "+str(chi1_range)+" | chi2_range = "+str(chi2_range),
			newline = '\n')
	else:
		filebuff = open(filename, 'a')
	
	#deal with the case in which ranges are not tuples
	for i, r in enumerate(range_list):
		if not isinstance(r,tuple):
			if isinstance(r, float):
				range_list[i] = (r,r)
			else:
				raise RuntimeError("Wrong type of limit given: expected tuple or float!")

	M_tot = 20.

	for i in tqdm(range(N_angles)):
		f = np.random.uniform(range_list[0][0], range_list[0][1])
		q = np.random.uniform(range_list[1][0], range_list[1][1])
		chi1 = np.random.uniform(range_list[2][0], range_list[2][1])
		chi2 = np.random.uniform(range_list[3][0], range_list[3][1])
		
		m1, m2 = q*M/(1+q), M/(1+q)
		
			#extracting random components (you can do better)
		chi1_vec = np.random.normal(0,1, (3,))
		chi1_vec = (chi1_vec / np.linalg.norm(chi1_vec)) * chi1 

		chi2_vec = np.random.normal(0,1, (3,))
		chi2_vec = (chi2_vec / np.linalg.norm(chi2_vec)) * chi2
		
		#print("fixed spins and q!!"); chi1_vec = [0.3,-0.1,-0.4]; chi2_vec = [0.4,-0.5,-0.2] #DEBUG
		if single_spin:
			chi1_vec, chi2_vec = set_effective_spins(m1, m2, chi1_vec, chi2_vec)
			if np.all(chi2_vec!=0.): #BH1 dataset
				continue

		t, alpha_, beta_, gamma_ = get_IMRPhenomTPHM_angles(m1, m2, *chi1_vec, *chi2_vec, f, delta_T)
		t_0 = -t[0]
		alpha_ = np.interp(time_grid, t/t_0, alpha_)
		beta_ = np.interp(time_grid, t/t_0, beta_)
		gamma_ = np.interp(time_grid, t/t_0, gamma_)

			#EMD for beta
		args_dict = {'extrema_detection':'parabol', 'energy_ratio_thr': 1e-14, 'std_thr':1e-14, 'total_power_thr':1e-15, 'svar_thr':1e-13}
		emd_model = EMD(splinekind = 'linear', kwargs = args_dict)
		imf_beta = emd_model(beta_).T #(D,2)
		
		if imf_beta.shape[1]>2: #more than 2 components:just keeping the trend
			if False:
				plt.figure()
				plt.plot(beta_)
				fig, ax = plt.subplots(imf_beta.shape[1],1)
				for j, ax_ in enumerate(ax):
					ax_.plot(imf_beta[:,j])
				plt.show()
			imf_beta = imf_beta[:,[-1]] #keeping just the trend, it is hopeless to keep the rest... :)
	
		if imf_beta.shape[1] == 1:
			envelope = np.zeros( (imf_beta.shape[0],))
			hil = np.zeros( (imf_beta.shape[0],), dtype = complex)
		else:
			max_sp, min_sp, _, _ = emd_model.extract_max_min_spline(np.array(range(imf_beta[:,0].shape[0])), imf_beta[:,0])
			envelope = (max_sp - min_sp)/2.
			hil = scipy.signal.hilbert(imf_beta[:,0]/envelope)

			#saving the quantities		
		alphagamma = alpha_ + gamma_
		betatrend = imf_beta[:,-1]
		betaenvelope = envelope
		betaangle = np.unwrap(np.angle(hil))
		
		if False: #plot debug
			print("f0 {}".format(f))
			fig, ax = plt.subplots(5,1, figsize = (20,20))
			ax[0].plot(time_grid, alphagamma, label = 'alphagamma')
			ax[1].plot(time_grid, beta_, label = 'beta')
			ax[1].plot(time_grid, betatrend+ envelope*np.cos(betaangle), label = 'beta reconstructed')
			ax[2].plot(time_grid, betatrend, label = 'betatrend')
			ax[3].plot(time_grid, betaenvelope, label = 'betaenvelope')
			ax[4].plot(time_grid, betaangle, label = 'betaangle')
			for ax_ in ax:
				ax_.legend()
			plt.tight_layout()
			plt.show()

		try:
			to_save = np.concatenate([[f], [q], chi1_vec, chi2_vec, [t_0], alphagamma, betatrend, betaenvelope, betaangle])[None,:]
		except ValueError:
			print("Very obscure and rare error: skipping this data.\n",betatrend, betaenvelope, betaangle)
			continue
		
		np.savetxt(filebuff, to_save)
		
		if verbose and (i+1)%100 == 0:
			print("Generated angle ", i+1)

	return




















	


