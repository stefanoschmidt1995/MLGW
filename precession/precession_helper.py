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
sys.path.insert(0,'../mlgw_v2') #this should be removed eventually
from GW_helper import *
from tf_stats_loss import tf_wasserstein_loss

try:
	import silence_tensorflow.auto #awsome!!!! :)
except:
	pass
import tensorflow as tf

#FIXME: set a unique return signature for the two get_alpha_beta
#TODO: find a good way to smoothen the function
#TODO: replace the robustness check

########## spline helpers
class smoothener():
	"Class which extract from a fast oscillating function the trend and the oscillation amplitude"

	def __init__(self, x,y):
		"Initialize the class with the function y(x) and computes the smoothener"
			#initializing the lists
		max_peaks, props = scipy.signal.find_peaks(y)
		min_peaks, props = scipy.signal.find_peaks(-y)
		min_peak_density = .2
		
		if len(max_peaks)/(np.max(x)-np.min(x)) >= min_peak_density and len(min_peaks)/(np.max(x)-np.min(x)) >= min_peak_density:

			min_val = np.maximum(np.max(max_peaks), np.max(min_peaks))
			step = np.minimum(np.min(np.diff(min_peaks)), np.min(np.diff(max_peaks)))
			continuation = np.arange(min_val, len(x), int(step), dtype = int)

			if len(continuation) > 0: continuation = continuation[1:]
			if len(continuation) <= 2: continuation = None

		else:
			max_peaks = range(len(x))
			min_peaks = range(len(x))
			continuation = None
		
		self.smoothener = self._build_smoothener(max_peaks, min_peaks, continuation, x, y) #function
		
		return
		
	def _build_smoothener(self,max_peaks , min_peaks, continuation, x,y):
		"Builds the smoothener"
		spline_max = scipy.interpolate.CubicSpline(x[max_peaks], y[max_peaks], extrapolate = True, bc_type = 'natural')
		spline_min = scipy.interpolate.CubicSpline(x[min_peaks], y[min_peaks], extrapolate = True, bc_type = 'natural')
		if continuation is not None:
			continuation_interp = scipy.interpolate.interp1d(x[continuation], y[continuation], fill_value = 'extrapolate')
			boundary = x[continuation[0]]		
		else:
			boundary = x[-1] +1

		def smooth(x_):
			y_ = np.zeros(x_.shape)
			use_spline = np.where(x_<=boundary)[0]
			y_[use_spline] = (spline_max(x_[use_spline])+spline_min(x_[use_spline]))/2.
			use_continuation = np.where(x_>boundary)[0]
			if len(use_continuation) >  0:
				y_[use_continuation] = continuation_interp(x_[use_continuation])
			return y_
		
		return smooth
	
	def __call__(self, y_):
		return self.smoothener(y_) #(D,)
		

def compute_spline_peaks(x,y):
	print("Dealing with {} points".format(y.shape[0]))
	max_list = []
	min_list = []
	mean_list = []
	min_peak_density = .2
	for i in range(y.shape[0]):
		max_peaks, props = scipy.signal.find_peaks(y[i,:])
		min_peaks, props = scipy.signal.find_peaks(-y[i,:])

		if len(max_peaks)/(np.max(x)-np.min(x)) >= min_peak_density and len(min_peaks)/(np.max(x)-np.min(x)) >= min_peak_density:

			min_val = np.maximum(np.max(max_peaks), np.max(min_peaks))
			step = np.minimum(np.min(np.diff(min_peaks)), np.min(np.diff(max_peaks)))
			continuation = np.arange(min_val, y.shape[1], int(step), dtype = int)
			continuation_min = np.arange(np.max(min_peaks), y.shape[1], int(np.min(np.diff(min_peaks))), dtype = int)
			print("continuation", continuation)

			if len(continuation) > 0: continuation = continuation[1:]

			#continuation = [y.shape[1]-1]
			max_peaks = np.concatenate([max_peaks, continuation])
			min_peaks = np.concatenate([min_peaks, continuation])
			
			max_list.append(scipy.interpolate.interp1d(x[max_peaks], y[i, max_peaks], fill_value = 'extrapolate'))
			min_list.append(scipy.interpolate.interp1d(x[min_peaks], y[i, min_peaks], fill_value = 'extrapolate'))
			#max_list.append(scipy.interpolate.CubicSpline(x[max_peaks], y[i, max_peaks], extrapolate = True, bc_type = 'natural'))
			#min_list.append(scipy.interpolate.CubicSpline(x[min_peaks], y[i, min_peaks], extrapolate = True, bc_type = 'natural'))
			#max_list.append(scipy.interpolate.UnivariateSpline(x[max_peaks], y[i, max_peaks], ext = 0))
			#min_list.append(scipy.interpolate.UnivariateSpline(x[min_peaks], y[i, min_peaks], ext = 0))
		else:
			#warnings.warn("As not enough peaks were located, the curve will be interpolated using all the points available")
			max_list.append(scipy.interpolate.CubicSpline(x, y[i, :], extrapolate = True, bc_type = 'natural'))
			min_list.append(scipy.interpolate.CubicSpline(x, y[i, :], extrapolate = True, bc_type = 'natural'))
		
	return min_list, max_list
	
def get_spline_mean(x,y, f_minmax = False):
	f_min, f_max = compute_spline_peaks(x,y)
	def mean(times, i =None):
		if i is not None:
			return (f_max[i](times)+f_min[i](times))/2.
		res = np.zeros((len(f_min),len(times)))
		for i in range((len(f_min))):
			res[i,:] = (f_max[i](times)+f_min[i](times))/2.
		return res #(N,D)

	if f_minmax:
		return mean, f_min, f_max	
	return mean
	
def get_grad_mean(x,y):
	interp_list = []
	for i in range(y.shape[0]):
		grad = np.gradient(y[i,:], x)
		max_peaks, props = scipy.signal.find_peaks(grad)
		min_peaks, props = scipy.signal.find_peaks(-grad)
		peaks = np.concatenate([max_peaks, min_peaks]) #indices of the grad peaks
		peaks = np.sort(peaks)
	
			#setting a minum density of peaks
		min_peak_density = 0
		
		if len(peaks)/(max(x) - min(x)) > min_peak_density:
			interp_list.append(scipy.interpolate.CubicSpline(x[peaks], y[i, peaks], extrapolate = True))
		else:
			interp_list.append(scipy.interpolate.CubicSpline(x, y[i, :], extrapolate = True))
		
		plt.title("Points")
		plt.plot(x, y[i,:])
		plt.plot(x, 0.01*grad)
		plt.plot(x[max_peaks], y[i,max_peaks],'o', ms =5, c = 'r')
		plt.plot(x[min_peaks], y[i,min_peaks],'o', ms =5, c = 'k')

	return interp_list
		
	
def compute_spline_extrema(x,y, get_spline = False):
	maxima = scipy.signal.argrelextrema(y, np.greater, axis = 1) #(N,M)
	minima = scipy.signal.argrelextrema(y, np.less, axis = 1) #(N,M')

	max_list = []
	min_list = []
	spline_list = []
	for i in range(y.shape[0]):
		ids_0_max = np.where(maxima[0]==i)[0]
		ids_0_min = np.where(minima[0]==i)[0]

		max_list.append(scipy.interpolate.CubicSpline(x[maxima[1][ids_0_max]], y[i, maxima[1][ids_0_max]], extrapolate = True))
		min_list.append(scipy.interpolate.CubicSpline(x[minima[1][ids_0_min]], y[i, minima[1][ids_0_min]], extrapolate = True))
		if get_spline:
			spline_list.append(scipy.interpolate.CubicSpline(x, y[i,:], extrapolate = True))
	if get_spline:
		return min_list, max_list, spline_list
	return min_list, max_list
	
#######################

def get_alpha_beta_M(M, q, chi1, chi2, theta1, theta2, delta_phi, f_ref = 20., smooth_oscillation = False, verbose = False):
	"""
get_alpha_beta
==============
	Returns angles alpha and beta by solving PN equations for spins. Uses module precession.
	Angles are evaluated on a user-given time grid (units: s/M_sun) s.t. the 0 of time is at separation r = M_tot.
	Inputs:
		M (N,)				total mass
		q (N,)				mass ratio (>1)
		chi1 (N,)			dimensionless spin magnitude of BH 1 (in [0,1])
		chi1 (N,)			dimensionless spin magnitude of BH 2 (in [0,1])
		theta1 (N,)			angle between spin 1 and L
		theta2 (N,)			angle between spin 2 and L
		delta_phi (N,)		angle between in plane projection of the spins
		f_ref				frequency at which the orbital parameters refer to (and at which the computation starts)
		verbose 			whether to suppress the output of precession package
	Outputs:
		times (D,)		times at which alpha, beta are evaluated (units s)
		alpha (N,D)		alpha angle evaluated at times
		beta (N,D)		beta angle evaluated at times (if not smooth_oscillation)
	"""
	M_sun = 4.93e-6
	
	if isinstance(q,np.ndarray):
		q = q[0]
		chi1 = chi1[0]
		chi2 = chi2[0]
		theta1 = theta1[0]
		theta2 = theta2[0]
		delta_phi = delta_phi[0]

		#initializing vectors
	if not verbose:
		devnull = open(os.devnull, "w")
		old_stdout = sys.stdout
		sys.stdout = devnull
	else:
		old_stdout = sys.stdout

		#computing alpha, beta
	q_ = 1./q #using conventions of precession package
	m1=M/(1.+q_) # Primary mass
	m2=q*M/(1.+q_) # Secondary mass
	S1=chi1*m1**2 # Primary spin magnitude
	S2=chi2*m2**2 # Secondary spin magnitude
	r_0 = precession.ftor(f_ref,M)
	print(r_0)
	
	xi,J, S = precession.from_the_angles(theta1,theta2, delta_phi, q_, S1,S2, r_0)
		
	J_vec,L_vec,S1_vec,S2_vec,S_vec = precession.Jframe_projection(xi, S, J, q_, S1, S2, r_0) #initial conditions given angles

	r_f = 1.* M #final separation: time grid is s.t. t = 0 when r = r_f
	sep = np.linspace(r_0, r_f, 5000)

	#J = precession.evolve_J(xi,J, sep, q_, S1,S2) #precession avg evolution

	Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, t = precession.orbit_vectors(*L_vec, *S1_vec, *S2_vec, sep, q_, time = True) #time evolution of L, S1, S2
	L = np.sqrt(Lx**2 + Ly**2 + Lz**2)

	alpha = np.unwrap(np.arctan2(Ly,Lx))
	beta = np.arccos(Lz/L)

	t_out = (t-t[-1])#*M_sun*Mtot #output grid
		
	if not verbose:
		sys.stdout = old_stdout
		devnull.close()

	return t_out, alpha, beta

def get_alpha_beta(q, chi1, chi2, theta1, theta2, delta_phi, times, f_ref = 20., smooth_oscillation = False, verbose = False):
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
		times (D,)			times at which alpha, beta are evaluated (units s/M_sun)
		f_ref				frequency at which the orbital parameters refer to (and at which the computation starts)
		smooth_oscillation	whether to smooth the oscillation and return the average part and the residuals
		verbose 			whether to suppress the output of precession package
	Outputs:
		alpha (N,D)		alpha angle evaluated at times
		beta (N,D)		beta angle evaluated at times (if not smooth_oscillation)
		beta (N,D,3)	[mean of beta angles, amplitude of the oscillating part, phase of the oscillating part] (if smooth_oscillation)
	"""
	#have a loook at precession.evolve_angles: it does exactly what we want..
	#https://github.com/dgerosa/precession/blob/precession_v1/precession/precession.py#L3043
	
	M_sun = 4.93e-6
	t_min = np.max(np.abs(times))
	r_0 = 2. * np.power(t_min/M_sun, .25) #starting point for the r integration #look eq. 4.26 Maggiore #uglyyyyy
	#print(f_ref, precession.rtof(r_0, 1.))
	#print(f_ref)
	r_0 = precession.ftor(f_ref,1)
		
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
	if smooth_oscillation:
		t_cutoff = -0.1 #shall I insert a cutoff here?
		beta = np.zeros((q.shape[0],times.shape[0], 3))
	else:
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
		if np.abs(delta_phi[i]) < 1e-6:#delta Phi cannot be zero(for some reason)
			delta_phi[i] = 1e-6
			
		xi,J, S = precession.from_the_angles(theta1[i],theta2[i], delta_phi[i], q_, S1,S2, r_0) 

		J_vec,L_vec,S1_vec,S2_vec,S_vec = precession.Jframe_projection(xi, S, J, q_, S1, S2, r_0) #initial conditions given angles

		r_f = 1.* M #final separation: time grid is s.t. t = 0 when r = r_f
		sep = np.linspace(r_0, r_f, 5000)

		Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, t = precession.orbit_vectors(*L_vec, *S1_vec, *S2_vec, sep, q_, time = True) #time evolution of L, S1, S2
		L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
		
		print(Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, t)
		#quit()
		
			#cos(beta(t)) = L(t).(0,0,1) #this is how I currently do!
			#cos(beta(t)) = L(t).L_vect #this is the convention that I want
		temp_alpha = np.unwrap(np.arctan2(Ly,Lx))
		temp_beta = np.arccos(Lz/L)
		
			#computing beta in the other reference frame
		L_0 = L_vec /np.linalg.norm(L_vec)
		L_t = (np.column_stack([Lx,Ly,Lz]).T/L).T #(D,3)
		temp_beta = np.einsum('ij,j->i', L_t, L_0) #cos beta
		print(L_t.shape, temp_beta.shape, L_vec)
		
		temp_beta = np.arccos(temp_beta)


		t_out = (t-t[-1])*M_sun #output grid
		print("\n\nTimes!! ",t_out[0], times[0])
		ids = np.where(t_out > np.min(times))[0]
		t_out = t_out[ids]
		temp_alpha = temp_alpha[ids]
		temp_beta = temp_beta[ids]
		
		alpha[i,:] = np.interp(times, t_out, temp_alpha)
		if not smooth_oscillation:
			#plt.plot(t_out,temp_beta)
			#plt.show()
			beta[i,:] = np.interp(times, t_out, temp_beta)
		if smooth_oscillation:
			#mean, f_min, f_max = get_spline_mean(t_out, temp_beta[None,:], f_minmax = True)
			s = smoothener(t_out, temp_beta)
			#beta[i,:,0] = mean(times) #avg beta
			beta[i,:,0] = s(times)
			#beta[i,:,1] = np.interp(times, t_out, temp_beta) - mean(times) #residuals of beta

				#dealing with amplitude and phase
			residual = (temp_beta - s(t_out))
			if np.mean(np.abs(residual))< 0.001:
				residual[:] = 0
			id_cutoff = np.where(t_out>t_cutoff)[0]
			not_id_cutoff = np.where(t_out<=t_cutoff)[0]
			residual[id_cutoff] = 0.
			
			
			m_list, M_list = compute_spline_peaks(t_out, residual[None,:])
			amp = lambda t: (M_list[0](t) - m_list[0](t))/2.
			beta[i,:,1] = amp(times) #amplitude
			temp_ph = residual / (amp(t_out)+1e-30)
			temp_ph[id_cutoff] = 0.
			beta[i,:,2] = np.interp(times, t_out, temp_ph) #phase
			beta[i,np.where(np.abs(beta[i,:,2])>1)[0],2] = np.sign(beta[i,np.where(np.abs(beta[i,:,2])>1)[0],2])
			
				#plotting
			if False:# np.max(np.abs(temp_beta-s(t_out))) > 2: #DEBUG
				plt.figure()
				plt.title("Alpha")
				plt.plot(times,alpha[i,:])

				plt.figure()			
				plt.title("Mean maxmin")
				plt.plot(times,beta[i,:,0])
				plt.plot(t_out,temp_beta)
				
				#plt.figure()
				#plt.title("Mean grad")
				#plt.plot(t_out, temp_beta)
				#plt.plot(t_out, mean_grad[0](t_out))
				
				#plt.figure()
				#plt.title("Gradient")
				#plt.plot(t_out,np.gradient(temp_beta, t_out))
				#plt.ylim([-0.6,0.6])
				
				plt.figure()
				plt.title("Amplitude")
				#plt.plot(t_out, amp(t_out))
				plt.plot(times, beta[i,:,1])
				plt.plot(t_out,np.squeeze(temp_beta - s(t_out) ))
				
				#plt.figure()
				#plt.plot(times,beta[i,:,1])
				
				plt.figure()
				plt.title("ph")
				plt.plot(times,beta[i,:,2])
				plt.show()
	
	if not verbose:
		sys.stdout = old_stdout
		devnull.close()

	if squeeze:
		return np.squeeze(alpha), np.squeeze(beta)

	return alpha, beta


class angle_generator():
	"This class provides a generator of angles for the training of the NN."
	def __init__(self, t_min, N_times, ranges, N_batch = 100, replace_step = 1, load_file = None, smooth_oscillation = True):
		"Input the size of the time grid and the starting time from the angle generation. Ranges for the 6 dimensional inputs shall be provided by a (6,2) array"
		self.t_min = np.abs(t_min)
		self.N_times = N_times
		self.N_batch = N_batch
		self.smooth_oscillation = smooth_oscillation
		self.ranges = ranges #(6,2)
		self.replace_step = replace_step #number of iteration before computing a new element
		if not (isinstance(self.replace_step, int) or (self.replace_step is None)):
			try:
				self.replace_step = int(self.replace_step)
			except ValueError:
				raise ValueError("Wrong format for replace_step: expected int but got {} instead".format(type(self.replace_step)))
		self.dataset = np.zeros((N_batch*N_times, 9 + int(smooth_oscillation)*2 )) #allocating memory for the dataset
		self._initialise_dataset(load_file)
		return

	def get_output_dim(self):
		return self.dataset.shape[1]

	def _initialise_dataset(self, load_file):
		if isinstance(load_file, str):
			if self.smooth_oscillation:
				params, alpha, beta_m, beta_amp, beta_ph, times = load_dataset(load_file, N_data=None, N_grid = None, shuffle = False, n_params = 6, N_entries = 4)
				alpha_beta = np.stack([alpha, beta_m, beta_amp, beta_ph], 2) #(N,D,4)
			else:
				params, alpha, beta, times = load_dataset(load_file, N_data=None, N_grid = self.N_times, shuffle = False, n_params = 6)
				alpha_beta = np.stack([alpha, beta], 2) #(N,D,2)
				
			if times.shape[0] < self.N_times:
				raise ValueError("The input file {} holds a input dataset with only {} grid points but {} grid points are asked. Plese provide a different dataset file or reduce the number of grid points for the dataset".format(load_file, times.shape[0], self.N_times))
			N_init = params.shape[0] #number of points in the dataset
			for i in range(N_init):
				ids_ = np.random.choice(len(times), self.N_times)
				i_start = i
				if i >= self.N_batch: break
				new_data = np.repeat(params[i,None,:], self.N_times, axis = 0) #(D,6)
				new_data = np.concatenate([times[ids_,None], new_data, alpha_beta[i,ids_,:]], axis =1) #(N,9/11)

				id_start = i*(self.N_times)
				
				#robustness check (temporary thing: it should eventually be improved) #VERY VERY UGLY
				if np.any(new_data[:,8]>3.14) or np.any(new_data[:,8]<0.1):
					if i >= 1:
						new_data = self.dataset[id_start-self.N_times:id_start,:]
					else:
						new_data = 0
				
				self.dataset[id_start:id_start + self.N_times,:] = new_data
		else:
			i_start = 0

			#adding the angles not in the dataset
		for i in range(i_start+1, self.N_batch):
			print("Generated angles ",i)
			self.replace_angle(i) #changing the i-th angle in the datset
		print("Dataset initialized")
		return

	def replace_angle(self, i):
		"Updates the angles corresponding to the i-th point of the batch. They are inserted in the dataset."
		params = np.random.uniform(self.ranges[:,0], self.ranges[:,1], size = (6,)) #(6,)
		times = np.random.uniform(-self.t_min, 0., (self.N_times,)) #(D,)
		if self.smooth_oscillation:
			alpha, beta = get_alpha_beta(*params, times, self.smooth_oscillation, verbose = False) #(D,), (D,3)
			alpha_beta = np.concatenate([alpha[:,None], beta], axis =1) #(D,4)
		else:
			alpha, beta = get_alpha_beta(*params, times, self.smooth_oscillation, verbose = False) #(D,)
			alpha_beta = np.column_stack([alpha, beta]) #(D,2)

		new_data = np.repeat(params[None,:], self.N_times, axis = 0) #(D,6)
		new_data = np.concatenate([times[:,None], new_data, alpha_beta], axis =1) #(D,9/11)

		id_start = i*(self.N_times)
		#robustness check (temporary thing: it should eventually be improved) #VERY VERY UGLY
		if np.any(new_data[:,8]>3.14) or np.any(new_data[:,8]<-0.1):
			return
		self.dataset[id_start:id_start + self.N_times,:] = new_data
		return

	def __call__(self):
		"Return a dataset of angles: each row is [t, q, chi1, chi2, theta1, theta2, deltaphi, alfa, beta]"
		i = -1 #keeps track of the dataset index we are replacing
		j = 0 #keeps track of the iteration number, to know when to replace the data
		while True:
			yield self.dataset
			if self.replace_step is not None:
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

	def __init__(self, name = "NN_precession_model", smooth_oscillation = True):
		super(NN_precession, self).__init__(name = name)
		print("Initializing model ",self.name)
		self.history = []
		self.metric = []
		self.epoch = 0
		self.smooth_oscillation = smooth_oscillation
		self.ranges = None
		if smooth_oscillation:
			self.scaling_consts = tf.constant([1e4, 1.,1e20,1e20], dtype = tf.float32) #scaling constants for the loss function (set by hand, kind of)
		else:
			self.scaling_consts = tf.constant([1., 1.], dtype = tf.float32) #scaling constants for the loss function (set by hand, kind of)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3) #default optimizer
		
		self._l_list = []
		self._l_list.append(tf.keras.layers.Dense(128*2, activation=tf.nn.tanh) )
		self._l_list.append(tf.keras.layers.Dense(128*1, activation=tf.nn.tanh) )
		if smooth_oscillation:
			self._l_list.append(tf.keras.layers.Dense(4, activation=tf.keras.activations.linear)) #outputs: alpha, beta
		else:
			self._l_list.append(tf.keras.layers.Dense(2, activation=tf.keras.activations.linear)) #outputs: alpha, beta
		
		self.build(input_shape = (None, 7)) #This is required to specify the input shape of the model and to state which are the trainable paramters		

	def call(self, inputs):
		"Inputs: [t, params (6,)]"
		output = inputs
		for l in self._l_list:
			output = l(output)
		return output #(N,n_vars)
		
	def call_alpha_beta(self,inputs):
		res = self.call(inputs) #(N, 2/4)
		if self.smooth_oscillation:
			beta = res[:,1] + res[:,2]*tf.math.cos(res[:,3]) #(N,)
			return tf.stack([res[:,0], beta], axis = 1) #(N,2)
		else:
			return res
			
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
		loss = tf.math.square(self.call(X[:,:7]) - X[:,7:]) #(N,2/4)
		loss = tf.math.divide(loss, self.scaling_consts)
		loss = tf.reduce_sum(loss[:,:2], axis = 1) /X.shape[1] #(N,) #DEBUG: NO AMP PH
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

	def fit(self, generator, N_epochs, learning_rate = 5e-4, save_output = True, plot_function = None, checkpoint_step = 20000, print_step = 10, validation_file = None):
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate) #default optimizer
		epoch_0 = self.epoch

			#initializing the validation file
		if isinstance(validation_file, str):
			if self.smooth_oscillation:
				val_params, val_alpha, val_beta, val_beta_amp, val_beta_ph, val_times = load_dataset(validation_file, N_data=None, N_grid = None, shuffle = False, n_params = 6, N_entries = 4)
			else:
				val_params, val_alpha, val_beta, val_times = load_dataset(validation_file, N_data=None, N_grid = None, shuffle = False, n_params = 6)

		#assigning ranges (if generator has them)
		try:
			self.ranges = generator.ranges
		except:
			self.range = None

		tf_dataset = tf.data.Dataset.from_generator(
     			generator,
    			output_signature = tf.TensorSpec(shape=(None,generator.get_output_dim()), dtype=tf.float32)
					)#.prefetch(tf.data.experimental.AUTOTUNE) #good idea?? Probably yes
		
		n_epoch = -1
		for X in tf_dataset:
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
				if save_output:
					self.save_weights("{}/{}".format(self.name, self.name)) #overwriting the newest
					np.savetxt(self.name+"/"+self.name+".loss", np.array(self.history))
					np.savetxt(self.name+"/"+self.name+".metric", np.array(self.metric))

			if n_epoch%checkpoint_step ==0 and n_epoch != 0:
				if save_output:
					self.save_weights("{}/{}/{}".format(self.name, str(self.epoch), self.name)) #saving to an archive

				if plot_function is not None:
					plot_function(self, "{}/{}".format(self.name, str(self.epoch)))

				if isinstance(validation_file, str): #computing validation metric
					if self.smooth_oscillation:
						val_alpha_NN, val_beta_NN, val_beta_amp_NN, val_beta_ph_NN = self.get_alpha_beta(*val_params.T,val_times, True)
						#here val_beta_NN is actually beta mean
					else:
						val_alpha_NN, val_beta_NN = self.get_alpha_beta(*val_params.T,val_times)
					loss_alpha = np.mean(np.divide(np.abs(val_alpha_NN- val_alpha),val_alpha))
					loss_beta = np.mean(np.divide(np.abs(val_beta_NN- val_beta),val_beta))

					self.metric.append((self.epoch, loss_alpha, loss_beta))
					print("\tMetric: {} {} {} ".format(self.metric[-1][0],self.metric[-1][1], self.metric[-1][2]))
					
		return self.history

	def load_everything(self, path):
		"Loads model and tries to read metric and loss"
		print("Loading model from: ",path)
		self.load_weights(path)
		print(path)
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

	def get_alpha_beta(self,q, chi1, chi2, theta1, theta2, delta_phi, times, get_mean = False):
		#do it better
		X = np.column_stack( [q, chi1, chi2, theta1, theta2, delta_phi]) #(N,6)

		if X.ndim == 1:
			X = X[None,:]

		N = X.shape[0]
		alpha = np.zeros((N , len(times)))
		beta = np.zeros((N, len(times)))
		
		if get_mean and self.smooth_oscillation:
			beta_amp = np.zeros((N, len(times)))
			beta_ph = np.zeros((N, len(times)))

		for i in range(len(times)):
			t = np.repeat([times[i]], N)[:,None]
			X_tf = np.concatenate([t,X],axis =1)
			X_tf = tf.convert_to_tensor(X_tf, dtype=tf.float32)
			if get_mean and self.smooth_oscillation:
				alpha_beta = self.call(X_tf) #(N,4)
				beta_amp[:,i] = alpha_beta[:,2]
				beta_ph[:,i] = alpha_beta[:,3]
			else:
				alpha_beta = self.call_alpha_beta(X_tf) #(N,2)
			alpha[:,i] = alpha_beta[:,0] 
			beta[:,i] = alpha_beta[:,1] 

		if get_mean and self.smooth_oscillation:
			return alpha, beta, beta_amp, beta_ph
		return alpha, beta


def plot_solution(model, N_sol, t_min,   seed, folder = ".", show = False, smooth_oscillation = False):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		params = np.random.uniform(model.ranges[:,0], model.ranges[:,1], size = (N_sol,6))
	except:
		ranges = np.array([(1.1,10.), (0.,1.), (0.,1.), (0., np.pi), (0., np.pi), (0., 2.*np.pi)])		
		params = np.random.uniform(ranges[:,0], ranges[:,1], size = (N_sol,6))

	np.random.set_state(state) #using original random state
	times = np.linspace(-np.abs(t_min), 0.,1000) #(D,)

	if smooth_oscillation:
		alpha, beta = get_alpha_beta(*params.T, times, smooth_oscillation = True, verbose = False) #true alpha and beta
		beta, beta_amp, beta_ph = beta[:,:,0], beta[:,:,1], beta[:,:,2]
		NN_alpha, NN_beta, NN_beta_amp, NN_beta_ph = model.get_alpha_beta(*params.T,times, True) #(N,D)
	else:
		alpha, beta = get_alpha_beta(*params.T, times, smooth_oscillation = False, verbose = False) #true alpha and beta
		NN_alpha, NN_beta = model.get_alpha_beta(*params.T,times) #(N,D)

		#plotting
	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\alpha$")
	plt.plot(times, NN_alpha.T, c = 'r')
	plt.plot(times, alpha.T, c= 'b')
	if isinstance(folder, str):
		plt.savefig(folder+"/alpha.pdf", transparent =True)

	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\beta$")
	plt.plot(times, NN_beta.T, c= 'r')
	plt.plot(times, beta.T, c= 'b')
	if isinstance(folder, str):
		plt.savefig(folder+"/beta.pdf", transparent =True)

	if smooth_oscillation:
			#plotting
		plt.figure()
		plt.xlabel("times (s/M_sun)")
		plt.ylabel(r"$A_\beta$")
		plt.plot(times, NN_beta_amp.T, c = 'r')
		plt.plot(times, beta_amp.T, c= 'b')
		if isinstance(folder, str):
			plt.savefig(folder+"/beta_amp.pdf", transparent =True)

		plt.figure()
		plt.xlabel("times (s/M_sun)")
		plt.ylabel(r"$\phi_\beta$")
		plt.plot(times, NN_beta_ph.T, c= 'r')
		plt.plot(times, beta_ph.T, c= 'b')
		if isinstance(folder, str):
			plt.savefig(folder+"/beta_ph.pdf", transparent =True)

	if show:
		plt.show()
	else:
		plt.close('all')


def plot_validation_set(model, N_sol, validation_file, folder = ".", show = False, smooth_oscillation = False):
	if smooth_oscillation:
		params, alpha, beta, beta_amp, beta_ph, times = load_dataset(validation_file, N_data=N_sol, N_grid = None, shuffle = False, n_params = 6, N_entries = 4)
		NN_alpha, NN_beta, NN_beta_amp, NN_beta_ph = model.get_alpha_beta(*params.T,times, True) #(N,D)
	else:
		params, alpha, beta, times = load_dataset(validation_file, N_data=N_sol, N_grid = None, shuffle = False, n_params = 6)
		NN_alpha, NN_beta = model.get_alpha_beta(*params.T,times) #(N,D)

		#plotting
	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\alpha$")
	plt.plot(times, NN_alpha.T, c = 'r')
	plt.plot(times, alpha.T, c= 'b')
	if isinstance(folder, str):
		plt.savefig(folder+"/alpha.pdf", transparent =True)

	plt.figure()
	plt.xlabel("times (s/M_sun)")
	plt.ylabel(r"$\beta$")
	plt.plot(times, NN_beta.T, c= 'r')
	plt.plot(times, beta.T, c= 'b')
	if isinstance(folder, str):
		plt.savefig(folder+"/beta.pdf", transparent =True)


	if show:
		plt.show()
	else:
		plt.close('all')


###############################################################################################################
###############################################################################################################
###############################################################################################################
def create_dataset_alpha_beta(N_angles, filename, N_grid, tau_min, q_range, chi1_range= (0.,1.), chi2_range = (0.,1.), theta1_range = (0., np.pi), theta2_range = (0., np.pi), delta_phi_range = (-np.pi, np.pi), smooth_oscillation = False, alpha =0.5, verbose = False ):
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
		smooth_oscillation	whether to represent beta with beta_avg, amplitude and phase
		alpha				distorsion parameter (for accumulating more grid points around the merger)
		verbose				Whether to print the output to screen
	"""
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")
	if not isinstance(filename, str):
		raise TypeError("filename is "+str(type(filename))+"! Expected to be a string.")

	range_list = [q_range, chi1_range, chi2_range, theta1_range, theta2_range, delta_phi_range]


	time_grid = np.linspace(np.power(np.abs(tau_min),alpha), 1e-20, N_grid)
	time_grid = -np.power(time_grid,1./alpha)
	time_grid[-1] = 0.
	
		#initializing file. If file is full, it is assumed to have the proper time grid
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file ", filename, " created")
		if smooth_oscillation == True:
			time_header = np.concatenate((np.zeros((6,)), time_grid, time_grid, time_grid, time_grid) )[None,:]	
		else:
			time_header = np.concatenate((np.zeros((6,)), time_grid, time_grid) )[None,:]
		np.savetxt(filebuff, time_header, header = "#Alpha, Beta dataset" +"\n# row: params (None,6) | alpha (None,"+str(N_grid)+")| beta (None,"+str(N_grid)+")\n# N_grid = "+str(N_grid)+" | tau_min ="+str(tau_min)+" | q_range = "+str(q_range)+" | chi1_range = "+str(chi1_range)+" | chi2_range = "+str(chi2_range)+" | theta1_range = "+str(theta1_range)+" | theta2_range = "+str(theta2_range)+" | delta_phi_range = "+str(delta_phi_range), newline = '\n')
	else:
		filebuff = open(filename,'a')
	
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
	
	b_size = 10 #batch size at which angles are stored before being saved
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

		alpha, beta = get_alpha_beta(*params.T, time_grid, smooth_oscillation = smooth_oscillation, verbose= verbose)
		if smooth_oscillation:
				#removing possible outliers
			ids = np.where(np.logical_and(beta[:,:,0]>3.14, beta[:,:,0]<-0.1))
			ids = set(ids[0])
			if len(ids)> 0:
				beta[list(ids),:,0] = 1. #very ugly...
		
			to_save = np.concatenate([params, alpha, beta[:,:,0], beta[:,:,1], beta[:,:,2]], axis = 1) #(N,4*D)
			print(to_save.shape)
		else:
			to_save = np.concatenate([params, alpha, beta], axis = 1)
		np.savetxt(filebuff, to_save) #saving the batch to file
		print("Generated angle: ", count)

	return





















	


