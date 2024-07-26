"""
Module precession_helper.py
===========================
	Module for training a ML model for fitting the precessing angles alpha, beta as a function of (theta1, theta2, deltaphi, chi1, chi2, q).
	Requires precession module (pip install precession) and tensorflow (pip install tensorflow)
"""

import numpy as np
import os
import sys
import warnings
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim
import scipy.signal
from mlgw.GW_helper import f_min, frequency22_merger, f_ISCO
from tqdm import tqdm
import precession

import tensorflow as tf

###############################################################################################################
###############################################################################################################
###############################################################################################################
class CosinesLayer(tf.keras.layers.Layer):
	def __init__(self, units=32, frequencies = (1,10)):
		super().__init__()
		self.units = units
		self.frequencies = tf.experimental.numpy.logspace(*np.log10(frequencies), self.units, dtype = tf.float32)
		

	def build(self, input_shape):
		w_init = tf.stack([self.frequencies, *[tf.random.normal([self.units], 0, 1e-3) for _ in range(1,input_shape[-1])]])
		#w_init = tf.stack([tf.random.shuffle(self.frequencies) for _ in range(input_shape[-1])])
		
		#print(w_init)
		
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer= lambda shape, dtype: w_init,
			trainable=True,
			name = 'cosine_freqs'
		)
		self.b = self.add_weight(
			shape=(self.units,), initializer="random_normal", trainable=True, name = 'cosine_phases'
		)

	def call(self, inputs):
		return tf.concat([inputs, tf.math.cos(tf.linalg.matmul(inputs, self.w) + self.b)], 1)

def augment_for_angles(theta):
	K = theta.shape[-1]
	assert K in [5,6,7]
	
	q = np.squeeze(theta[:,0])
	phi1, phi2 = 0., 0.

	if K == 7:
		s1, s2, t1, t2, phi1, phi2 = theta[:,1:].T
		theta_full = theta
	elif K == 6:
		s1, s2, t1, t2, phi1 = theta[:,1:].T
		theta_full = np.column_stack([*theta.T, np.zeros(theta[:,0].shape)])
	elif K == 5:
		s1, s2, t1, t2 = theta[:,1:].T
		theta_full = np.column_stack([*theta.T, np.zeros(theta[:,0].shape), np.zeros(theta[:,0].shape)])

	s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
	s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	
	chi1_p, chi2_p =  np.sqrt(s1x**2 + s1y**2), np.sqrt(s2x**2 + s2y**2)
	chi_P_2D_norm, _, _ = get_S_effective_norm(theta_full)
	
	return np.column_stack([*theta.T, q**2, chi_P_2D_norm, chi1_p, chi2_p, s1z, s2z])

def to_polar(s):
	"Given the 3 dimensionless components of a spin, it computes the spherical coordinates representation"
	s_norm = np.linalg.norm(s, axis =1) + 1e-10
	theta = np.arccos(s[:,2]/s_norm)
	phi = np.arctan2(s[:,1], s[:,0])

	return np.column_stack([s_norm, theta, phi]) #(N,3)

def to_cartesian(s, t, phi):
	return s*np.sin(t)*np.cos(phi), s*np.sin(t)*np.sin(phi), s*np.cos(t)

def Rot(angle, axis = 'z'):
	angle = np.asarray(angle)
	squeeze =  (angle.ndim ==0)
	angle = np.atleast_1d(angle)
	if axis == 'z':
		R = np.stack([[[np.cos(a), -np.sin(a), 0],[np.sin(a), np.cos(a), 0], [0,0,1]] for a in angle], axis = 0)	
	elif axis == 'y':
		R = np.stack([[[np.cos(a), 0, np.sin(a)], [0,1,0],[-np.sin(a), 0, np.cos(a)]] for a in angle], axis = 0)

	if squeeze: R = np.squeeze(R)

	return R
		

def to_J0_frame(L0_spins, alpha0, beta0, gamma0):
	#FIXME: this is crazy complicated!!!
	L0_spins = np.asarray(L0_spins)
	
		#Implementing the rotation in https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM_EulerAngles.c#L1303
	#IMRPhenomT_rotate_z(cosPhiJ,    sinPhiJ,    &(LNhatx->data->data[i]), &(LNhaty->data->data[i]), &(LNhatz->data->data[i]));
    #IMRPhenomT_rotate_y(cosThetaJ,  sinThetaJ,  &(LNhatx->data->data[i]), &(LNhaty->data->data[i]), &(LNhatz->data->data[i]));
    #IMRPhenomT_rotate_z(cosKappa,   sinKappa,   &(LNhatx->data->data[i]), &(LNhaty->data->data[i]), &(LNhatz->data->data[i]));

	return np.linalg.multi_dot([Rot(alpha0,'z'), Rot(beta0,'y'), Rot(gamma0,'z'), L0_spins.T]).T
	
def get_alpha0_beta0_gamma0(theta, L_0):
	"""
	Given the input vector theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z] and the *dimensionless* orbital momentum, it computes the Euler angles connecting the L0 and the J0 frames. The orbital angular momentum implicitly defines the reference frequency by and can be computed with:
	
	```Python
	gen = mlgw.GW_generator()
	t_grid = np.linspace(-40, 0.001, int(40.001*4096.))
	L, _ = gen.get_L(theta, t_grid)
	L_0 = L[0]
	```
	"""
	theta = np.asarray(theta)
	squeeze = (theta.ndim ==1)
	theta = np.atleast_2d(theta)
	L_0 = np.atleast_1d(L_0)
	assert theta.shape[1] == 8
	
	m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = theta.T
	
	L_dim = L_0*(m1+m2)**2
	L_vect = np.stack([np.zeros(L_dim.shape), np.zeros(L_dim.shape), L_dim], axis = 1)
	S1 = (np.stack([s1x, s1y, s1z], axis = 1).T*m1**2).T
	S2 = (np.stack([s2x, s2y, s2z], axis = 1).T*m2**2).T
	S = S1 + S2
	
	J = L_vect + S
	J_norm = (J.T/np.linalg.norm(J, axis = 1)).T

	theta_JL0 = np.arccos(J_norm[:,2]) #polar angle of J in the L0 frame
	phi_JL0 = np.arctan2(J_norm[:,1], J_norm[:,0]) #azimuthal angle of J in the L0 frame

	#alpha offset
	#https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRPhenomTPHM_EulerAngles.c#L1287
	alpha_offset = np.arctan2(S[:,1], S[:,0]) - np.pi
	kappa = np.pi-alpha_offset

		#This is the direction of L in the Jframe, if you care...
	#L_Jframe =  np.einsum('ijk,ikl,iln,n->in', Rot(-kappa,'z'), Rot(-theta_JL0,'y'),Rot(-phi_JL0,'z'), np.array([0,0,1]))

	#alpha0, beta0, gamma0 parametrize the rotation from the L0 frame to the J frame and viceversa
	if squeeze:
		alpha_offset, theta_JL0, phi_JL0 = alpha_offset[0], theta_JL0[0], phi_JL0[0]

	return alpha_offset, theta_JL0, np.pi-phi_JL0
	
	


###############################################################################################################
###############################################################################################################

class angle_params_keeper:
	def __init__(self, Psi = None):
		if isinstance(Psi, angle_params_keeper): self.assign(Psi())
		else: self.assign(Psi)

	def assign(self, Psi):
		if Psi is None: Psi = np.zeros((10,))
		
		Psi = np.asarray(Psi)
		if Psi.shape[-1]==6:
			Psi = np.stack([*Psi.T[:5], *np.zeros((4, *Psi.T[-1].shape)), *Psi.T[5:]], axis = -1)
		if Psi.shape[-1]==4:
			Psi = np.stack([*Psi.T[:2],np.zeros(Psi.T[-1].shape), *Psi.T[2:], *np.zeros((5, *Psi.T[-1].shape))], axis = -1)
		
		self.A_alpha, self.B_alpha, self.alpha0, self.A_beta, self.B_beta, self.amp_beta, self.A_ph_beta, self.B_ph_beta, self.ph0_beta, self.gamma0  = Psi.T


	def __call__(self):
		return np.array([self.A_alpha, self.B_alpha, self.alpha0,
			self.A_beta, self.B_beta, self.amp_beta, self.A_ph_beta, self.B_ph_beta, self.ph0_beta, self.gamma0]).T

	def __str__(self):
		to_return = "Psi object\n"
		for k, v in self.__dict__.items():
			if k != 'Psi':
				to_return += '\t{}: {}\n'.format(k,v)
		return to_return


class angle_manager:
	
	def __init__(self, mlgw_model, times, fref, fstart, beta_residuals = True):
		self.mlgw_model = mlgw_model
		self.times = times
		self.fref = fref
		self.fstart = fstart
		self.dt = np.mean(np.diff(self.times))
		assert np.allclose(np.diff(self.times), self.dt), "An equally spaced time grid must be given!"
		self.beta_residuals = beta_residuals
		
		self.t_max = -0.05 #cutoff for the training: we stop generating angles after this
		self.ids_, = np.where(self.times<self.t_max)
		
		self.pad_seg = (0, len(self.times)-len(self.ids_))

	def get_L(self, theta, ph = None):
		#TODO: move this shit in the GW generator!
		theta = np.asarray(theta)
		assert theta.shape == (8,)
		
		theta_NP = theta[[0,1,4,7]]

		if ph is None:
			_, ph = self.mlgw_model.get_modes(theta_NP, self.times, (2,2)) #returns amplitude and phase of the wave

		
		m1, m2 = theta[[0,1]]
		M = m1+m2
		mu = (m1*m2)/M
		mu_tilde = (mu**3/M**4)/4.93e-6
	
		omega_orb = -0.5*np.gradient(np.squeeze(ph), self.times)[self.ids_]
	
		L = (mu_tilde/omega_orb)**(1./3.) # this is L/M**2
		
		L, omega_orb = np.pad(L, self.pad_seg, mode ='edge'), np.pad(omega_orb, self.pad_seg, mode ='edge')
		
		return L, omega_orb
	
	def get_beta_trend(self, L, a, b):
		#return a/(L+b)
		
		return a/(L+1) + b
		
		#return (L+a)/np.sqrt(L**2+b) #This is very good to fit cos(beta)

	def fit_beta_trend(self, L, beta):
		"""
		Returns the coefficients a,b that transform the angular momentum to beta, plus the average amplitude of the residual and their phase
		"""

		L, beta = L[self.ids_], beta[self.ids_]

		loss = lambda x: np.mean(np.square(self.get_beta_trend(L, *x) - beta))
		
		def loss_derivative(x):
			loss_i = self.get_beta_trend(L, *x) - beta
			#grad_a = 2*loss_i/(L+x[1])
			#grad_b = -2*loss_i*x[0]/(L+x[1])**2
			grad_a = 2*loss_i/(L+1)
			grad_b = 2*loss_i
			return np.array([np.mean(grad_a), np.mean(grad_b)])
		
		b0 = (beta[0]*L[0] - beta[-1]*L[-1])/(beta[-1]-beta[0])
		a0 = beta[0]*L[0] + b0*beta[0]
		
		b0 = beta[0]-a0/(L[0]+1)
		a0 = (beta[-1]-beta[0])/(1/(L[-1]+1)-1/(L[0]+1))
		
		res = scipy.optimize.minimize(loss, x0 = np.array([a0, b0]), jac = loss_derivative)
		#res = scipy.optimize.minimize(loss, x0 = np.random.normal(0,1,2))

		return *res.x, res.success
	
	def get_residual_amp_ph(self, residuals):
		if len(residuals) == len(self.times):
			residuals = residuals[self.ids_]
	
		hil = scipy.signal.hilbert(residuals)
		amp_res, ph_res = np.mean(np.abs(hil)[100:-100]), np.unwrap(np.angle(hil))
		
		ph_res	= np.pad(ph_res, self.pad_seg, mode ='edge') #ph_beta_res is only valid for t<0!!!!!

		return amp_res, ph_res	
	
	def get_beta(self, L, omega_orb, M, q, Psi):
		Psi = angle_params_keeper(Psi)
		beta = self.get_beta_trend(L, Psi.A_beta, Psi.B_beta)
		
		if self.beta_residuals:
			ph_beta_res = self.get_integrated_Omega_p(L, omega_orb, M, q, Psi.A_ph_beta, Psi.B_ph_beta, Psi.ph0_beta)
			beta += Psi.amp_beta*np.cos(ph_beta_res)
		return beta
	
	def get_alpha(self, L, omega_orb, M, q, Psi):
		Psi = angle_params_keeper(Psi)
		alpha = self.get_integrated_Omega_p(L, omega_orb, M, q, Psi.A_alpha, Psi.B_alpha, Psi.alpha0)
		return alpha
	
	def get_Omega_p(self, L, omega_orb, M, q, f, g):
		M*=4.93e-6
		J = L * f + g
		return (3+1.5/q)*J*M*omega_orb**2
	
	def get_integrated_Omega_p(self, L, omega_orb, M, q, f, g, h):
		Omega_p = self.get_Omega_p(L, omega_orb, M, q, f, g)
		
		return np.cumsum(Omega_p)*self.dt+h
	
	def fit_Omega_p(self, alpha, L, omega_orb, M, q):
	
		ids_, = np.where(self.times<self.t_max)
		alpha, L, omega_orb = alpha[self.ids_], L[self.ids_], omega_orb[self.ids_]
		
		loss = lambda x: np.mean(np.square(self.get_integrated_Omega_p(L, omega_orb, M, q, *x) - alpha))
		
		x0 = np.array([1,1, alpha[0]])
		x0 = [*np.random.normal(0,1,2), alpha[0]]
		res = scipy.optimize.minimize(loss, x0 = x0, method = 'BFGS')
		#print('################\n',res)
		
		return *res.x, res.success
	
	def get_alpha_beta_gamma(self, theta, Psi):
	
		Psi = angle_params_keeper(Psi)
	
		M, q = theta[0]+theta[1], theta[0]/theta[1]
		L, omega_orb = self.get_L(theta)
		
		alpha = self.get_alpha(L, omega_orb, M, q, Psi)
		beta = self.get_beta(L, omega_orb, M, q, Psi)
		
		alpha_dot = self.get_Omega_p(L, omega_orb, M, q, Psi.A_alpha, Psi.B_alpha)
	
		gamma = np.cumsum(-alpha_dot*np.cos(beta))*self.dt + Psi.gamma0
		
		return alpha, beta, gamma
	
	def get_angles_at_ref_frequency(self, theta):
		alpha, beta, gamma, _ = get_IMRPhenomTPHM_angles(*theta, t_grid = None, fref = self.fref, fstart = self.fstart)
		beta = np.arccos(beta)
		return alpha[0], beta[0], gamma[0]
	
	def get_reduced_alpha_beta(self, theta, plot = False):
		theta = np.asarray(theta)
		assert theta.shape == (8,)
		success = True
		
		M, q = theta[0]+theta[1], theta[0]/theta[1]
		
		alpha, beta, gamma = get_IMRPhenomTPHM_angles(*theta, t_grid = self.times, fref = self.fref, fstart = self.fstart)
		beta = np.arccos(beta);
		L, omega_orb = self.get_L(theta)
		
		Psi = angle_params_keeper()
		
			#Fitting beta trend
		Psi.A_beta, Psi.B_beta, s = self.fit_beta_trend(L, beta)#; print(s)
		success = success and s
		
			#Fitting beta ph
		beta_res = beta-self.get_beta_trend(L, Psi.A_beta, Psi.B_beta)
		if self.beta_residuals:
			Psi.amp_beta, ph_beta_res = self.get_residual_amp_ph(beta_res)
			Psi.A_ph_beta, Psi.B_ph_beta, Psi.ph0_beta, s = self.fit_Omega_p(ph_beta_res, L, omega_orb, M, q)
			#success = success and s #We don't care if the residual model fails...
		else:
			Psi.amp_beta, Psi.A_ph_beta, Psi.B_ph_beta, Psi.ph0_beta = 0., 0., 0., 0.
			ph_beta_res = np.full((len(self.times), ), np.nan)
		
			#Fitting alpha
		Psi.A_alpha, Psi.B_alpha, Psi.alpha0, s = self.fit_Omega_p(alpha, L, omega_orb, M, q)#; print(s)
		success = success and s
		alpha_res = alpha-self.get_alpha(L, omega_orb, M, q, Psi)
		
		Psi.gamma0 = gamma[0]

		if plot:
			alpha_pred, beta_pred, gamma_pred = self.get_alpha_beta_gamma(theta, Psi)

			ph_beta_res_pred = self.get_integrated_Omega_p(L, omega_orb, M, q, Psi.A_ph_beta, Psi.B_ph_beta, Psi.ph0_beta)

			if self.beta_residuals:
				assert np.allclose(beta_pred-self.get_beta_trend(L, Psi.A_beta, Psi.B_beta), Psi.amp_beta*np.cos(ph_beta_res_pred), equal_nan = True)

			fig, axes = plt.subplots(5,1, figsize = (6.4, 7.4), sharex = True)
			axes[0].set_title('Beta trend')
			axes[0].plot(self.times, beta, c = 'coral', label = 'true')
			axes[0].plot(self.times, self.get_beta_trend(L, Psi.A_beta, Psi.B_beta), c = 'cyan', label = 'fit')
			axes[0].legend()
			
			axes[1].set_title('Beta trend residuals')
			axes[1].plot(self.times, beta-self.get_beta_trend(L, Psi.A_beta, Psi.B_beta), c = 'coral')
			axes[1].plot(self.times, Psi.amp_beta*np.cos(ph_beta_res_pred), c = 'cyan')
			axes[1].axhline(Psi.amp_beta, c = 'k')
			axes[1].axhline(-Psi.amp_beta, c = 'k')

			axes[2].set_title('Phase residual model accuracy')
			axes[2].plot(self.times, ph_beta_res, c = 'coral')
			axes[2].plot(self.times, ph_beta_res_pred, c = 'cyan')
			
			axes[3].set_title('Beta full')
			axes[3].plot(self.times, beta, c = 'coral')
			axes[3].plot(self.times, beta_pred, c = 'cyan')

			axes[4].set_title('Beta full - residuals')
			axes[4].plot(self.times, beta - beta_pred, c = 'k', ls = '--')

			plt.tight_layout()
			
			fig, axes = plt.subplots(2,1, sharex = True)
			axes[0].set_title('Alpha')
			axes[0].plot(self.times, alpha, c = 'coral', label = 'true')
			axes[0].plot(self.times, alpha_pred, c = 'cyan', label = 'fit')
			axes[0].legend()

			axes[1].set_title('Residuals')
			axes[1].plot(self.times, alpha_res, c = 'k', ls = '--')
			lim = np.max(np.abs(alpha_res)[self.times<-1])
			axes[1].set_ylim([-2*lim, 2*lim])

			plt.tight_layout()
			
			fig, axes = plt.subplots(2,1, sharex = True)
			axes[0].set_title('Gamma model')
			axes[0].plot(self.times, gamma, c = 'coral', label = 'true')
			axes[0].plot(self.times, gamma_pred, c = 'cyan', label = 'fit')
			axes[1].set_title('Residuals')
			axes[1].plot(self.times, gamma-gamma_pred, c = 'coral')
			lim = np.max(np.abs(gamma-gamma_pred)[self.times<-1])
			axes[1].set_ylim([-2*lim, 2*lim])
			axes[0].legend()
			plt.tight_layout()
			
			plt.show()

		return Psi(), alpha_res, success

		

###############################################################################################################
###############################################################################################################

def get_random_chi(N, chi_range = (0.,0.8)):
	"""
	Extract a random chi value

	Inputs:
		N: int
			Number of spins to extract
		chi_range: tuple
			Range (min, max) for the magnitude of the spin
	Output:
		chi: :class:`~numpy:numpy.ndarray`
			shape (N,3) - Extracted spins
	"""
	chi = np.random.uniform(chi_range[0], chi_range[1], (N,))
	chi_vec = np.random.normal(0,1, (N,3))
	chi_vec = (chi_vec / np.linalg.norm(chi_vec, axis = 1)) * chi
	return chi_vec

def set_effective_spins(m1, m2, chi1, chi2):
	"""
	Given a generic spin configuration, it assigns the spins to a single BH. The inplane spin is assigned according to the spin parameter (https://arxiv.org/pdf/2012.02209.pdf).
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
	raise NotImplementedError
	if isinstance(m1,(float, int)):
		m1 = np.array([m1]) #(1,)
		m2 = np.array([m2]) #(1,)
		chi1 = np.array([chi1])#(1,3)
		chi2 = np.array([chi2])#(1,3)
		squeeze = True
	else:
		squeeze = False
	print(chi1.shape)
	
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

def get_S_effective_norm(theta, mtot = 20.):
	"""
	Computes the norm of the effective spin parameter, given masses and in plane dimensionless components of the spins.
	The spin parameter is defined in https://arxiv.org/pdf/2012.02209.pdf
	Inputs:
		theta (8,)/(N,8)					Parameters of the BBH, as returned by the angle dataset [q, s1, s2, t1, t2, phi1, phi2, fstart]
	Outputs:
		chi_P_2D ()/(N,)		norm of the chi_P 2d
		ids_S1_pos				Indices where S1_perp > S2_perp
		ids_S2_pos				Indices where S2_perp > S1_perp
		
	"""
	theta = np.atleast_2d(theta)
	m1, m2 = 20*theta[:,0]/(1+theta[:,0]), 20/(1+theta[:,0])
	s1, s2, t1, t2, phi1, phi2 = theta[:,1:].T
	s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
	s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	chi1_p, chi2_p = np.stack([s1x, s1y], axis = 1), np.stack([s2x, s2y], axis = 1)
	S1_p, S2_p = compute_S_effective(mtot*theta[:,0]/(1+theta[:,0]), mtot/(1+theta[:,0]), chi1_p, chi2_p)
	ids_s1_p, = np.where(np.sum(np.square(S1_p), axis = 1)>0.)
	ids_s2_p, = np.where(np.sum(np.square(S2_p), axis = 1)>0.)
	return np.linalg.norm(S1_p + S2_p, axis =1), ids_s1_p, ids_s2_p


def compute_S_effective(m1,m2, chi1_perp, chi2_perp):
	"""
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


def get_IMRPhenomTPHM_angles(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, t_grid = None, fref = 0., fstart = None, inclination = 0, phiref = 0):

	ModeArray = lalsim.SimInspiralCreateModeArray()
	for mode in [(2,2)]:
		lalsim.SimInspiralModeArrayActivateMode(ModeArray, mode[0], mode[1])
	
	#TODO: be aware that the angles depend on the inclination and reference frequency, for some reason I can't explain yet
	#This needs to be carefully addressed in the model
	
	if t_grid is not None:
		t_min = np.abs(t_grid[0])
		deltaT = np.min(np.diff(t_grid))
	else:
		assert fstart, "If a time grid is not given, fstart must be specified"
		deltaT = 1/4096.

	if not fstart: fstart = 0.9*f_min(m1/m2, m1+m2, t_min/(m1+m2))

	lalparams = lal.CreateDict()
	lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
	lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, 0)
	#lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, precver)
	#lalsim.SimInspiralWaveformParamsInsertPhenomXPFinalSpinMod(lalparams, FS)
	lalsim.SimInspiralWaveformParamsInsertPhenomXHMAmpInterpolMB(lalparams,  1)


	hlmQAT, alphaT, cosbetaT, gammaT, af = lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI, s1x, s1y, s1z, s2x, s2y, s2z, 1e6*lal.PC_SI, inclination, deltaT, fstart, fref, phiref, lalparams, 0)
	#print(s1x, s1y, s1z, s2x, s2y, s2z, fstart, fref)
	
	times = np.linspace(-len(alphaT.data.data)*deltaT, 0, len(alphaT.data.data))
	
	if t_grid is not None:
		#print('##\nUser grid, generated grid: ', t_grid[0], times[0])
		#print('mtot, fstart, fref ', m1+m2, fstart, fref)
		if t_grid[0]<times[0]:
			msg = 'The given fstart = {} produces an angle which is too short for the given grid!\nStart of user time grid is {}s while the angles start from {}'.format(fstart, t_grid[0], times[0])
			raise ValueError(msg)
		alpha = np.interp(t_grid, times, alphaT.data.data)
		cosbeta = np.interp(t_grid, times, cosbetaT.data.data)
		gamma = np.interp(t_grid, times, gammaT.data.data)
	
		return alpha, cosbeta, gamma
	else:
		return np.array(alphaT.data.data), np.array(cosbetaT.data.data), np.array(gammaT.data.data), times

def create_dataset_reduced_alpha_beta(gen, N_data, filename, t_coal = 0.5, q_range = (1.,5.), mtot = 20., s1_range = (0.,0.9), s2_range = (0.,0.9), t1_range = (0,np.pi), t2_range = (0,np.pi), phi1_range = 0, phi2_range = 0, t_step = 1e-5, alpha = 0.35):
	"""
	Creates a dataset for the reduced version of the Euler angles alpha and beta, plus the residuals of alpha
	"""
	dirname = os.path.dirname(filename)
	if dirname: os.makedirs(dirname, exist_ok = True)
	
	D_theta = 9

	t_grid = np.linspace(-t_coal*mtot, 0.01, int(t_coal*mtot+0.01)*4096)
	manager = angle_manager(gen, t_grid, 5, 5, beta_residuals = True)

	print("Generating angles alpha and beta")

		##### Create a buffer to save the WFs
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file {} created".format(filename))
		
		header = "# Euler angles alpha, beta \n# row: theta {} | Psi (None, 10) \n# t_coal = {} | q_range = {} | mtot = {} | s1_range = {} | s2_range = {} | t1_range = {} | t2_range = {} | phi1_range = {} | phi2_range = {} \n".format(D_theta, t_coal, q_range, mtot, s1_range, s2_range, t1_range, t2_range, phi1_range, phi2_range)
		filebuff.write(header)
	else:
		print("Saving angles to existing file {}".format(filename))
		filebuff = open(filename,'a')	
	
	for n_angle in tqdm(range(N_data), desc = 'Generating dataset of the angles'):
			#setting value for data
		q = np.random.uniform(*q_range) if isinstance(q_range, (tuple, list)) else float(q_range)
		s1 = np.random.uniform(*s1_range) if isinstance(s1_range, (tuple, list)) else float(s1_range)
		s2 = np.random.uniform(*s2_range) if isinstance(s2_range, (tuple, list)) else float(s2_range)
		t1 = np.arccos(np.random.uniform(*np.cos(t1_range))) if isinstance(t1_range, (tuple, list)) else float(t1_range)
		t2 = np.arccos(np.random.uniform(*np.cos(t2_range))) if isinstance(t2_range, (tuple, list)) else float(t2_range)
		phi1 = np.random.uniform(*phi1_range) if isinstance(phi1_range, (tuple, list)) else float(phi1_range)
		phi2 = np.random.uniform(*phi2_range) if isinstance(phi2_range, (tuple, list)) else float(phi2_range)

		m1, m2 = q * mtot / (1+q), mtot / (1+q)
		s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
		s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
	
		fstart, _ = gen.get_fref_angles([m1, m2, s1z, s2z])
		
			#f_ref and f_start are expressed in terms of the 22 frequency
		manager.fref, manager.fstart = fstart, fstart
		
		theta = [q, s1, s2, t1, t2, phi1, phi2, fstart]

			#Trying to optimize a few times before claiming a failure...
		for _ in range(10):
			Psi, alpha_res, success = manager.get_reduced_alpha_beta([m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])
			if success: break
		
		#print('#####\ntheta dataset: ', [q, s1, s2, t1, t2, phi1, phi2, fstart])
		#print('theta m1,m2: ', [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z])
		#print('Psi: ', Psi[[0,1,3,4]])
		
		if success:
			#TODO: save the residual dataset somewhere to file. Soooooo boring!!!
		
			to_save = np.concatenate([theta, Psi])[None,:] #(1,D)
			np.savetxt(filebuff, to_save)
	
			del to_save
		else:
			warnings.warn('Failed to generate angles for theta = {}'.format(theta))

	filebuff.close()


def create_dataset_alpha_beta_gamma(N_data, N_grid, filename,  t_coal = 0.5, q_range = (1.,5.), m2_range = None, s1_range = (0.,0.9), s2_range = (0.,0.9), t1_range = (0,np.pi), t2_range = (0,np.pi), phi1_range = 0, phi2_range = 0, fref = None, fstart = None, t_step = 1e-5, alpha = 0.35):
	#TODO: write this better!!
	"""
	Creates a dataset for the Euler angles alpha, beta, gamma.
	"""
	dirname = os.path.dirname(filename)
	if dirname: os.makedirs(dirname, exist_ok = True)
	
		#checking if N_grid is fine
	if not isinstance(N_grid, int):
		raise TypeError("N_grid is "+str(type(N_grid))+"! Expected to be a int.")

	if isinstance(m2_range, tuple):
		D_theta = 10 #m2 must be included as a feature
	else:
		D_theta = 9

	print("Generating angles alpha and beta")

		#creating time_grid
	time_grid = -np.power(np.linspace(np.power(np.abs(t_coal), alpha), 0, N_grid), 1/alpha)

		#setting t_coal_freq for generating a waves
	t_coal_freq = 0.05 if np.abs(t_coal) < 0.05 else np.abs(t_coal)


		##### Create a buffer to save the WFs
		
	if not os.path.isfile(filename): #file doesn't exist: must be created with proper header
		filebuff = open(filename,'w')
		print("New file {} created".format(filename))
		time_header = np.concatenate([np.zeros((D_theta,)), time_grid, time_grid])[None,:]
		
		header = "# Euler angles alpha, beta \n# row: theta {} | alpha (None,{})| beta (None,{})\n# N_grid = {} | t_coal = {} | t_step = {} | q_range = {} | m2_range = {} | s1_range = {} | s2_range = {} | t1_range = {} | t2_range = {} | phi1_range = {} | phi2_range = {} | fref = {}| fstart = {} ".format(D_theta, N_grid, N_grid, N_grid, t_coal, t_step, q_range, m2_range, s1_range, s2_range, t1_range, t2_range, phi1_range, phi2_range, fref, fstart)
		np.savetxt(filebuff, time_header, header = header, newline = '\n')
	else:
		print("Saving angles to existing file {}".format(filename))
		filebuff = open(filename,'a')

		##### Creating WFs
	ModeArray = lalsim.SimInspiralCreateModeArray()
	for mode in [(2,2)]:
		lalsim.SimInspiralModeArrayActivateMode(ModeArray, mode[0], mode[1])
	lalparams = lal.CreateDict()
	lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
	lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lalparams, 0)
	lalsim.SimInspiralWaveformParamsInsertPhenomXHMAmpInterpolMB(lalparams,  1)

	for n_angle in tqdm(range(N_data), desc = 'Generating dataset of the angles'):
			#setting value for data
		q = np.random.uniform(*q_range) if isinstance(q_range, (tuple, list)) else float(q_range)
		if m2_range is None:
			m2 = 20. / (1+q)
		else:
			m2 = np.random.uniform(*m2_range) if isinstance(m2_range, (tuple, list)) else m2_range
		m1 = q * m2

		s1 = np.random.uniform(*s1_range) if isinstance(s1_range, (tuple, list)) else float(s1_range)
		s2 = np.random.uniform(*s2_range) if isinstance(s2_range, (tuple, list)) else float(s2_range)
		t1 = np.arccos(np.random.uniform(*np.cos(t1_range))) if isinstance(t1_range, (tuple, list)) else float(t1_range)
		t2 = np.arccos(np.random.uniform(*np.cos(t2_range))) if isinstance(t2_range, (tuple, list)) else float(t2_range)
		phi1 = np.random.uniform(*phi1_range) if isinstance(phi1_range, (tuple, list)) else float(phi1_range)
		phi2 = np.random.uniform(*phi2_range) if isinstance(phi2_range, (tuple, list)) else float(phi2_range)
		
		s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
		s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
		
			#computing fmin and fref
		fstart_ = fstart if fstart else .9*f_min(t_coal_freq, q, m1+m2)
		fref_ = fref if fref else f_ISCO(m1+m2) #frequency22_merger(m1, m2, s1z, s2z)

			#computing tilts at infinity
		#tilts_infty = tilt_infty.hybrid_spin_evolution.calc_tilts_at_infty_hybrid_evolve(
		# 	m1*lalsim.lal.MSUN_SI, m2*lalsim.lal.MSUN_SI,
		#	s1, s2,	t1, t2, 0., fref_, version ='v2')
		#t1_infty, t2_infty = tilts_infty['tilt1_inf'], tilts_infty['tilt2_inf']

		if isinstance(m2_range, tuple):
			temp_theta = [m1, m2, s1, s2, t1, t2, phi1, phi2, fref_, fstart_]	
		else:
			temp_theta = [q, s1, s2, t1, t2, phi1, phi2, fref_, fstart_]

		#print(temp_theta) #DEBUG

			#getting the angles in the J frame!
		hlmQAT, alphaT, cosbetaT, gammaT, af = lalsim.SimIMRPhenomTPHM_CoprecModes(m1*lal.MSUN_SI, m2*lal.MSUN_SI,
			s1x, s1y, s1z,
			s2x, s2y, s2z,
			1., 0., t_step, fstart_, fref_, 0., lalparams, 0)

		#print(s1*np.sin(t1), 0., s1*np.cos(t1), s2*np.sin(t2), 0., s2*np.cos(t2), fstart_, fref_)

		time_full = np.linspace(-alphaT.data.length*t_step, 0., alphaT.data.length) #reduced time grid at which wave is computed

		if np.abs(time_grid[0]) > np.abs(time_full[0]/(m1+m2)):
			warnings.warn("The chosen start frequency of {} Hz is too short for the chosen grid length of {} s/M_sun".format(fstart_, time_grid[0]))

			#computing waves to the chosen std grid and saving to file
		alpha_ = np.interp(time_grid, time_full/(m1+m2), alphaT.data.data)
		cosbeta_ = np.interp(time_grid, time_full/(m1+m2), cosbetaT.data.data)
		#alpha_ = alpha_ - alpha_[0] #TODO: understand whether alpha needs to be set to zero at the end of the grid!

		if not True:
			#TODO: remove this shit!!
			fig, axes = plt.subplots(2,1, sharex = True)
			plt.suptitle('q, s1, s2 = {:5.3f}, {:5.3f}, {:5.3f}\nt1_infty, t2_infty = {:5.3f}, {:5.3f}\nt1, t2 = {:5.3f}, {:5.3f}'.format(q, s1, s2, t1_infty, t2_infty, t1, t2))
			axes[0].plot(time_grid, alpha_)
			axes[1].plot(time_grid, cosbeta_)
			plt.xlabel(r'Time (s/M_sun)')
			plt.tight_layout()
			plt.show()

		to_save = np.concatenate([temp_theta, alpha_, cosbeta_])[None,:] #(1,D)
		np.savetxt(filebuff, to_save)
	
		del hlmQAT, alphaT, cosbetaT, gammaT
		del to_save

	filebuff.close()
	return
	
	
class residual_network_angles(tf.keras.Model):
	"""1D convolutional autoencoder for the Euler angles"""
	def __init__(self, input_dim, output_dim, layers, residual_layers = None, n_residual_comps = None, activation = 'sigmoid'):
		super(residual_network_angles, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.alpha_res = 1.		
	
		assert isinstance(layers, list), "Argument layers must be a list"
		
		layers_regression = [tf.keras.Input((input_dim,))]
		layers_regression.extend([tf.keras.layers.Dense(units= l, activation=activation) for l in layers])
		layers_regression.append(tf.keras.layers.Dense(units= output_dim))
		
		self.regression = tf.keras.Sequential(layers_regression)
		
		if residual_layers:
			if n_residual_comps is None:
				n_residual_comps = output_dim
			self.ids_res = slice(0, n_residual_comps)
			self.ids_non_res = slice(n_residual_comps, output_dim)
		
			assert isinstance(residual_layers, list), "Argument residual_layers must be a list"
			layers_residual_regression = [tf.keras.Input((input_dim,))]
			layers_residual_regression.extend([tf.keras.layers.Dense(units= l, activation=activation) for l in residual_layers])
			layers_residual_regression.append(tf.keras.layers.Dense(units= n_residual_comps))
		
			self.residual_regression = tf.keras.Sequential(layers_residual_regression)
			self.concat_layer = tf.keras.layers.Concatenate()
			self.reshape_layer = tf.keras.layers.Reshape((1,))
			self.batch_norm = tf.keras.layers.BatchNormalization()
		else:
			self.residual_regression = None
		
	def __call__(self, x, training = False):

		y = self.regression(x, training)

		if not self.residual_regression:
			return y

		res = self.residual_regression(x, training)
		res = res*self.alpha_res + y[:,self.ids_res]

		return self.concat_layer([res[:,self.ids_res], y[:,self.ids_non_res]])
	
	def compile(self, loss, optimizer, loss_res = None, optimizer_res = None, **kwargs):

		self.regression.compile(loss = loss, optimizer = optimizer, **kwargs)
		if self.residual_regression: self.residual_regression.compile(loss = loss_res, optimizer = optimizer_res, **kwargs)
		
	
	def fit(self, x, y, validation_data = None, **kwargs):
		
		history = self.regression.fit(x, y, validation_data = validation_data, **kwargs)
		history_residual = None
		
		if self.residual_regression:
		
			print("#### Starting residual training")
			
			y_res = y - self.regression(x)
			
			self.alpha_res = np.sqrt(tf.math.reduce_variance(y_res))
			
			print(self.alpha_res)
			
			y_res = y_res[:,self.ids_res]/self.alpha_res
			
			if validation_data:
				x_val, y_val = validation_data
				y_val_res = (y_val - self.regression(x_val))[:,self.ids_res]/self.alpha_res
			
			history_residual = self.residual_regression.fit(x, y_res, validation_data = (x_val, y_val_res), **kwargs)

			#plt.scatter(x[:,0], self.residual_regression(x).numpy()[:,0], s = 2)		
			#plt.scatter(x[:,0], y_res.numpy()[:,0], s = 2)
			#plt.show()
		
		return history, history_residual
			
		
		
		










	



