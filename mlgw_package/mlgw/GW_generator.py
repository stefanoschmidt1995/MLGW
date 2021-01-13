"""
Module GW_generator.py
======================
	Definition of class MLGW_generator and mode_generator.
	- GW_generator bounds together many mode_generator and builds the complete WF as a sum of different modes.	
	- mode_generator generates a specific l,m mode of GW signal of a BBH coalescence when given orbital parameters of the BBH.
		The model performs the regression:
			theta = (q,s1,s2) ---> g ---> A, ph = W g
		First regression is done by a MoE model; the second regression is a PCA model. Some optional parameters can be given to specify the observer position.
		It makes use of modules EM_MoE.py and ML_routines.py for an implementation of a PCA model and a MoE fitted by EM algorithm.
"""
#################

import os
import sys
import warnings
import numpy as np
import ast
import inspect
sys.path.insert(1, os.path.dirname(__file__)) 	#adding to path folder where mlgw package is installed (ugly?)
from EM_MoE import *			#MoE model
from ML_routines import *		#PCA model
from scipy.special import factorial as fact

warnings.simplefilter("always", UserWarning) #always print a UserWarning message ??

#############DEBUG PROFILING
try:
	from line_profiler import LineProfiler

	def do_profile(follow=[]):
		def inner(func):
			def profiled_func(*args, **kwargs):
				try:
					profiler = LineProfiler()
					profiler.add_function(func)
					for f in follow:
						profiler.add_function(f)
					profiler.enable_by_count()
					return func(*args, **kwargs)
				finally:
					profiler.print_stats()
			return profiled_func
		return inner
except:
	pass

################# GW_generator class
def list_models(print_out = True):
	"""
list_models
===========
	Print to screen the models available by default in the relevant folder.
	Input:
		print_out	whether the output should be printed (if False, it is returned as a string)
	"""
	if not print_out:
		to_return = ""
	else:
		to_return = None
	models = os.listdir(os.path.dirname(inspect.getfile(list_models))+"/TD_models")
	models.sort()
	for model in models:
		folder = os.path.dirname(inspect.getfile(list_models))+"/TD_models/"+model
		files = os.listdir(folder)
		if "README" in files:
			with open(folder+"/README") as f:
				contents = f.read()
			temp_dict = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
			try:
				temp_dict = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
				description = temp_dict['description']
				description = ": "+ description
			except:
				description = ""
		else:
			description = ""
		model = model.replace("_"," ")
		if print_out:
			print(model+description)
		else:
			to_return += model+description+"\n"

	return to_return


class GW_generator(object):
	"""
GW_generator
============
	This class holds a collection of mode_generator istances and provides the code to generate a full GW signal with the higher modes, with the ML model.
	It builds the WF as:
		h = h_+ + i h_x = sum_l sum_m Y_lm H_lm(t)
	The model shall be saved in a single folder, which collects a different subfolder "lm" for each mode to generate. Each mode is independent from the others and modes can be added at will.
	Some default models are already included in the package.
	"""

	def __init__(self, folder = 0):
		"""
	__init__
	========
		Initialise class by loading the modes from file.
		A number of pre-fitted models for the modes are released: they can be loaded with folder argument by specifying an integer index (default 0. They are all saved in "__dir__/TD_models/model_(index_given)". A list of the available models can be listed with list models().
		Each model is composed by many modes. Each mode is represented by a mode_generator istance, each saved in a different folder within the folder.
		Input:
			folder		address to folder in which everything is kept (if None, models must be loaded manually with load())
		"""
		self.modes = [] #list of modes (classes mode_generator)
		self.id_22 = None #index of the 22 mode

		if folder is not None:
			if type(folder) is int:
				int_folder = folder
				folder = os.path.dirname(inspect.getfile(GW_generator))+"/TD_models/model_"+str(folder)
				if not os.path.isdir(folder):
					raise RuntimeError("Given value {0} for pre-fitted model is not valid. Available models are:\n{1}".format(str(int_folder), list_models(False)))
			self.load(folder)
		return

	def __extract_mode(self, folder):
		"""
	__extract_mode
	============
		Given a folder name, it extract (if present) the tuple of the mode the folder contains.
		Each mode folder must start with "lm".
		Input:
			folder		folder holding a mode
		Output:
			mode 	tuple for the mode (None if no mode is found in name)
		"""
		name = os.path.basename(folder)
		l = name[0]	
		m = name[1]
		try:
			lm = (int(l), int(m))
			assert l>=m
		except:
			warnings.warn('Folder {}: name not recognized as a valid mode - skipping its content'.format(name))
			return None
		return lm

	def load(self, folder):
		"""
	load
	====
		Loads the GW generator by loading the different mode_generator classes.
		Each mode is loaded from a dedicated folder in the given folder of the model.
		An optional README files holds some information about the model.
		Input:
			address to folder in which everything is kept
		"""
		if not os.path.isdir(folder):
			raise RuntimeError("Unable to load folder "+folder+": no such directory!")

		if not folder.endswith('/'):
			folder = folder + "/"
		print("Loading model from: ", folder)
		file_list = os.listdir(folder)
		
		if 'README' in file_list:
			with open(folder+"README") as f:
				contents = f.read()
			self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
			try:
				self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
				assert type(self.readme) == dict
			except:
				warnings.warn("README file is not a valid dictionary: entry ignored")
				self.readme = None
			file_list.remove('README')
		else:
			self.readme = None

		#loading modes
		for mode in file_list:
			lm = self.__extract_mode(folder+mode)
			if lm is None:
				continue
			if lm == (2,2): #saving index of 22 mode
				self.id_22 = len(self.modes)
			self.modes.append(mode_generator(lm, folder+mode)) #loads mode_generator
			print('    Loaded mode {}'.format(lm))

		return

	def __get_precessing_params(self, m1, m2, s1, s2):
		"""
	__get_precessing_params
	=======================
		Given the two masses and (dimensionless) spins, it computes the angles between the two spins and the orbital angular momentum (theta1, theta2) and the angle between the projections of the two spins onto the orbital plane (delta_Phi). Please, refer to eqs. (1-4) of https://arxiv.org/abs/1605.01067.
		Spins must be in the L frame, in which the orbital angular momentum has only the z compoment; they are evaluated when at a given orbital frequency f = 20 Hz (????????????????????????? check better here)
		Returns the six variables (i.e. q, chi1, chi2, theta1, theta2, delta_Phi) useful for reconstructing precession angles alpha and beta with the NN.
		Assumes that always (m1>m2)
		Inputs:
			m1 ()/(N,)			mass of BH 1
			m2 ()/(N,)			mass of BH 2
			s1 (3,)/(N,3)		(dimensionless) spin components of BH 1			
			s2 (3,)/(N,3)		(dimensionless) spin components of BH 2
		Ouput:
			q ()/(N,)				mass ratio (>1)
			chi1 ()/(N,)			dimensionless spin 1 magnitude
			chi2 ()/(N,)			dimensionless spin 1 magnitude
			theta1 ()/(N,)			angle between spin 1 and the orbital angular momentum
			theta2 ()/(N,)			angle between spin 2 and the orbital angular momentum
			delta_Phi ()/(N,)		angle between the projections of the two spins onto the orbital plane
		"""
		if s1.ndim == s2.ndim == 1:
			if not (m1.ndim == m2.ndim == 0):
				raise RuntimeError("Shape of m1,m2 is inconsistent with shape of spins: expected 0 dim array.")
			s1 = s1[None,:] #(1,3)
			s2 = s2[None,:]	#(1,3)
			m1 = m1[None]
			m2 = m2[None]
			squeeze = True
		else:
			squeeze = False
			
		if not (s1.shape[1] == s2.shape[1] ==3):
			raise RuntimeError("Spin vectors must have 3 components! Instead they have {} and {} components".format(s1.shape[1], s2.shape[1]))
		
		
		chi1 = np.linalg.norm(s1,axis = 1) #(N,)
		chi2 = np.linalg.norm(s2,axis = 1) #(N,)
		theta1 = np.arccos(s1[:,2]/chi1)
		theta2 = np.arccos(s2[:,2]/chi2)
		L = np.array([0.,0.,1.])
		
		plane_1 = np.array([s1[:,1], -s1[:,0],0.]) #s1xL
		plane_2 = np.array([s2[:,1], -s2[:,0],0.])
		sign = np.sign(np.cross(plane_1,plane_2)[:,2]) #(N,) #computing the sign
		
		plane_1 = plane_1 / np.linalg.norm(plane_1, axis =1) #(N,3)
		plane_2 = plane_2 / np.linalg.norm(plane_2, axis =1) #(N,3)
		delta_Phi = np.arccos(np.sum(np.multiply(plane_1,plane_2), axis =1)) #(N,)

		delta_Phi = np.multiply(delta_Phi, sign) #(N,) #setting the right sign
		
		return m1/m2, chi1, chi2, theta1, theta2, delta_Phi
		
	def summary(self, filename = None):
		"""
	summary
	=======
		Prints to screen a summary of the model currently used.
		If filename is given, output is also redirected to file.
		Input:
			filename	if not None, redirects the output to file
		"""
		output = "###### Summary for MLGW model ######\n"
		if self.readme is not None:
			keys = list(self.readme.keys())
			if "description" in keys:
				output += self.readme['description'] + "\n"
				keys.remove('description')
			for k in keys:
				output += "   "+k+": "+self.readme[k] + "\n"

		if type(filename) is str:
			text_file = open(filename, "a")
			text_file.write(output)
			text_file.close()
			return
		elif filename is not None:
			warnings.warn("Filename must be a string! "+str(type(filename))+" given. Output is redirected to standard output." )
		print(output)
		return

	def list_modes(self, print_screen = False):
		"""
	list_modes
	==========
		Returns a list of the available modes.
		If print_screen is True, it also prints to screen
		Output:
			mode_list	list with the available modes
		"""
		mode_list = []
		for mode in self.modes:
			mode_list.append(mode.lm())
		if print_screen: print(mode_list)
		return mode_list


	def __call__(self, t_grid, m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano):
		"""
	__call__
	========
		Generates a WF according to the model. It makes all the required preprocessing to include wave dependance on the full 14 parameters space of the GW forms. It outputs the plus cross polarization of the WF.
		All the available modes are employed to build the WF.
		The WF is shifted such that the peak of the 22 mode is placed at t=0. If the reference phase is 0, the phase of the 22 mode is 0 at the beginning of the time grid.
		Note that the dependence on the longitudinal ascension node, the eccentricity, the mean periastron anomaly and the orthogonal spin components is not currently implemented and it is mainted for compatibility with lal.
		Input:
			t_grid	(N_grid,)		Grid of (physical) time points to evaluate the wave at
			m1	()/(N,)				Mass of BH 1
			m2	()/(N,)				Mass of BH 2
			spin1_x/y/z	()/(N,)		Each variable represents a spin component of BH 1
			spin2_x/y/z				Each variable represents a spin component of BH 2
			D_L	()/(N,)				Luminosity distance
			i ()/(N,)				Inclination
			phi_0 ()/(N,)			Reference phase for the wave
			long_asc_nodes ()/(N,)	Logitudinal ascentional nodes (currently not implemented)
			eccentricity ()/(N,)	Eccentricity of the orbit (currently not implemented)
			mean_per_ano ()/(N,)	Mean periastron anomaly (currently not implemented)
		Ouput:
			h_plus, h_cross (1,D)/(N,D)		desidered polarizations
		"""
		theta = np.column_stack((m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano)) #(N,D)
		return self.get_WF(theta, t_grid= t_grid, modes = (2,2))


	def get_WF(self, theta, t_grid, modes = (2,2) ):
		"""
	get_WF
	======
		Generates a WF according to the model. It makes all the required preprocessing to include wave dependance on the full 14 parameters space of the GW forms. It outputs the plus cross polarization of the WF.
		All the available modes are employed to build the WF.
		The WF is shifted such that the peak of the 22 mode is placed at t=0. If the reference phase is 0, the phase of the 22 mode is 0 at the beginning of the time grid.
		If no geometrical variables are given, it is set by default D_L = 1 Mpc, iota = phi_0 = 0.
		It accepts data in one of the following layout of D features:
			D = 3	[q, spin1_z, spin2_z]
			D = 4	[m1, m2, spin1_z, spin2_z]
			D = 5	[m1, m2, spin1_z , spin2_z, D_L]
			D = 6	[m1, m2, spin1_z , spin2_z, D_L, inclination]
			D = 7	[m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
			D = 14	[m1, m2, spin1 (3,), spin2 (3,), D_L, inclination, phi_0, long_asc_nodes, eccentricity, mean_per_ano]
		In the D = 3 layout, the total mass is set to 20 M_sun by default.
		Warning: last layout (D=14) is made only for compatibility with lalsuite software. The implemented variables are those in D=7 layout; the other are dummy variables and will not be considered.
		Unit of measures:
			[mass] = M_sun
			[D_L] = Mpc
			[spin] = adimensional
		User might choose which modes are to be included in the WF.
		Input:
			theta (N,D)		source parameters to make prediction at
			t_grid (D',)	a grid in (reduced) time to evaluate the wave at (uses np.interp)
			modes			list of modes employed for building the WF (if None, every mode available is employed)
		Ouput:
			h_plus, h_cross (D,)/(N,D)		desidered polarizations (if it applies)
		"""
		if isinstance(modes,tuple) and modes != (2,2):
			modes = [modes]
		theta = np.array(theta) #to ensure user theta is copied into new array
		if theta.ndim == 1:
			to_reshape = True #whether return a one dimensional array
			theta = theta[np.newaxis,:] #(1,D)
		else:
			to_reshape = False
		
		D= theta.shape[1] #number of features given
		if D <3:
			raise RuntimeError("Unable to generata WF. Too few parameters given!!")
			return

			#creating a standard theta vector for __get_WF
		if D==3:
			new_theta = np.zeros((theta.shape[0],7))
			new_theta[:,4] = 1.
			new_theta[:,[2,3]] = theta[:,[1,2]] #setting spins
			new_theta[:,[0,1]] = [theta[:,0]*20./(1+theta[:,0]), 20./(1+theta[:,0])] #setting m1,m2 with M = 20
			theta = new_theta #(N,7)

		if D>3 and D!=7:
			new_theta = np.zeros((theta.shape[0],7))
			new_theta[:,4] = 1.
			if D== 14:
				if np.any(np.column_stack((theta[:,2:4], theta[:,5:7])) != 0):
					warnings.warn("Given nonzero spin_x/spin_y components. Model currently supports only spin_z component. Other spin components are ignored")
				indices = [0,1,4,7,8,9,10]
				indices_new_theta = range(7)
			else:
				indices = [i for i in range(D)]
				indices_new_theta = indices


				#building vector to keep standard layout for __get_WF
			new_theta[:, indices_new_theta] = theta[:,indices]
			theta = new_theta #(N,7)

		if np.any(np.logical_and(theta[:,[2,3]]>=1,theta[:,[2,3]]<=-1)):
			raise ValueError("Wrong value for spins, please set a value in range [-1,1]")

			#generating waves and returning to user
		h_plus, h_cross = self.__get_WF(theta, t_grid, modes) #(N,D)
		if to_reshape:
			return h_plus[0,:], h_cross[0,:] #(D,)
		return h_plus, h_cross #(N,D)

	def __get_twisted_modes(self, theta, t_grid, modes):
		"""
	__get_twisted_modes
	======================
		Return the twisted modes of the model, evaluated in the given time grid.
		The twisted mode depends on angles alpha, beta, gamma and it is performed as in eqs. (17-20) in https://arxiv.org/abs/2005.05338
		The function returns the real and imaginary part of the twisted mode.
		Each mode is aligned s.t. the peak of the (untwisted) 22 mode is at t=0
		Input:
			theta (N,8)/(8,)	source parameters to make prediction at (m1, m2, s1 (3,), s2 (3,))
			t_grid (D',)		a grid in (reduced) time to evaluate the wave at (uses np.interp)
			modes				list (or a single tuple) of modes to be returned (if None, every mode available is employed)
		Output:
			real, imag (N, D', K)	real and imaginary part of the K modes required by the user (if mode is a tuple, no third dimension)
		"""


	def __get_WF(self, theta, t_grid, modes):
		"""
	__get_WF
	========
		Generates the waves in time domain, building it as a sum of modes weighted by spherical harmonics. Called by get_WF.
		Accepts only input features as [q,s1,s2] or [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0].
		Input:
			theta (N,D)		source parameters to make prediction at (D=7)
			t_grid (D',)	a grid in (reduced) time to evaluate the wave at (uses np.interp)
			modes			list of modes employed for building the WF (if None, every mode available is employed)
		Ouput:
			h_plus, h_cross (D,)/(N,D)		desidered polarizations (if it applies)
		"""
		D= theta.shape[1] #number of features given
		assert D == 7

			#computing amplitude prefactor
		prefactor = 4.7864188273360336e-20 # G/c^2*(M_sun/Mpc)
		m_tot_us = theta[:,0] + theta[:,1]	#total mass in solar masses for the user  (N,)
		amp_prefactor = prefactor*m_tot_us/theta[:,4] # G/c^2 (M / d_L) 

		h_plus = np.zeros((theta.shape[0],t_grid.shape[0]))
		h_cross = np.zeros((theta.shape[0],t_grid.shape[0]))

			#if only mode 22 is required, it is treated separately for speed up
		if modes == (2,2):# or modes == [(2,2)]:
			amp_22, ph_22 = self.modes[self.id_22].get_mode(theta[:,:4], t_grid, out_type = "ampph")
			amp_22 =  np.sqrt(5/(4.*np.pi))*np.multiply(amp_22.T, amp_prefactor).T #G/c^2*(M_sun/Mpc) nu *(M/M_sun)/(d_L/Mpc)
				#setting spherical harmonics by hand
			c_i = np.cos(theta[:,5]) #(N,)
			h_p = np.multiply(np.multiply(amp_22.T,np.cos(ph_22.T+2.*theta[:,6])), 0.5*(1+np.square(c_i)) ).T
			h_c = np.multiply(np.multiply(amp_22.T,np.sin(ph_22.T+2.*theta[:,6])), c_i ).T
			return h_p, h_c

		if modes is None:
			modes = self.list_modes()

		for mode in self.modes:	
			if mode.lm() not in modes: #skipping a non-necessary or non-existing mode
				continue
			#print("got modes {}".format(mode.lm()))
			amp_lm, ph_lm = mode.get_mode(theta[:,:4], t_grid, out_type = "ampph")
			amp_lm =  np.multiply(amp_lm.T, amp_prefactor).T #G/c^2*(M_sun/Mpc) nu *(M/M_sun)/(d_L/Mpc)
				# setting spherical harmonics: amp, ph, D_L,iota, phi_0
			h_lm_real, h_lm_imag = self.__set_spherical_harmonics(mode.lm(), amp_lm, ph_lm, theta[:,5], theta[:,6])
			h_plus = h_plus + h_lm_real
			h_cross = h_cross + h_lm_imag

		return h_plus, h_cross

	def get_modes(self, theta, t_grid, modes = (2,2), out_type = "ampph"):
		"""
	get_modes
	=========
		Return the modes in the model, evaluated in the given time grid.
		It can return amplitude and phase or the real and imaginary part.
		Each mode is aligned s.t. the peak of the 22 mode is at t=0
		Input:
			theta (N,D)/(D,)	source parameters to make prediction at (D = 3,4)
			t_grid (D',)		a grid in (reduced) time to evaluate the wave at (uses np.interp)
			modes				list of modes to be returned (if None, every mode available is employed)
			out_type			whether amplitude and phase ("ampph") or real and imaginary part ("realimag") shall be returned
		Output:
			amp, ph (N, D', K)		amplitude and phase of the K modes required by the user (if K =1, no third dimension)
			real, imag (N, D', K)	real and imaginary part of the K modes required by the user (if K =1, no third dimension)
		"""
		if out_type not in ["realimag", "ampph"]:
			raise ValueError("Wrong output type chosen. Expected \"realimag\", \"ampph\", given \""+out_type+"\"")

		theta = np.array(theta)
		if isinstance(modes,tuple): #it means that the last dimension should be deleted
			modes = [modes]
			remove_last_dim = True
		else:
			remove_last_dim = False

		try:
			K = len(modes)
		except:
			K = len(self.modes)
		if modes is None:
			modes = self.list_modes()

		if theta.ndim == 1:
			theta = theta[None,:]
			remove_first_dim = True
		else:
			remove_first_dim = False
		if theta.shape[1] == 7:
			theta = theta[:,:4]

		res1 = np.zeros((theta.shape[0],t_grid.shape[0],K))
		res2 = np.zeros((theta.shape[0],t_grid.shape[0],K))

		for mode in self.modes:	
			if mode.lm() not in modes: #skipping a non-necessary mode
				continue
			else: #computing index to save the mode at
				i = modes.index(mode.lm())
			#print("got modes {}".format(mode.lm()))
			res1[:,:,i], res2[:,:,i] = mode.get_mode(theta, t_grid, out_type = out_type)

		if remove_last_dim:
			res1, res2 = res1[...,0], res2[...,0] #(N,D)
		if remove_first_dim:
			res1, res2 = res1[0,...], res2[0,...] #(D,)/(D,K)
		return res1, res2
		

	def __set_spherical_harmonics(self, mode, amp, ph, iota, phi_0):
		"""
	__set_spherical_harmonics
	=========================
		Given amplitude and phase of a mode, it returns the quantity [Y_lm*A*e^(i*ph)+ Y_l-m*A*e^(-i*ph)]. This amounts to the contribution to the WF given by the mode.
		We parametrize: Y_lm(iota, phi_0) = d_lm(iota) * exp(i*m*phi_0)
		It also include negative m modes with: h_lm = (-1)**l h*_lmm (https://arxiv.org/abs/1501.00918 eq. (5) )
		Input:
			mode			(l,m) of the current mode
			amp, ph (N,D)	amplitude and phase of the WFs (as generated by the ML)
			iota (N,)		inclination for each wave
			phi_0 (N,)		reference phase for each wave
		Output:
			h_lm_real, h_lm_imag (N,D)	processed strain, with d, iota, phi_0 dependence included.
		"""
		l,m = mode
			#computing the iota dependence of the WF
		d_lm = self.__get_Wigner_d_function((l,m),iota) #(N,)
		d_lmm = self.__get_Wigner_d_function((l,-m),iota) #(N,)
		const = np.sqrt( (2.*l+1.)/(4.*np.pi) )
		parity = np.power(-1,l) #are you sure of that? apparently yes...

		h_lm_real = np.multiply(np.multiply(amp.T,np.cos(ph.T+m*phi_0)), const*(d_lm + parity * d_lmm) ).T #(N,D)
		h_lm_imag = np.multiply(np.multiply(amp.T,np.sin(ph.T+m*phi_0)), const*(d_lm - parity * d_lmm) ).T #(N,D)

		return h_lm_real, h_lm_imag

	def __get_Wigner_d_function(self, mode, iota):
		"""
	__get_Wigner_d_function
	=======================
		Return Wigner d function. This encodes the inclination dependent part d_lm of the spherical harmonics Y_lm, according to:
			Y_lm(iota, phi_0) = d_lm(iota) * exp(i*m*phi_0)
		See https://arxiv.org/pdf/0709.0093.pdf for an explicit expression.
		Input:
			mode			(l,m) of the current mode
			iota (N,)		inclination for to compute the amplitude at
		Output:
			d_lm (N,)		Amplitude of the spherical harmonics d_lm(iota)
		"""
		l,m = mode
		s = 2
		d_lm = np.zeros(iota.shape) #(N,)
    
		cos_i = np.cos(iota*0.5) #(N,)
		sin_i = np.sin(iota*0.5) #(N,)
    
			#starting computation (sloooow??)
		ki = max(0, m-s)
		kf = min(l+m, l-s)
		#print(ki,kf)
    	
		for k in range(ki,kf+1):
			norm = fact(k) * fact(l+m-k) * fact(l-s-k) * fact(s-m+k) #normalization constant
			d_lm = d_lm +  (pow(-1.,k) * np.power(cos_i,2*l+m-s-2*k) * np.power(sin_i,2*k+s-m) ) / norm

		const = np.sqrt(fact(l+m) * fact(l-m) * fact(l+s) * fact(l-s))
		return const*d_lm
	
	def get_mode_obj(self, mode):
		"""
	get_mode_obj
	============
		Returns an instance of class mode_generator which hold the ML model for the required mode.
		Input:
			mode		(l,m) of the required mode
		Output:
			mode_obj	istance of mode_generator
		"""
		for mode_ in self.modes:	
			if mode_.lm() == mode: #check if it is the correct mode
				return mode_
		return None

class mode_generator(object):
	"""
mode_generator
==============
	This class holds all the parts of ML models and acts as single (l,m) mode generator. Model is composed by a PCA model to reduce dimensionality of a WF datasets and by several MoE models to fit PCA in terms of source parameters. WFs can be generated both in time domain and frequency domain.
	Everything is hold in a PCA model (class PCA_model defined in ML_routines) and in two lists of MoE models (class MoE_model defined in EM_MoE). All models are loaded from files in a folder given by user. Files must be named exactly as follows:
		amp(ph)_exp_#		for amplitude (phase) of expert model for PCA component #
		amp(ph)_gat_#		for amplitude (phase) of gating function for PCA component #
		amp(ph)_feat		for list of features to use for MoE models
		amp(ph)_PCA_model	for PCA model for amplitude (phase)
		times/frequencies	file holding grid points at which waves generated by PCA are evaluated
	No suffixes shall be given to files.
	The class doesn't implement methods for fitting: it only provides a useful tool to gather them.
	"""
	def __init__(self, mode, folder = None):
		"""
	__init__
	========
		Initialise class by loading models from file.
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model; times/frequencies}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		A compulsory file times must hold a list of grid points at which the generated ML wave is evaluated.
		An optional README file holds more information about the model (in the format of a dictionary).
		Input:
			mode		tuple (l,m) of the mode which the model refers to
			folder		address to the folder in which everything is kept (if None, models must be loaded manually with load())
		"""
		self.times = None
		self.mode = mode #(l,m) tuple
		self.readme = None	

		if folder is not None:
			self.load(folder, verbose = False)
		return
	
	def lm(self):
		"""
	lm
	==
		Returns the (l,m) index of the mode.
		"""
		return self.mode

	def __read_features(self, feat_file):	
		"""
	__read_features
	===============
		Extract the features of a MoE regression from a given file.
		Input:
			feat_file	path to file
		Output:
			feat_list	list of features
		"""
		f = open(feat_file, "r")
		feat_list = f.readlines()
		for i in range(len(feat_list)):
			feat_list[i] = feat_list[i].rstrip()
		f.close()
		return feat_list

	def load(self, folder, verbose = False):
		"""
	load
	====
		Builds up all the models from given folder.
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		It loads time vector.
		If given, it loads as a dictionary the README file. Dictionary should include entries (all optional): 'description', 'mode', 'train model', 'q range', 's1 range', 's2 range'.
		Input:
			folder		address to folder in which everything is kept
			verbose		whether to print output
		"""
		if not os.path.isdir(folder):
			raise RuntimeError("Unable to load folder "+folder+": no such directory!")

		if verbose: #define a verboseprint if verbose is true
			def verboseprint(*args, **kwargs):
				print(*args, **kwargs)
		else:
			verboseprint = lambda *a, **k: None # do-nothing function

		if not folder.endswith('/'):
			folder = folder + "/"
		verboseprint("Loading model for "+str(self.mode)+" from: ", folder)
		file_list = os.listdir(folder)

			#loading PCA
		self.amp_PCA = PCA_model()
		self.amp_PCA.load_model(folder+"amp_PCA_model")
		self.ph_PCA = PCA_model()
		self.ph_PCA.load_model(folder+"ph_PCA_model")

		verboseprint("  Loaded PCA model for amplitude with ", self.amp_PCA.get_V_matrix().shape[1], " PC")
		verboseprint("  Loaded PCA model for phase with ", self.ph_PCA.get_V_matrix().shape[1], " PC")

			#loading features
		self.amp_features = self.__read_features(folder+"amp_feat")
		self.ph_features = self.__read_features(folder+"ph_feat")

		verboseprint("  Loaded features for amplitude: ", self.amp_features)
		verboseprint("  Loaded features for phase: ", self.ph_features)
	
			#loading MoE models
		verboseprint("  Loading MoE models")
			#amplitude
		self.MoE_models_amp = []
		k = 0
		while "amp_exp_"+str(k) in file_list and  "amp_gat_"+str(k) in file_list:
			self.MoE_models_amp.append(MoE_model(3+len(self.amp_features),1))
			self.MoE_models_amp[-1].load(folder+"amp_exp_"+str(k),folder+"amp_gat_"+str(k))
			verboseprint("    Loaded amplitude model for comp: ", k)
			k += 1
		
			#phase
		self.MoE_models_ph = []
		k = 0
		while "ph_exp_"+str(k) in file_list and  "ph_gat_"+str(k) in file_list:
			self.MoE_models_ph.append(MoE_model(3+len(self.ph_features),1))
			self.MoE_models_ph[-1].load(folder+"ph_exp_"+str(k),folder+"ph_gat_"+str(k))
			verboseprint("    Loaded phase model for comp: ", k)
			k += 1

		if "times" in file_list:
			verboseprint("  Loaded time vector")
			self.times = np.loadtxt(folder+"times")
		else:
			raise RuntimeError("Unable to load model: no time vector given!")

		if 'README' in file_list:
			with open(folder+"README") as f:
				contents = f.read()
			self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
			try:
				self.readme = ast.literal_eval(contents) #dictionary holding some relevant information about the model loaded
				assert type(self.readme) == dict
			except:
				warnings.warn("README file is not a valid dictionary: entry ignored")
				self.readme = None
		else:
			self.readme = None

		np.matmul(np.zeros((2,2)),np.ones((2,2))) #this has something to do with a speed up of matmul. Once it is called once, matmul gets much faster!
		return

	def MoE_models(self, model_type, k_list=None):
		"""
	MoE_models
	==========
		Returns the MoE model(s).
		Input:
			model_type		"amp" or "ph" to state which MoE models shall be returned
			k_list []		index(indices) of the model to be returned (if None all models are returned)
		Output:
			models []	list of models to be returned
		"""
		if k_list is None:
			k_list = range(self.K)
		if model_type == "amp":
			return self.MoE_models_amp[k]
		if model_type == "ph":
			return self.MoE_models_ph[k]
		return None

	def PCA_models(self, model_type):
		"""
	PCA_models
	==========
		Returns the PCA model.
		Input:
			model_type		"amp" or "ph" to state which PCA model shall be returned
		Output:
			
		"""
		if model_type == "amp":
			return self.amp_PCA
		if model_type == "ph":
			return self.ph_PCA
		return None

	def summary(self, filename = None):
		"""
	summary
	=======
		Prints to screen a summary of the model currently used.
		If filename is given, output is redirected to file.
		Input:
		Output:
			filename	if not None, redirects the output to file
		"""
		amp_exp_list = [str(model.get_iperparams()[1]) for model in self.MoE_models_amp]
		ph_exp_list = [str(model.get_iperparams()[1]) for model in self.MoE_models_ph]

		output = "###### Summary for MLGW model ######\n"
		if self.readme is not None:
			keys = list(self.readme.keys())
			if "description" in keys:
				output += self.readme['description'] + "\n"
				keys.remove('description')
			for k in keys:
				output += "   "+k+": "+self.readme[k] + "\n"

		output += "   Grid size: "+str(self.amp_PCA.get_PCA_params()[0].shape[0]) +" \n"
		output += "   Minimum time: "+str(np.abs(self.times[0]))+" s/M_sun\n"
			#amplitude summary
		output += "   ## Model for Amplitude \n"
		output += "      - #PCs:          "+str(self.amp_PCA.get_PCA_params()[0].shape[1])+"\n"
		output += "      - #Experts:      "+(" ".join(amp_exp_list))+"\n"
		output += "      - #Features:     "+str(self.MoE_models_amp[0].get_iperparams()[0])+"\n"
		output += "      - Features:      "+(" ".join(self.amp_features))+"\n"
			#phase summary
		output += "   ## Model for Phase \n"
		output += "      - #PCs:          "+str(self.ph_PCA.get_PCA_params()[0].shape[1])+"\n"
		output += "      - #Experts:      "+(" ".join(ph_exp_list))+"\n"
		output += "      - #Features:     "+str(self.MoE_models_ph[0].get_iperparams()[0])+"\n"
		output += "      - Features:      "+(" ".join(self.ph_features))+"\n"
		output += "####################################"
	
		if type(filename) is str:
			text_file = open(filename, "a")
			text_file.write(output)
			text_file.close()
			return
		elif filename is not None:
			warnings.warn("Filename must be a string! "+str(type(filename))+" given. Output is redirected to standard output." )
		print(output)
		return

	def get_time_grid(self):
		"""
	get_time_grid
	=============
		Returns the time grid at which the output of the models is evaluated. Grid is in reduced units (s/M_sun).
		Output:
			time_grid (D,)	points in time grid at which all waves are evaluated
		"""
		return self.times


	def get_mode(self, theta, t_grid, out_type = "ampph"):
		"""
	get_mode
	========
		Generates the mode according to the MLGW model.
		hlm(t; theta) = A(t) * exp(1j*phi(t)) 
		The mode is time-shifted such that zero of time is where the 22 mode has a peak.
		It accepts data in one of the following layout of D features:
			D = 3	[q, spin1_z, spin2_z]
			D = 4	[m1, m2, spin1_z, spin2_z]
		Unit of measures:
			[mass] = M_sun
			[spin] = adimensional
		If D = 3, the mode is evalutated at the std total mass M = 20 M_sun
		Output waveforms are returned with amplitude and pahse (out_type = "ampph") or with real and imaginary part (out_type = "realimag").
		Input:
			theta (N,D)			source parameters to make prediction at
			t_grid (D',)		grid in time to evaluate the wave at (uses np.interp)
			out_type (str)		the output to be returned ('ampph', 'realimag')
		Ouput:
			amp, phase (1,D)/(N,D)			desidered amplitude and phase (if it applies)
			hlm_real, hlm_im (1,D)/(N,D)	desidered h_22 components (if it applies)
		"""
		if out_type not in ["realimag", "ampph"]:
			raise ValueError("Wrong output type chosen. Expected \"realimag\", \"ampph\", given \""+out_type+"\"")

		theta = np.array(theta) #to ensure that theta is copied into new array
		if not isinstance(t_grid, np.ndarray): #making sure that t_grid is np.array
			t_grid = np.array(t_grid)

		if theta.ndim == 1:
			to_reshape = True #whether return a one dimensional array
			theta = theta[np.newaxis,:] #(1,D)
		else:
			to_reshape = False
		
		D= theta.shape[1] #number of features given
		if D not in [3,4]:
			raise RuntimeError("Unable to generata mode. Wrong number of BBH parameters!!")
			return

			#checking if grid is ok
		if t_grid.ndim != 1:
			raise RuntimeError("Unable to generata mode. Wrong shape ({}) of time grid!!".format(t_grid.shape))
			return

			#generating waves and returning to user
		res1, res2 = self.__get_mode(theta, t_grid, out_type) #(N,D)
		if to_reshape:
			return res1[0,:], res2[0,:] #(D,)
		return res1, res2 #(N,D)

	#@do_profile(follow=[])
	def __get_mode(self, theta, t_grid, out_type):
		"""
	__get_mode
	==========
		Generates the mode in domain and perform. Called by get_mode.
		Accepts only input features as [q,s1,s2] or [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0].
		Input:
			theta (N,D)			source parameters to make prediction at (D=3 or D=4)
			t_grid (D',)		a grid in time to evaluate the wave at (uses np.interp)
			out_type (str)		the output to be returned ('ampph', 'realimag')
		Output:
			amp, phase (1,D)/(N,D)			desidered amplitude and phase (if it applies)
			hlm_real, hlm_im (1,D)/(N,D)	desidered h_22 components (if it applies)
		"""
		D= theta.shape[1] #number of features given
		assert D in [3,4] #check that the number of dimension is fine

			#setting theta_std & m_tot_us
		if D == 3:
			theta_std = theta
			m_tot_us = 20. * np.ones((theta.shape[0],)) 
		else:
			q = np.divide(theta[:,0],theta[:,1]) #theta[:,0]/theta[:,1] #mass ratio (general) (N,)
			m_tot_us = theta[:,0] + theta[:,1]	#total mass in solar masses for the user
			theta_std = np.column_stack((q,theta[:,2],theta[:,3])) #(N,3)

			to_switch = np.where(theta_std[:,0] < 1.) #holds the indices of the events to swap

				#switching masses (where relevant)
			theta_std[to_switch,0] = np.power(theta_std[to_switch,0], -1)
			theta_std[to_switch,1], theta_std[to_switch,2] = theta_std[to_switch,2], theta_std[to_switch,1]

		amp, ph =  self.get_raw_mode(theta_std) #raw WF (N, N_grid)

			#doing interpolations
			############
		new_amp = np.zeros((amp.shape[0], t_grid.shape[0]))
		new_ph = np.zeros((amp.shape[0], t_grid.shape[0]))

		for i in range(amp.shape[0]):
				#computing the true red grid
			interp_grid = np.divide(t_grid, m_tot_us[i])

				#putting the wave on the user grid
			new_amp[i,:] = np.interp(interp_grid, self.times, amp[i,:], left = 0, right = 0) #set to zero outside the domain
			new_ph[i,:]  = np.interp(interp_grid, self.times, ph[i,:])

				#warning if the model extrapolates outiside the grid
			if (interp_grid[0] < self.times[0]):
				warnings.warn("Warning: time grid given is too long for the fitted model. Set 0 amplitude outside the fitting domain.")

			#amplitude and phase of the mode (maximum of amp at t=0)
		amp = new_amp
		ph = (new_ph.T - new_ph[:,0]).T #phase is zero at the beginning of the WF

		if out_type == 'ampph':
			return amp, ph
		else:
			hlm_real = np.multiply(amp, np.cos(ph))
			hlm_imag = np.multiply(amp, np.sin(ph))
			return hlm_real, hlm_imag

	def get_raw_mode(self, theta):
		"""
	get_raw_mode
	============
		Generates a mode according to the MLGW model with a parameters vector in MLGW model style (params=  [q,s1z,s2z]).
		All waves are evaluated at a luminosity distance of 1 Mpc and inclination 0. They are generated at masses m1 = q * m2 and m2 = 20/(1+q), so that M_tot = 20.
		Grid is the standard one.
		Input:
			theta (N,3)		source parameters to make prediction at
		Ouput:
			amp,ph (N,D)	desidered amplitude and phase
		"""
		rec_PCA_amp, rec_PCA_ph = self.get_red_coefficients(theta) #(N,K)

		rec_amp = self.amp_PCA.reconstruct_data(rec_PCA_amp) #(N,D)
		rec_ph = self.ph_PCA.reconstruct_data(rec_PCA_ph) #(N,D)

		return rec_amp, rec_ph

	#@do_profile(follow=[])
	def get_red_coefficients(self, theta):
		"""
	get_red_coefficients
	====================
		Returns the PCA reduced coefficients, as estimated by the MoE models.
		Input:
			theta (N,3)		source parameters to make prediction at
		Ouput:
			red_amp,red_ph (N,K)	PCA reduced amplitude and phase
		"""
		assert theta.shape[1] == 3

			#adding extra features
		amp_theta = add_extra_features(theta, self.amp_features, log_list = [0])
		ph_theta = add_extra_features(theta, self.ph_features, log_list = [0])

			#making predictions for amplitude
		rec_PCA_amp = np.zeros((amp_theta.shape[0], self.amp_PCA.get_dimensions()[1]))
		for k in range(len(self.MoE_models_amp)):
			if k >= self.amp_PCA.get_dimensions()[1]: break
			rec_PCA_amp[:,k] = self.MoE_models_amp[k].predict(amp_theta)

			#making predictions for phase
		rec_PCA_ph = np.zeros((ph_theta.shape[0], self.ph_PCA.get_dimensions()[1]))
		for k in range(len(self.MoE_models_ph)):
			if k >= self.ph_PCA.get_dimensions()[1]: break
			rec_PCA_ph[:,k] = self.MoE_models_ph[k].predict(ph_theta)

		return rec_PCA_amp, rec_PCA_ph

	def __MoE_gradients(self, theta, MoE_model, feature_list):
		"""
	__MoE_gradients
	===============
		Computes the gradient of a MoE model with basis function expansion at the given value of theta.
		Gradient is computed with the chain rule:
			D_i y= D_j y * D_j/D_i
		where D_j/D_i is the jacobian of the feature augmentation.
		Input:
			theta (N,3)		Values of orbital parameters to compute the gradient at
			MoE_model		A mixture of expert model to make the gradient of
			feature_list	List of features used in data augmentation
		Output:
			gradients (N,3)		Gradients for the model
		"""
			#L = len(feature_list)
		jac_transf = jac_extra_features(theta, feature_list, log_list = [0]) #(N,3+L,3)
		MoE_grads = MoE_model.get_gradient(add_extra_features(theta, feature_list, log_list = [0])) #(N,3+L)
		gradients = np.multiply(jac_transf, MoE_grads[:,:,None]) #(N,3+L,3)
		gradients = np.sum(gradients, axis =1) #(N,3)
		return gradients

	def get_raw_grads(self, theta):
		"""
	get_raw_grads
	=============
		Computes the gradients of the amplitude and phase w.r.t. (q,s1,s2).
		Gradients are functions dependent on time and are evaluated on the internal reduced grid (mode_generator.get_time_grid()).
		Input:
			theta (N,3)		Values of orbital parameters to compute the gradient at
		Output:
			grad_amp (N,D,3)	Gradients of the amplitude
			grad_ph (N,D,3)		Gradients of the phase
		"""
			#computing gradient for the reduced coefficients g
		#amp
		D, K_amp = self.amp_PCA.get_dimensions()
		grad_g_amp = np.zeros((theta.shape[0], K_amp, theta.shape[1])) #(N,K,3)
		for k in range(K_amp):
			grad_g_amp[:,k,:] = self.__MoE_gradients(theta, self.MoE_models_amp[k], self.amp_features) #(N,3)
		#ph
		D, K_ph = self.ph_PCA.get_dimensions()
		grad_g_ph = np.zeros((theta.shape[0], K_ph, theta.shape[1])) #(N,K,3)
		for k in range(K_ph):
			grad_g_ph[:,k,:] = self.__MoE_gradients(theta, self.MoE_models_ph[k], self.ph_features) #(N,3)
		
			#computing gradients
		#amp
		grad_amp = np.zeros((theta.shape[0], D, theta.shape[1])) #(N,D,3)
		for i in range(theta.shape[1]):
			grad_amp[:,:,i] = self.amp_PCA.reconstruct_data(grad_g_amp[:,:,i]) - self.amp_PCA.PCA_params[1] #(N,D)
		#ph
		grad_ph = np.zeros((theta.shape[0], D, theta.shape[1])) #(N,D,3)
		for i in range(theta.shape[1]):
			grad_ph[:,:,i] = self.ph_PCA.reconstruct_data(grad_g_ph[:,:,i]) - self.ph_PCA.PCA_params[1] #(N,D)

		return grad_amp, grad_ph

	def get_mode_grads(self, theta, t_grid, out_type ="realimag"):
		"""
	get_mode_grads
	==============
		Returns the gradient of the mode
			hlm = A exp(1j*phi) = A cos(phi) + i* A sin(phi)
		with respect to theta = (M, q, s1, s2).
		Gradients are evaluated on the user given time grid t_grid.
		It returns the real and imaginary part of the gradients.
		Input:
			theta (N,4)		orbital parameters with format (m1, m2, s1, s2)
			t_grid (D,)		time grid to evaluate the gradients at
			out_type		whether to compute gradients of the real and imaginary part ('realimag') or of amplitude and phase ('ampph')
		Output:
			grad_Re(h) (N,D,4)		Gradients of the real part of the waveform
			grad_Im(h) (N,D,4)		Gradients of the imaginary part of the waveform
		"""
		if out_type not in ["realimag", "ampph"]:
			raise ValueError("Wrong output type chosen. Expected \"realimag\", \"ampph\", given \""+out_type+"\"")

		if theta.shape[1] >= 4:
			theta = theta[:,:4]
		elif theta.shape[1]<4:
			raise ValueError("Wrong input values for theta: expected shape (None,4) [m1,m2,s1,s2]")

			#creating theta_std
		q = np.divide(theta[:,0],theta[:,1]) #theta[:,0]/theta[:,1] #mass ratio (general) (N,)
		m_tot_us = theta[:,0] + theta[:,1]	#total mass in solar masses for the user
		theta_std = np.column_stack((q,theta[:,2],theta[:,3])) #(N,3)
			#switching masses (where relevant)
		to_switch = np.where(theta_std[:,0] < 1.) #holds the indices of the events to swap
		theta_std[to_switch,0] = np.power(theta_std[to_switch,0], -1)
		theta_std[to_switch,1], theta_std[to_switch,2] = theta_std[to_switch,2], theta_std[to_switch,1]

		grad_amp = np.zeros((theta_std.shape[0], len(t_grid), 4))
		grad_ph = np.zeros((theta_std.shape[0], len(t_grid), 4))

		#dealing with gradients w.r.t. (q,s1,s2)
		grad_q_amp, grad_q_ph = self.get_raw_grads(theta_std) #(N,D_std,3)
			#interpolating gradients on the user grid
		for i in range(theta_std.shape[0]):
			for j in range(1,4):
				#print(t_grid.shape,self.times.shape)
				grad_amp[i,:,j] = np.interp(t_grid, self.times * m_tot_us[i], grad_q_amp[i,:,j-1],left = 0, right = 0) #set to zero outside the domain #(D,)
				grad_ph[i,:,j]  = np.interp(t_grid, self.times * m_tot_us[i], grad_q_ph[i,:,j-1]) #(D,)

		#dealing with gradients w.r.t. M
		amp, ph = self.get_mode(theta, t_grid, out_type = "ampph") #true wave evaluated at t_grid #(N,D)
		for i in range(theta_std.shape[0]):
			grad_M_amp = np.gradient(amp[i,:], t_grid) #(D,)
			grad_M_ph = np.gradient(ph[i,:], t_grid) #(D,)
				#don't know why but things work here...
			grad_amp[i,:,0] = - np.multiply(t_grid/m_tot_us[i], grad_M_amp) #(D,)
			grad_ph[i,:,0]  = -np.multiply(t_grid/m_tot_us[i], grad_M_ph) #(D,)

		grad_ph = np.subtract(grad_ph,grad_ph[:,0,None,:]) #unclear... but apparently compulsory
			#check when grad is zero and keeping it
		diff = np.concatenate((np.diff(ph, axis = 1), np.zeros((ph.shape[0],1))), axis =1)
		zero = np.where(diff== 0)
		grad_ph[zero[0],zero[1],:] = 0 #takes care of the flat part after ringdown (gradient there shall be zero!!)


		if out_type == "ampph":
			#switching back spins
			#sure of it???
			grad_amp[to_switch,:,2], grad_amp[to_switch,:,3] = grad_amp[to_switch,:,3], grad_amp[to_switch,:,2]
			grad_ph[to_switch,:,2], grad_ph[to_switch,:,3] = grad_ph[to_switch,:,3], grad_ph[to_switch,:,2]
			return grad_amp, grad_ph
		if out_type == "realimag":
			#computing gradients of the real and imaginary part
			ph = np.subtract(ph.T,ph[:,0]).T
			grad_Re = np.multiply(grad_amp, np.cos(ph)[:,:,None]) - np.multiply(np.multiply(grad_ph, np.sin(ph)[:,:,None]), amp[:,:,None]) #(N,D,4)
			grad_Im = np.multiply(grad_amp, np.sin(ph)[:,:,None]) + np.multiply(np.multiply(grad_ph, np.cos(ph)[:,:,None]), amp[:,:,None])#(N,D,4)

			grad_Re[to_switch,:,2], grad_Re[to_switch,:,3] = grad_Re[to_switch,:,3], grad_Re[to_switch,:,2]
			grad_Im[to_switch,:,2], grad_Im[to_switch,:,3] = grad_Im[to_switch,:,3], grad_Im[to_switch,:,2]
			return grad_Re, grad_Im












