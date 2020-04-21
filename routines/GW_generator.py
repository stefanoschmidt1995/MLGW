"""
Module GW_generator.py
======================
	Definition of class MLGW_generator. The class generates a GW signal of a BBH coalescence when given orbital parameters of the BBH.
	Model performs the regression:
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
	This class holds all the parts of ML models and acts as GW generator. Model is composed by a PCA model to reduce dimensionality of a WF datasets and by several MoE models to fit PCA in terms of source parameters. WFs can be generated both in time domain and frequency domain.
	Everything is hold in a PCA model (class PCA_model defined in ML_routines) and in two lists of MoE models (class MoE_model defined in EM_MoE). All models are loaded from files in a folder given by user. Files must be named exactly as follows:
		amp(ph)_exp_#		for amplitude (phase) of expert model for PCA component #
		amp(ph)_gat_#		for amplitude (phase) of gating function for PCA component #
		amp(ph)_feat		for list of features to use for MoE models
		amp(ph)_PCA_model	for PCA model for amplitude (phase)
		times/frequencies	file holding grid points at which waves generated by PCA are evaluated
	No suffixes shall be given to files.
	The class doesn't implement methods for fitting: it only provides a useful tool to gather them.
	"""
	def __init__(self, folder = 0):
		"""
	__init__
	========
		Initialise class by loading models from file.
		A number of pre-fitted models are released: they can be loaded with folder argument by specifying an integer index (default 0. They are all saved in "__dir__/TD_models/model_(index_given)". The pre-fitted models available can be listed with list models().
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model; times/frequencies}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		A compulsory file times/frequencies must hold a list of grid points at which the generated ML wave is evaluated.
		An optional README file holds more information about the model (in the format of a dictionary).
		Input:
			folder				address to folder in which everything is kept (if None, models must be loaded manually with load())
		"""
		self.times = None
		
		if folder is not None:
			if type(folder) is int:
				int_folder = folder
				folder = os.path.dirname(inspect.getfile(GW_generator))+"/TD_models/model_"+str(folder)
				if not os.path.isdir(folder):
					raise RuntimeError("Given value {0} for pre-fitted model is not valid. Available models are:\n{1}".format(str(int_folder), list_models(False)))
			self.load(folder)
		return

	def load(self, folder):
		"""
	load
	====
		Builds up all the models from given folder.
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		It loads time vector.
		If given it loads as a dictionary the README file. Dictionary should include entries (all optional): 'description', 'train model', 'q range', 's1 range', 's2 range'.
		Input:
			address to folder in which everything is kept
		"""
		if not os.path.isdir(folder):
			raise RuntimeError("Unable to load folder "+folder+": no such directory!")

		if not folder.endswith('/'):
			folder = folder + "/"
		print("Loading model from: ", folder)
		file_list = os.listdir(folder)

			#loading PCA
		self.amp_PCA = PCA_model()
		self.amp_PCA.load_model(folder+"amp_PCA_model")
		self.ph_PCA = PCA_model()
		self.ph_PCA.load_model(folder+"ph_PCA_model")

		print("  Loaded PCA model for amplitude with ", self.amp_PCA.get_V_matrix().shape[1], " PC")
		print("  Loaded PCA model for phase with ", self.ph_PCA.get_V_matrix().shape[1], " PC")

			#loading features
		f = open(folder+"amp_feat", "r")
		self.amp_features = f.readlines()
		for i in range(len(self.amp_features)):
			self.amp_features[i] = self.amp_features[i].rstrip()

		f = open(folder+"ph_feat", "r")
		self.ph_features = f.readlines()
		for i in range(len(self.ph_features)):
			self.ph_features[i] = self.ph_features[i].rstrip()
		
		print("  Loaded features for amplitude: ", self.amp_features)
		print("  Loaded features for phase: ", self.ph_features)
	
			#loading MoE models
		print("  Loading MoE models")
			#amplitude
		self.MoE_models_amp = []
		k = 0
		while "amp_exp_"+str(k) in file_list and  "amp_gat_"+str(k) in file_list:
			self.MoE_models_amp.append(MoE_model(3+len(self.amp_features),1))
			self.MoE_models_amp[-1].load(folder+"amp_exp_"+str(k),folder+"amp_gat_"+str(k))
			print("    Loaded amplitude model for comp: ", k)
			k += 1
		
			#phase
		self.MoE_models_ph = []
		k = 0
		while "ph_exp_"+str(k) in file_list and  "ph_gat_"+str(k) in file_list:
			self.MoE_models_ph.append(MoE_model(3+len(self.ph_features),1))
			self.MoE_models_ph[-1].load(folder+"ph_exp_"+str(k),folder+"ph_gat_"+str(k))
			print("    Loaded phase model for comp: ", k)
			k += 1

		if "times" in file_list:
			print("  Loaded time vector")
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
		Returns the MoE model(s).
		Input:
			model_type		"amp" or "ph" to state which MoE models shall be returned
		Output:
			
		"""
		if model_type == "amp":
			return self.amp_PCA
		if model_type == "ph":
			return self.ph_PCA
		return None

	def summary(self, filename = None):
		"""
	PCA_models
	==========
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
		Returns the time grid at which the outputs of the models are evaluated. Grid is in reduced units.
		Output:
			time_grid (D,)	points in time grid at which all waves are evaluated
		"""
		return self.times


	def __call__(self, t_grid, m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano , out_type = 'h+x'):
		"""
	__call__
	========
		Generates a WF according to the MLGW model. It makes all the required preprocessing to include wave dependance on the full 14 parameters space of the GW forms.
		Output waveforms can be represented with plus cross polarization, amplitude and phase or h_22 component of the multipole expansion.
		Input:
			t_grid	(N_grid,)		Grid of (physical) time/frequency points to evaluate the wave at
			m1	()/(N,)				Mass of BH 1
			m2	()/(N,)				Mass of BH 1
			spin1_x/y/z	()/(N,)		Each variable represents a spin component of BH 1
			spin2_x/y/z				Each variable represents a spin component of BH 1
			D_L	()/(N,)				Luminosity distance
			i ()/(N,)				Inclination
			phi_0 ()/(N,)			Reference phase for the wave
			long_asc_nodes ()/(N,)	Logitudinal ascentional nodes (currently not implemented)
			eccentricity ()/(N,)	Eccentricity of the orbit (currently not implemented)
			mean_per_ano ()/(N,)	Mean per ano (currently not implemented)
			out_type (str)			The output to be returned ('h+x', 'ampph', 'h22')
		Ouput:
			h_plus, h_cross (1,D)/(N,D)		desidered polarizations (if it applies)
			amp, phase (1,D)/(N,D)			desidered amplitude and phase (if it applies)
			h22_real, h_22_im (1,D)/(N,D)	desidered h_22 component (if it applies)
		"""
		theta = np.column_stack((m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano)) #(N,D)
		return self.get_WF(theta, out_type = out_type, t_grid= t_grid, red_grid = False)


	def get_WF(self, theta, t_grid = None, out_type = 'h+x', red_grid = False):
		"""
	get_WF
	======
		Generates a WF according to the MLGW model. It makes all the required preprocessing to include wave dependance on the full 15 parameters space of the GW forms.
		Wherever not specified, all waves are evaluated at a luminosity distance of 1 Mpc.
		It accepts data in one of the following layout of D features:
			D = 3	[q, spin1_z, spin2_z]
			D = 4	[m1, m2, spin1_z, spin2_z]
			D = 5	[m1, m2, spin1_z , spin2_z, D_L]
			D = 6	[m1, m2, spin1_z , spin2_z, D_L, inclination]
			D = 7	[m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0]
			D = 14	[m1, m2, spin1 (3,), spin2 (3,), D_L, inclination, phi_0, long_asc_nodes, eccentricity, mean_per_ano]
		Warning: last layout (D=14) is made only for compatibility with lalsuite software. The implemented variables are those in D=7 layout; the other are dummy variables and will not be considered.
		Unit of measures:
			[mass] = M_sun
			[D_L] = Mpc
			[spin] = adimensional
		Output waveforms can be represented with plus cross polarization, amplitude and phase or h_22 component of the multipole expansion.
		Input:
			theta (N,D)		source parameters to make prediction at
			t_grid (D',)	a grid in (physical or reduced) time/frequency to evaluate the wave at (uses np.interp)
			out_type (str)	The output to be returned ('h+x', 'ampph', 'h22')
			red_grid		whether given t_grid is in reduced space (True) or physical space (False)
		Ouput:
			h_plus, h_cross (1,D)/(N,D)		desidered polarizations (if it applies)
			amp, phase (1,D)/(N,D)			desidered amplitude and phase (if it applies)
			h22_real, h_22_im (1,D)/(N,D)	desidered h_22 component (if it applies)
		"""
		if t_grid is None:
			t_grid = self.times
			if red_grid == False:
				red_grid = True
				warnings.warn("As no grid is given, the default reduced grid is used to evaluate the output. red_grid option is set to True.")

		theta = np.array(theta) #to ensure user theta is copied into new array
		if theta.ndim == 1:
			theta = theta[np.newaxis,:] #(1,D)
		
		D= theta.shape[1] #number of features given
		if D <3:
			raise RuntimeError("Unable to generata WF. Too few parameters given!!")
			return

			#creating a standard theta vector for __get_WF
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

			#generating waves and returning to user
		res1, res2 = self.__get_WF(theta, t_grid, out_type, red_grid)
			#res1,res2 = h_plus, h_cross if plus_cross = True
			#res1,res2 = amp, ph if plus_cross = False
		return res1, res2

	#@do_profile(follow=[])
	def __get_WF(self, theta, t_grid, out_type, red_grid):
		"""
	__get_WF
	========
		Generates the waves in time domain and perform . Called by get_WF.
		Accepts only input features as [q,s1,s2] or [m1, m2, spin1_z , spin2_z, D_L, inclination, phi_0].
		Input:
			theta (N,D)		source parameters to make prediction at (D=3 or D=7)
			t_grid (D',)	a grid in (physical or reduced) time to evaluate the wave at (uses np.interp)
			out_type (str)	the output to be returned ('h+x', 'ampph', 'h22')
			red_grid		whether given t_grid is in reduced space (True) or physical space (False)
		Output:
			h_plus, h_cross (1,D)/(N,D)		desidered polarizations (if it applies)
			amp, phase (1,D)/(N,D)			desidered amplitude and phase (if it applies)
			h22_real, h_22_im (1,D)/(N,D)	desidered h_22 components (if it applies)
		"""
		D= theta.shape[1] #number of features given
		assert D in [3,7] #check that the number of dimension is fine
		const = 4*np.sqrt(5/(64*np.pi)) #constant of proportionality between h_22 and h_tilde

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

		amp, ph =  self.get_raw_WF(theta_std) #raw WF (N, N_grid)
		amp = 1e-21*amp #scaling back amplitude: less numerically efficient but quicker than doing it at the end

			#doing interpolations
		m_tot_std = 20.
			############
		new_amp = np.zeros((amp.shape[0], t_grid.shape[0]))
		new_ph = np.zeros((amp.shape[0], t_grid.shape[0]))

		for i in range(amp.shape[0]):
			if not red_grid:
				interp_grid = np.divide(t_grid,m_tot_us[i])
			else:
				interp_grid = t_grid
				#putting the wave on the user grid
			new_amp[i,:] = np.interp(interp_grid, self.times, amp[i,:]* m_tot_us[i]/m_tot_std ,left = 0, right = 0) #set to zero outside the domain
			new_ph[i,:]  = np.interp(interp_grid, self.times, ph[i,:])

				#warning if the model extrapolates outiside the grid
			if (interp_grid[0] < self.times[0]):
				warnings.warn("Warning: time grid given is too long for the fitted model. Set 0 amplitude outside the fitting domain.")

		amp = new_amp
		ph = np.subtract(new_ph.T,new_ph[:,0]).T #phase are zero at t = 0 #SLOOOW: do you need it??

			#### Dealing with distance, inclination and phi_0
		if D==7:
			dist_pref = theta[:,4] #std_dist = 1 Mpc
			iota = theta[:,5] #std_inclination = 0.
			phi_0 = theta[:,6] #reference phase

			h_22_real = np.multiply(amp, np.cos(ph) )
			h_22_imag = np.multiply(amp, np.sin(ph) )

			if out_type == 'h22':
				return h_22_real/const, h_22_imag/const

					#parametrization of the wave
				#h = h_p +i h_c = Y_22 * h_22 + Y_2-2 * h_2-2
				#h_22 = h*_2-2
				#Y_2+-2 = sqrt(5/(64pi))*(1+-cos(inclination))**2 exp(+-2i phi)

			h_p, h_c = self.__set_d_iota_phi_dependence(h_22_real,h_22_imag, dist_pref, iota, phi_0)

			if out_type == 'h+x':
				return h_p, h_c
			elif out_type == 'ampph':
				amp =  np.sqrt(np.square(h_p)+np.square(h_c)) 
				ph = np.unwrap(np.arctan2(h_c,h_p)) #attention here... 
				return amp, ph
			else:
				return np.zeros(h_p.shape), np.zeros(h_c.shape)
		if D==3: 
			if out_type == 'ampph':
				return amp, ph
			else:
				h_p = np.multiply(amp, np.cos(ph))
				h_c = np.multiply(amp, np.sin(ph))
				if out_type == 'h+x':
					return h_p, h_c
				elif out_type == 'h22':
					return h_p/const, h_c/const
				else:
					return np.zeros(h_p.shape), np.zeros(h_c.shape)

	def __set_d_iota_phi_dependence(self, h_p, h_c, dist, iota, phi_0):
		"""
	__set_d_iota_phi_dependence
	===========================
		Given h_p, h_c in standard form, it returns the strain with included dependence on distance, inclination iota and reference phase phi_0. It uses the formula
			h(d,iota,phi_0) = Y_22 * h_22 + Y_2-2 * h*_22
		where Y_2+-2 = sqrt(5/(64pi))*(1+-cos(inclination))**2 exp(+-2i phi)
		Input:
			h_p, h_c (N,D)	polarization of the standard wave (as generated by ML)
			iota (N,)		inclination for each wave
			phi_0 (N,)		reference phase for each wave
		Output:
			h_p, h_c (N,D)	processed strain, with d, iota, phi_0 dependence included.
		"""
		c_i = np.cos(iota) #(N,)
			#dealing with h_p
		new_h_p = np.multiply(h_p.T, np.cos(2*phi_0)) - np.multiply(h_c.T, np.sin(2*phi_0)) #(D,N) #included phi dependence
		new_h_p = np.multiply(new_h_p, 0.5*(1+np.square(c_i))/dist ).T #(N,D) #included iota dependence

			#dealing with h_p
		new_h_c = np.multiply(h_p.T, np.sin(2*phi_0)) + np.multiply(h_c.T, np.cos(2*phi_0)) #(D,N) #included phi dependence
		new_h_c = np.multiply(new_h_c, c_i/dist ).T #(N,D) #included iota dependence

		return new_h_p, new_h_c

	def __Y_2m(self,m, iota, phi):
		"""
	__Y_2m
	======
		Spin 2 spherical harmonics with l = 2. Only m = +2,-2 is implemented.
		It has the following expression:
			Y_l+-2 = (1+-cos(iota))**2 e**(2i*m*phi_0)
		Input:
			m			index m (only +/-2 is implemented)
			iota (N,)	inclination angle
			phi_0 (N,)	reference phase
		Output:
			Y_2m (N,)	value of the function
		"""
		const = 1./4.#np.sqrt(5./(64*np.pi)) #the constant was already fitted in the wave
		c_i = np.cos(iota) #(N,)
		Y_2m = const * np.square(1+np.multiply(np.sign(m), c_i)) #(N,)
		Y_2m = np.multiply(Y_2m, np.exp(1j*np.multiply(np.sign(m), 2*phi)) ) #(N,)
		return Y_2m

	def get_raw_WF(self, theta):
		"""
	get_raw_WF
	==========
		Generates a WF according to the MLGW model with a parameters vector in MLGW model style (params=  [q,s1z,s2z]).
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
			rec_PCA_amp[:,k] = self.MoE_models_amp[k].predict(amp_theta) #why is this much sower than the phase part?

			#making predictions for phase
		rec_PCA_ph = np.zeros((ph_theta.shape[0], self.ph_PCA.get_dimensions()[1]))
		for k in range(len(self.MoE_models_ph)):
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
		Computes the gradients (at points theta) of the amplitude and phase w.r.t. (q,s1,s2).
		Gradients are functions dependent on time and are evaluated on the internal reduced grid (GW_generator.get_time_grid()).
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

	def __grads_theta(self, theta, t_grid):
		"""
	__get_grads_theta
	=================
		Returns the gradient of the waveform
			h = A exp(1j*phi) = A cos(phi) + i* A sin(phi)
		with respect to theta = (M, q, s1, s2).
		Gradients are evaluated on the user given time grid t_grid.
		It returns the real and imaginary part of the gradients.
		Input:
			theta (N,4)		orbital parameters with format (m1, m2, s1, s2)
			t_grid (D,)		time grid to evaluate the gradients at
		Output:
			grad_Re(h) (N,D,4)		Gradients of the real part of the waveform
			grad_Im(h) (N,D,4)		Gradients of the imaginary part of the waveform
		"""
		assert theta.shape[1] == 4
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
		grad_q_amp = 1e-21*grad_q_amp
		m_tot_std = 20.
			#interpolating gradients on the user grid
		for i in range(theta_std.shape[0]):
			for j in range(1,4):
				#print(t_grid.shape,self.times.shape)
				grad_amp[i,:,j] = np.interp(t_grid, self.times * m_tot_us[i], grad_q_amp[i,:,j-1]* m_tot_us[i]/m_tot_std ,left = 0, right = 0) #set to zero outside the domain #(D,)
				grad_ph[i,:,j]  = np.interp(t_grid, self.times * m_tot_us[i], grad_q_ph[i,:,j-1]) #(D,)

		#dealing with gradients w.r.t. M
		amp, ph = self.get_WF(theta, t_grid, plus_cross = False, red_grid = False) #true wave evaluated at t_grid #(N,D)
		for i in range(theta_std.shape[0]):
			grad_M_amp = np.gradient(amp[i,:], t_grid) #(D,)
			grad_M_ph = np.gradient(ph[i,:], t_grid) #(D,)
			grad_amp[i,:,0] = amp[i,:]/m_tot_us[i] - np.multiply(t_grid/m_tot_us[i], grad_M_amp) #(D,)
			grad_ph[i,:,0]  = -np.multiply(t_grid/m_tot_us[i], grad_M_ph) #(D,)

		grad_ph = np.subtract(grad_ph,grad_ph[:,0,None,:]) #unclear... but apparently compulsory
			#check when grad is zero and keeping it
		diff = np.concatenate((np.diff(ph, axis = 1), np.zeros((ph.shape[0],1))), axis =1)
		zero = np.where(diff== 0)
		grad_ph[zero[0],zero[1],:] = 0 #takes care of the flat part after ringdown (gradient there shall be zero!!)

		#computing gradients of the real and imaginary part
		ph = np.subtract(ph.T,ph[:,0]).T
		grad_Re = np.multiply(grad_amp, np.cos(ph)[:,:,None]) - np.multiply(np.multiply(grad_ph, np.sin(ph)[:,:,None]), amp[:,:,None]) #(N,D,4)
		grad_Im = np.multiply(grad_amp, np.sin(ph)[:,:,None]) + np.multiply(np.multiply(grad_ph, np.cos(ph)[:,:,None]), amp[:,:,None])#(N,D,4)

			#switching back spins
			#sure of it???
		grad_Re[to_switch,:,2], grad_Re[to_switch,:,3] = grad_Re[to_switch,:,3], grad_Re[to_switch,:,2]
		grad_Im[to_switch,:,2], grad_Im[to_switch,:,3] = grad_Im[to_switch,:,3], grad_Im[to_switch,:,2]

		return grad_Re, grad_Im

	def get_grads(self, theta, t_grid):
		"""
	get_grads
	=========
		Returns the gradients of the waveform h(m1,m2,s1,s2,d_L, iota, phi) with respect to (M, q, s1, s2, d_L, iota, phi).
		Gradients are evaluated on the user given time grid t_grid.
		It returns the real and imaginary part of the gradients.
		Input:
			theta (N,7)		orbital parameters with format (m1, m2, s1, s2, d_L, iota, phi_0)
			t_grid (D,)		time grid to evaluate the gradients at
		Output:
			grad_Re(h) (N,D,7)		Gradients of the real part of the waveform
			grad_Im(h) (N,D,7)		Gradients of the imaginary part of the waveform
		"""
		theta = np.array(theta)
		if theta.ndim == 1:
			theta = theta[None,:]
		assert theta.shape[1]==7
		grad_Re = np.zeros((theta.shape[0],len(t_grid),theta.shape[1])) #(N,D,7)
		grad_Im = np.zeros((theta.shape[0],len(t_grid),theta.shape[1])) #(N,D,7)
		
			#gradients w.r.t. orbital parameters (M, q, s1, s2)
		grad_theta_tilde_Re, grad_theta_tilde_Im = self.__grads_theta(theta[:,:4], t_grid) #(N,D,4)
		for i in range(4):
			grad_theta_tilde_Re[:,:,i], grad_theta_tilde_Im[:,:,i] = self.__set_d_iota_phi_dependence(grad_theta_tilde_Re[:,:,i], grad_theta_tilde_Im[:,:,i], theta[:,4], theta[:,5], theta[:,6]) #(N,D)

			#gradients w.r.t. d
		dist = theta[:,4]
		grad_d_Re, grad_d_Im = self.get_WF(theta, t_grid)
		grad_d_Re = -np.divide(grad_d_Re.T, dist).T
		grad_d_Im = -np.divide(grad_d_Im.T, dist).T
		
			#gradients w.r.t. iota
		iota_0_theta = np.array(theta)
		iota_0_theta[:,5]=0.
		grad_iota_Re, grad_iota_Im = self.get_WF(iota_0_theta, t_grid) #WF with iota =0 #(N,D)
		grad_iota_Re = np.multiply(grad_iota_Re.T, -0.5*np.sin(2*theta[:,5])).T
		grad_iota_Im = np.multiply(grad_iota_Im.T, -np.sin(theta[:,5])).T

			#gradients w.r.t. phi_0
		Re_h, Im_h = self.get_WF(theta[:,:4],t_grid, plus_cross =True) #(N,D)
		c_i = np.cos(theta[:,5]) #(N,)
		phi_0 = theta[:,6]
			#dealing with h_p
		grad_phi_Re = np.multiply(Re_h.T, np.sin(2*phi_0)) + np.multiply(Im_h.T, np.cos(2*phi_0)) #(D,N) #included phi dependence
		grad_phi_Re = -2*np.multiply(grad_phi_Re, 0.5*(1+np.square(c_i))/dist ).T #(N,D) #included iota dependence
			#dealing with h_p
		grad_phi_Im = np.multiply(Re_h.T, np.cos(2*phi_0)) - np.multiply(Im_h.T, np.sin(2*phi_0)) #(D,N) #included phi dependence
		grad_phi_Im = 2.*np.multiply(grad_phi_Im, c_i/dist ).T #(N,D) #included iota dependence

			#producing overall gradients
		grad_Re[:,:,:4] = grad_theta_tilde_Re
		grad_Im[:,:,:4] = grad_theta_tilde_Im
		grad_Re[:,:,4] = grad_d_Re
		grad_Im[:,:,4] = grad_d_Im		
		grad_Re[:,:,5] = grad_iota_Re
		grad_Im[:,:,5] = grad_iota_Im	
		grad_Re[:,:,6] = grad_phi_Re
		grad_Im[:,:,6] = grad_phi_Im	

		return grad_Re, grad_Im











