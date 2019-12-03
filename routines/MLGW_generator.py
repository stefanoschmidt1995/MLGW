#Class for dealing with all the components of a MLGW model made of PCA+MoE

import os
from EM_MoE import *		#MoE model
from ML_routines import *	#PCA model

class MLGW_generator(object):
	"""
	This class hold all the parts of MLGW model. Model is composed by a PCA model to reduce dimensionality of a WF datasets and by several MoE models to fit PCA in terms of source parameters.
	Everything is hold in a PCA model (class PCA_model defined in ML_routines) and in two lists of MoE models (class MoE_model defined in EM_MoE). All models are loaded from files in a folder given by user. Files must be named exactly as follows:
		amp(ph)_exp_#		for amplitude (phase) of expert model for PCA component #
		amp(ph)_gat_#		for amplitude (phase) of gating function for PCA component #
		amp(ph)_feat		for list of features to use for MoE models
		amp(ph)_PCA_model	for PCA model for amplitude (phase)
	No suffixes shall be given to files.
	The class currently doesn't implement methods for fitting: it only provides a useful tool to gather them.
	"""
	def __init__(self, folder = None):
		"""
		Initialise class by loading models from file.
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		An optional frequency vector can be given in file "frequencies". This is not required by model but makes things easier for the user who wants to know at which frequencies waves are generated.
		Input:
			folder	address to folder in which everything is kept (if None, models must be loaded manually with load())
		"""
		self.frequencies = None
		if folder is not None:
			self.load(folder)
		return

	def load(self, folder):
		"""
		Builds up all the models from given folder.
		Everything useful for the model must be put within the folder with the standard names:
			{amp(ph)_exp_# ; amp(ph)_gat_#	; amp(ph)_feat ; amp(ph)_PCA_model}
		There can be an arbitrary number of exp and gating functions as long as they match with each other and they are less than PCA components.
		It loads frequencies.
		Input:
			address to folder in which everything is kept
		"""
		if not folder.endswith('/'):
			folder = folder + "/"
		print(folder)
		file_list = os.listdir(folder)

			#loading PCA
		self.amp_PCA = PCA_model()
		self.amp_PCA.load_model(folder+"amp_PCA_model")
		self.ph_PCA = PCA_model()
		self.ph_PCA.load_model(folder+"ph_PCA_model")

		print("Loaded PCA model for amplitude with ", self.amp_PCA.get_V_matrix().shape[1], " PC")
		print("Loaded PCA model for phase with ", self.ph_PCA.get_V_matrix().shape[1], " PC")

			#loading features
		f = open(folder+"amp_feat", "r")
		self.amp_features = f.readlines()
		for i in range(len(self.amp_features)):
			self.amp_features[i] = self.amp_features[i].rstrip()

		f = open(folder+"ph_feat", "r")
		self.ph_features = f.readlines()
		for i in range(len(self.ph_features)):
			self.ph_features[i] = self.ph_features[i].rstrip()
		
		print("Loaded features for amplitude: ", self.amp_features)
		print("Loaded features for phase: ", self.ph_features)
	
			#loading MoE models
		print("Loading MoE models")
			#amplitude
		self.MoE_models_amp = []
		k = 0
		while "amp_exp_"+str(k) in file_list and  "amp_gat_"+str(k) in file_list:
			self.MoE_models_amp.append(MoE_model(3+len(self.amp_features),1))
			self.MoE_models_amp[-1].load(folder+"amp_exp_"+str(k),folder+"amp_gat_"+str(k))
			print("   Loaded amplitude model for comp: ", k)
			k += 1
		
			#phase
		self.MoE_models_ph = []
		k = 0
		while "ph_exp_"+str(k) in file_list and  "ph_gat_"+str(k) in file_list:
			self.MoE_models_ph.append(MoE_model(3+len(self.ph_features),1))
			self.MoE_models_ph[-1].load(folder+"ph_exp_"+str(k),folder+"ph_gat_"+str(k))
			print("   Loaded phase model for comp: ", k)
			k += 1

		if "frequencies" in file_list:
			self.frequencies = np.loadtxt(folder+"frequencies")
			print("Loaded frequency vector")
		else:
			raise RuntimeError("Unable to load model: no frequency vector given!")

		return

	def MoE_models(self, model_type, k_list=None):
		"""
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

	def get_frequencies(self):
		"""
		Returns the frequencies (if speciefied) at which model is evaluated.
		Output:
			frequencies (D,)	points in frequency grid at which all waves are evaluated
		"""
		return self.frequencies

	def __call__(self, frequencies, m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano , plus_cross = True):
		"""
		Generates a WF according to the MLGW model. It makes all the required preprocessing to include wave dependance on the full 15 parameters space of the GW forms.
		Input:
			frequencies	(N_grid,)	Grid of frequency points to evaluate the wave at
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
			plus_cross				Whether to return h_+ and h_x components (if false amp and phase are returned)
		Ouput:
			h_plus, h_cross (1,D)/(N,D)	desidered polarizations (if it applies)
			
		"""
		theta = np.column_stack((m1, m2, spin1_x, spin1_y, spin1_z, spin2_x, spin2_y, spin2_z, D_L, i, phi_0, long_asc_nodes, eccentricity, mean_per_ano)) #(N,D)
		return self.get_WF(theta, plus_cross = plus_cross, freq_grid= frequencies)

	def get_WF(self, theta, plus_cross = True, freq_grid = None):
		"""
		Generates a WF according to the MLGW model. It makes all the required preprocessing to include wave dependance on the full 15 parameters space of the GW forms.
		Wherever not specified, all waves are evaluated at a luminosity distance of 1 Mpc.
		It accepts data in one of the following layout of D features:
			D = 3	[q, spin1_z, spin2_z]
			D = 4	[m1, m2, spin1_z, spin2_z]
			D = 5	[m1, m2, spin1_z , spin2_z, D_L]
			D = 6	[m1, m2, spin1_z , spin2_z, D_L, inclination]
			D = 14	[m1, m2, spin1 (3,), spin2 (3,), D_L, inclination, phi_0, long_asc_nodes, eccentricity, mean_per_ano]
		Unit of measures:
			[mass] = M_sun
			[D_L] = Mpc
		Input:
			theta (N,D)		source parameters to make prediction at
			plus_cross		whether to return h_+ and h_x components (if false amp and phase are returned)
			freq_grid (D',)	a grid in frequency to evaluate the wave at (uses np.inter)
		Ouput:
			h_plus, h_cross (N,D)	desidered polarizations (if it applies)
			amp,ph (N,D)			desidered amplitude and phase (if it applies)
		"""
		if freq_grid is None:
			freq_grid = self.frequencies

			#some (useless) physical constants
		LAL_MRSUN_SI = 1.476625061404649406193430731479084713e3 	# M_sun in meters (2GM_sun/c**2)
		LAL_MTSUN_SI = 4.925491025543575903411922162094833998e-6	# M_sun in seconds (2GM_sun/c**3)
		LAL_PC_SI = 3.085677581491367278913937957796471611e16		# 1 pc in meters
		LAL_MSUN_SI = 1.988546954961461467461011951140572744e30		# M_sun in kilograms

		if theta.ndim == 1:
			theta = theta[np.newaxis,:] #(1,D)
		
		D= theta.shape[1] #number of features given
		if D <3:
			raise RuntimeError("Unable to generata WF. Too few parameters given!!")
			return

		if D == 3:
			return self.__get_WF__(theta, plus_cross, freq_grid)

			#here starts the complicated part of scaling things
		q = np.divide(np.max(theta[:,0:2]),np.min(theta[:,0:2])) #mass ratio (N,)
		if D == 14:
			theta_std = np.column_stack((q,theta[:,4], theta[:,7])) #(N,3)
			if np.any(np.column_stack((theta[:,2:4], theta[:,5:7])) != 0):
				print("Given nonzero spin_x/spin_y components. Model currently supports only spin_z component. Other spin components are ignored")
		else:
			theta_std = np.column_stack((q,theta[:,2:])) #(N,3)

		h_p, h_c =  self.__get_WF__(theta_std, True, freq_grid) #(N, N_grid)

		mass_scale_factor = np.divide( (1+q)*10, (theta[:,0]+theta[:,1]) ) #prefactor for mass corrections (M_std/M_us) (N,)
		#mass_pref = np.divide( np.sqrt(theta[:,0]*theta[:,1])/np.power(theta[:,0]+theta[:,1],1./6.),
		#					10*np.sqrt(q)/np.power(10*(q+1),1./6.) ) #prefactor for mass corrections (N,)

		#chirp_mass_user_5 = np.divide( np.power(theta[:,0]*theta[:,1], 3.), theta[:,0]+theta[:,1]) #(M_c)**5
		#chirp_mass_std_5 = 1e5*np.divide( np.power(q, 3.), (1+q))

		dist_pref = np.ones((h_c.shape[0],)) #scaling factor for distance (N,)
		cos_i = np.ones((h_c.shape[0],)) # cos(inclination) (N,)

		if D>=5 and D != 14: #distance corrections are done
			dist_pref = theta[:,4] #std_dist = 1 Mpc
		if D == 14:
			dist_pref = theta[:,8] #std_dist = 1 Mpc

		if D>=6 and D != 14: #inclinations corrections are done
			cos_i = np.cos(theta[:,5]) #std_dist = 1 Mpc
		if D == 14:
			cos_i = np.cos(theta[:,9]) #std_dist = 1 Mpc

		#print(theta)
		#print(theta_std)
		#print(mass_scale_factor,dist_pref, cos_i)

			#scaling for mass correction
		print(freq_grid,  np.multiply(freq_grid, mass_scale_factor[0]))
		for j in range(h_p.shape[0]):
			pass
			#h_p[j,:] = np.interp(freq_grid, np.divide(freq_grid, mass_scale_factor[j]), h_p[j,:] )
			#h_c[j,:] = np.interp(freq_grid, np.divide(freq_grid, mass_scale_factor[j]), h_c[j,:] )

			#scaling to required distance
		h_p = np.divide(h_p.T, dist_pref).T
		h_c = np.divide(h_c.T, dist_pref).T
			#scaling for setting inclination
		#????


		print("ciao")
		if plus_cross:
			return h_p, h_c
		else:
			h = h_p +1j*h_c
			amp = np.abs(h)
			ph = np.unwrap(np.angle(h))
			return amp, ph


	def __get_WF__(self, theta, plus_cross = True, freq_grid = None):
		"""
		Generates a WF according to the MLGW model with a parameters vector in MLGW model style (params=  [q,s1z,s2z]).
		All waves are evaluated at a luminosity distance of 1 Mpc and are generated at masses m1 = q * m2 and m2 = 10 M_sun.
		Input:
			theta (N,3)		source parameters to make prediction at
			plus_cross		whether to return h_+ and h_x components (if false amp and phase are returned)
			freq_grid (D',)	a grid in frequency to evaluate the wave at (uses np.inter)
		Ouput:
			h_plus, h_cross (N,D)	desidered polarizations (if it applies)
			amp,ph (N,D)			desidered amplitude and phase (if it applies)
		"""
		assert theta.shape[1] == 3

			#adding extra features
		amp_theta = add_extra_features(theta, self.amp_features)
		ph_theta = add_extra_features(theta, self.ph_features)

			#making predictions for amplitude
		rec_PCA_dataset_amp = np.zeros((amp_theta.shape[0], self.amp_PCA.get_V_matrix().shape[1]))
		for k in range(len(self.MoE_models_amp)):
			rec_PCA_dataset_amp[:,k] = self.MoE_models_amp[k].predict(amp_theta)

			#making predictions for phase
		rec_PCA_dataset_ph = np.zeros((ph_theta.shape[0], self.ph_PCA.get_V_matrix().shape[1]))
		for k in range(len(self.MoE_models_ph)):
			rec_PCA_dataset_ph[:,k] = self.MoE_models_ph[k].predict(ph_theta)

		rec_amp_dataset = self.amp_PCA.reconstruct_data(rec_PCA_dataset_amp)
		rec_ph_dataset = self.ph_PCA.reconstruct_data(rec_PCA_dataset_ph)

		if freq_grid is not None and self.frequencies is not None:
			new_rec_amp_dataset = np.zeros((rec_amp_dataset.shape[0], freq_grid.shape[0]))
			new_rec_ph_dataset = np.zeros((rec_ph_dataset.shape[0], freq_grid.shape[0]))
			for i in range(rec_amp_dataset.shape[0]):
				new_rec_amp_dataset[i,:] = np.interp(freq_grid, self.frequencies, rec_amp_dataset[i,:])
				new_rec_ph_dataset[i,:] = np.interp(freq_grid, self.frequencies, rec_ph_dataset[i,:])
			rec_amp_dataset = 1e-21*new_rec_amp_dataset
			rec_ph_dataset = new_rec_ph_dataset

		if not plus_cross:
			return rec_amp_dataset, rec_ph_dataset
		if plus_cross:
			h = np.multiply(rec_amp_dataset,np.exp(1j*rec_ph_dataset)) #complex vector
			return h.real, h.imag

		






		
