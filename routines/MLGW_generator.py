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
		Tries to load frequencies.
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

	def get_WF(self, theta, plus_cross = True, freq_grid = None):
		"""
		Generates a WF according to the MLGW model
		Input:
			theta (N,3)		source parameters to make prediction at
			plus_cross		whether to return h_+ and h_x components (if false amp and phase are returned)
			freq_grid (D',)	a grid in frequency to evaluate the wave at (uses np.inter)
		Ouput:
			h_plus, h_cross (N,D)	desidered polarizations
			amp,ph (N,D)			desidered amplitude and phase
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
			rec_amp_dataset = new_rec_amp_dataset
			rec_ph_dataset = new_rec_ph_dataset

		if not plus_cross:
			return rec_amp_dataset, rec_ph_dataset
		if plus_cross:
			h = np.multiply(rec_amp_dataset,np.exp(1j*rec_ph_dataset)) #complex vector
			return h.real, h.imag

		






		
