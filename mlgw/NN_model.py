"""
Module NN_model.py
==================

Implements a Neural Network model to generate the reduced PCA coeffiecients of a WF.
"""
import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
from shutil import copy2
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras_tuner import BayesianOptimization, HyperModel
import tensorflow as tf
from tensorflow import keras
from GW_helper import compute_optimal_mismatch
from ML_routines import PCA_model, augment_features
from keras.layers import Dense
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import backend as backend


from pathlib import Path

############################################################

class PcaData: #needs to be cleaned up still
	#FIXME: PCA data needs not to take care of data augmentation, since this is taken care by the NN
	def __init__(self, PCA_data_location, PC_comp, quant, features=[], ratio=1, N=None):
		'''
		Loads PCA data from file location (PCA_data_location, str) of the specified quantity (quant, str) either "ph" or "amp". If PC_comp=None then it loads all of the components, if an integer it will load the all PCs up to that integer, and if a list it will load the specified PCs in the list.
		
		'''
		
		PCA_data_location = Path(PCA_data_location)
		self.data_loc = PCA_data_location
		
		self.times = np.genfromtxt(PCA_data_location/"times.dat")
		self.features = features
		self.PC_comp = PC_comp
		self.quantity = quant
		
		train_theta = np.genfromtxt(PCA_data_location/"PCA_train_theta.dat")
		test_theta = np.genfromtxt(PCA_data_location/"PCA_test_theta.dat")
		
		if quant == 'ph':
			self.pca = PCA_model(PCA_data_location/"ph_PCA_model.dat")
			train_var = np.genfromtxt(PCA_data_location/"PCA_train_ph.dat")
			test_var = np.genfromtxt(PCA_data_location/"PCA_test_ph.dat")
			if len(train_var.shape) == 1:
				train_var = np.reshape(train_var, (train_var.shape[0],1))
				test_var = np.reshape(test_var, (test_var.shape[0],1))
				print(train_var.shape)
		if quant == 'amp':
			self.pca = PCA_model(PCA_data_location/"amp_PCA_model.dat")
			train_var = np.genfromtxt(PCA_data_location/"PCA_train_amp.dat")
			test_var = np.genfromtxt(PCA_data_location/"PCA_test_amp.dat")
		
		if PC_comp is None: 
			PC_comp = train_var.shape[1]
		if isinstance(PC_comp,int): PC_comp = list(range(PC_comp))

		self.PC_comp = PC_comp
		if N is None:
			if ratio == 1:
				self.train_theta = train_theta
				self.test_theta = test_theta
				self.train_var = train_var[:,PC_comp]
				self.test_var = test_var[:,PC_comp]
			if ratio < 1:
				train_indc = np.random.sample(range(len(train_theta)),round(ratio*len(train_theta)))
				test_indc = np.random.sample(range(len(test_theta)),round(ratio*len(test_theta)))
				self.train_theta = train_theta[train_indc]
				self.test_theta = test_theta[test_indc]
				self.train_var = train_var[train_indc][:,PC_comp]
				self.test_var = test_var[test_indc][:,PC_comp]
		else:
			self.train_theta = train_theta[:N]
			self.test_theta = test_theta
			self.train_var = train_var[:N,PC_comp]
			self.test_var = test_var[:,PC_comp]

		#self.augment_features()
		self.train_theta = augment_features(self.train_theta, features = self.features)
		self.test_theta = augment_features(self.test_theta, features = self.features)

	def compute_WF(amp, ph, ratio=1, ph_shift = []):
		(N,D) = amp.shape
		
		if len(ph_shift) == 0:
			return np.multiply(amp[:round(N*ratio)],np.e**(1j*ph[:round(N*ratio)]))
		
		else:
			for i in range(len(ph)):
				ph[i,:] -= ph_shift[i]
			return np.multiply(amp[:round(N*ratio)],np.e**(1j*ph[:round(N*ratio)]))
	
	def ConvertPcaData(old_data_loc, pca_model_loc, save_loc, quantity=""):
		'''
			Converts data to new pca data using a different pca_model and saves it at 
			the specified location.
			so: reconstruct old data using old model --> reduce data with new model
		'''
		times = np.genfromtxt(old_data_loc+"times.dat")
		train_theta = np.genfromtxt(old_data_loc+"PCA_train_theta.dat")
		test_theta = np.genfromtxt(old_data_loc+"PCA_test_theta.dat")
		
		train_ph = np.genfromtxt(old_data_loc+"PCA_train_ph.dat")
		test_ph = np.genfromtxt(old_data_loc+"PCA_test_ph.dat")
		train_amp = np.genfromtxt(old_data_loc+"PCA_train_amp.dat")
		test_amp = np.genfromtxt(old_data_loc+"PCA_test_amp.dat")
		
		ph_PCA_old = PCA_model(old_data_loc+'ph_PCA_model.dat')
		amp_PCA_old = PCA_model(old_data_loc+'amp_PCA_model.dat')
		
		ph_PCA_new = PCA_model(pca_model_loc+'ph_PCA_model.dat')
		amp_PCA_new = PCA_model(pca_model_loc+'amp_PCA_model.dat')
	
		train_ph = ph_PCA_old.reconstruct_data(train_ph)
		test_ph = ph_PCA_old.reconstruct_data(test_ph)
		train_amp = amp_PCA_old.reconstruct_data(train_amp)
		test_amp = amp_PCA_old.reconstruct_data(test_amp)
		
		train_ph = ph_PCA_new.reduce_data(train_ph)
		test_ph = ph_PCA_new.reduce_data(test_ph)
		train_amp = amp_PCA_new.reduce_data(train_amp)
		test_amp = amp_PCA_new.reduce_data(test_amp)
		
		#save data to save location
		amp_PCA_new.save_model(save_loc+"amp_PCA_model.dat")
		ph_PCA_new.save_model(save_loc+"ph_PCA_model.dat")
		np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
		np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
		np.savetxt(save_loc+"PCA_train_amp.dat", train_amp)
		np.savetxt(save_loc+"PCA_test_amp.dat", test_amp)
		np.savetxt(save_loc+"PCA_train_ph.dat", train_ph)
		np.savetxt(save_loc+"PCA_test_ph.dat", test_ph)
		np.savetxt(save_loc+"times.dat", times)
	
	def MergePCAsets(file_loc_1, file_loc_2, save_loc, shuffle = True):
		'''
			assumes both files have the same pca model for amp and ph
		'''
		ph_PCA = PCA_model(file_loc_1+'ph_PCA_model.dat') #should be the same for file_loc 1 and 2
		amp_PCA = PCA_model(file_loc_2+'amp_PCA_model.dat')
		
		train_theta_1 = np.genfromtxt(file_loc_1+"PCA_train_theta.dat")
		test_theta_1 = np.genfromtxt(file_loc_1+"PCA_test_theta.dat")
		train_ph_1 = np.genfromtxt(file_loc_1+"PCA_train_ph.dat")
		test_ph_1 = np.genfromtxt(file_loc_1+"PCA_test_ph.dat")
		train_amp_1 = np.genfromtxt(file_loc_1+"PCA_train_amp.dat")
		test_amp_1 = np.genfromtxt(file_loc_1+"PCA_test_amp.dat")
		times_1 = np.genfromtxt(file_loc_1+"times.dat")
		
		train_theta_2 = np.genfromtxt(file_loc_2+"PCA_train_theta.dat")
		test_theta_2 = np.genfromtxt(file_loc_2+"PCA_test_theta.dat")
		train_ph_2 = np.genfromtxt(file_loc_2+"PCA_train_ph.dat")
		test_ph_2 = np.genfromtxt(file_loc_2+"PCA_test_ph.dat")
		train_amp_2 = np.genfromtxt(file_loc_2+"PCA_train_amp.dat")
		test_amp_2 = np.genfromtxt(file_loc_2+"PCA_test_amp.dat")
		times_2 = np.genfromtxt(file_loc_2+"times.dat")
		
		N1_train,N2_train = train_ph_1.shape[0], train_ph_2.shape[0]
		indc_train = np.random.sample(range(N1_train+N2_train),N1_train+N2_train)
		N1_test,N2_test = test_ph_1.shape[0], test_ph_2.shape[0]
		indc_test = np.random.sample(range(N1_test+N2_test),N1_test+N2_test)
		
		train_theta = np.concatenate((train_theta_1,train_theta_2),axis=0)[indc_train,:]
		test_theta = np.concatenate((test_theta_1,test_theta_2),axis=0)[indc_test,:]
		train_ph = np.concatenate((train_ph_1,train_ph_2),axis=0)[indc_train,:]
		test_ph = np.concatenate((test_ph_1,test_ph_2),axis=0)[indc_test,:]
		train_amp = np.concatenate((train_amp_1,train_amp_2),axis=0)[indc_train,:]
		test_amp = np.concatenate((test_amp_1,test_amp_2),axis=0)[indc_test,:]
		times = np.concatenate((times_1,times_2),axis=0)
		
		amp_PCA.save_model(save_loc+"amp_PCA_model.dat")
		ph_PCA.save_model(save_loc+"ph_PCA_model.dat")
		np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
		np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
		np.savetxt(save_loc+"PCA_train_amp.dat", train_amp)
		np.savetxt(save_loc+"PCA_test_amp.dat", test_amp)
		np.savetxt(save_loc+"PCA_train_ph.dat", train_ph)
		np.savetxt(save_loc+"PCA_test_ph.dat", test_ph)
		np.savetxt(save_loc+"times.dat", times)

class CustomLoss:
	def custom_MSE_loss(weights):
		weights = np.array(weights)
		# returns the custom loss function given the weights
		def loss_function(y_true, y_pred):
			if len(weights) != len(y_true[0]):
				print("the length of weights does not match amount of PCA components")
			
			loss = tf.square((y_true - y_pred) * weights )
			return tf.math.reduce_mean(loss, axis=-1)
		
		return loss_function

	def custom_exp_loss(exp, weights):
		weights = np.array(weights)
		
		def loss_function(y_true, y_pred):
			if len(weights) != len(y_true[0]):
				print("the length of weights does not match amount of PCA components")
			
			loss = tf.abs((y_true - y_pred)*weights)**exp
			
			return tf.math.reduce_mean(loss, axis=-1)
		
		return loss_function

class Schedulers:
	def __init__(self, name, exp=0, decay_epoch = 500, min_lr = 0):
		def scheduler(epoch, lr):
			if epoch < decay_epoch or lr < min_lr:
				return lr
			else:
				return lr * np.exp(exp) #exp 0 is just no decay
		
		self.scheduler = scheduler
		self.exp = exp
		self.name = name
		self.decay_epoch = decay_epoch

class LossFunctions:
	def __init__(self, name, weights = [], exp=2):
		self.weights = weights
		if name == "mean_squared_error":
			self.name = 'mean_squared_error'
			self.LF = tf.keras.losses.mean_squared_error
		if name == "mean_absolute_error":
			self.name = 'mean_absolute_error'
			self.LF = tf.keras.losses.mean_absolute_error
		if name == 'mean_squared_logarithmic_error':
			self.name = 'mean_squared_logarithmic_error'
			self.LF = tf.keras.losses.msle
		if name == 'custom_mse':
			self.name = 'custom_mse'
			self.LF = CustomLoss.custom_MSE_loss(weights)
		if name == "custom_exp":
			self.name = "custom_exp"
			self.LF = CustomLoss.custom_exp_loss(exp, weights)

class Optimizers:
	def __init__(self, name, lr=0):
		if name == 'Adam':
			self.name = 'Adam'
			if lr != 0:
				self.opt = tf.keras.optimizers.Adam(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.Adam()
				self.lr = "default"
		if name == 'Adagrad':
			self.name = 'Adagrad'
			if lr != 0:
				self.opt = tf.keras.optimizers.Adagrad(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.Adagrad()
				self.lr = "default"
		if name == 'Adadelta':
			self.name = 'Adadelta'
			if lr != 0:
				self.opt = tf.keras.optimizers.Adadelta(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.Adadelta()
				self.lr = "default"
		if name == 'RMSprop':
			self.name = 'RMSprop'
			if lr != 0:
				self.opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.RMSprop()
				self.lr = "default"
		if name == 'SGD':
			self.name = 'SGD'
			if lr != 0:
				self.opt = tf.keras.optimizers.SGD(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.SGD()
				self.lr = "default"
		if name == 'Nadam':
			self.name = 'Nadam'
			if lr != 0:
				self.opt = tf.keras.optimizers.Nadam(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.Nadam()
				self.lr = "default"
		if name == 'Adamax':
			self.name = 'Adamax'
			if lr != 0:
				self.opt = tf.keras.optimizers.Adamax(learning_rate=lr)
				self.lr = lr
			else:
				self.opt = tf.keras.optimizers.Adamax()
				self.lr = "default"

class mlgw_NN(keras.Sequential):
	def __init__(self, layers = None, name = None, features=None):
		if name is None: name = 'sequential'
		if isinstance(features, str): features = [features]
		if features is None: features = ['']

		id_ = name.find('---')
		if id_ == -1:
			name = name +'---' + '--'.join(features)
		else:
			if features[0] == '':
				feat_str = name[id_+3:]
				features = feat_str.split('--')
			else:
				name = name[:id_+3] + '--'.join(features)
		super().__init__(layers, name)
		self.features = [f.strip() for f in features]

	def fit(self, x = None, y = None, epochs = None, validation_data=None, batch_size=None, callbacks=None, **kwargs):
		if x.shape[-1] == 3:
			x = augment_features(x, features=self.features)
			if validation_data:
				validation_data = (augment_features(validation_data[0], features=self.features), validation_data[1]) 
		return super().fit(x = x, y = y, epochs=epochs, validation_data = validation_data,
				batch_size = batch_size, callbacks = callbacks, **kwargs)

	def predict(self, x, **kwargs):
		if x.shape[-1] == 3:
			x = augment_features(x, features=self.features)
		return super().predict(x, **kwargs)

	@classmethod
	def load_from_folder(cls, model_loc, name = None):
		model_loc = str(model_loc)
		nn_file = glob.glob(model_loc+'*keras')
		assert len(nn_file)	== 1, "More than one neural network model is in the given folder!"

		return cls.load_weights_and_features(nn_file[0], feat_file[0], name)
	
	@classmethod
	def load_from_file(cls, nn_file, name = None):
		with tf.keras.utils.CustomObjectScope({'mlgw_NN': mlgw_NN}):
			model = keras.models.load_model(nn_file, compile=False)
		if name is None: name = model.name
		return cls(model.layers, name, features = None)
	
class NN_HyperModel(HyperModel):
	def __init__(self,  output_nodes, hyperparameter_ranges, loss_weights):
		self.hyperparameter_ranges = hyperparameter_ranges
		self.loss_weights = [1]*output_nodes if not isinstance(loss_weights,list) else loss_weights
		self.output_nodes = output_nodes
		
	def build(self, hp):
		
			#This apparently helps to save memory
			#https://stackoverflow.com/questions/42047497/keras-out-of-memory-when-doing-hyper-parameter-grid-search
		backend.clear_session()
		
		#FIXME: the enumeration here is super ugly: any chance to improve it?
		if isinstance(self.hyperparameter_ranges["units"], (tuple, list)):
			units = hp.Choice('units', self.hyperparameter_ranges["units"])
		else:
			units = hp.Fixed('units', self.hyperparameter_ranges["units"])
		if isinstance(self.hyperparameter_ranges["layers"], (tuple, list)):
			layers = hp.Choice('layers', self.hyperparameter_ranges["layers"])
		else:
			layers = hp.Fixed('layers', self.hyperparameter_ranges["layers"])
		if isinstance(self.hyperparameter_ranges["activation"], (tuple, list)):
			activation = hp.Choice('activation', self.hyperparameter_ranges["activation"])
		else:
			activation = hp.Fixed('activation', self.hyperparameter_ranges["activation"])
		if isinstance(self.hyperparameter_ranges["learning_rate"], (tuple, list)):
			lr = hp.Choice('learning rate', self.hyperparameter_ranges["learning_rate"])
		else:
			lr = hp.Fixed('learning rate',self.hyperparameter_ranges["learning_rate"])
		if isinstance(self.hyperparameter_ranges["feature_order"], (tuple, list)):
			feature_order = hp.Choice('feature order', self.hyperparameter_ranges["feature_order"])
		else:
			feature_order = hp.Fixed('feature order', self.hyperparameter_ranges["feature_order"])
		if isinstance(self.hyperparameter_ranges["features"], (tuple, list)):
			features = hp.Choice('features', self.hyperparameter_ranges["features"])
		else:
			features = hp.Fixed('features', self.hyperparameter_ranges["features"])
		
		print("The number of units are", units)
		feats = '{}-{}'.format(feature_order, features)
		model = mlgw_NN(features=feats)
		D = len(augment_features([1,1,1], features=feats)[0]) #number of input features
		print("number of features: ", D)

		model.add(Dense(units,
						activation=activation,
						input_shape=(D,)))
		for _ in range(layers):
			model.add(Dense(units,
						activation=activation))
		model.add(Dense(self.output_nodes, activation='linear'))

		model.compile(loss=LossFunctions('custom_mse',weights=self.loss_weights).LF,
				optimizer=Nadam(learning_rate=lr))
		return model

def save_model(model, history, out_folder, hyperparameters, PCA_data, PCA_data_loc = None, residual=False):
	#saves the weights and features of a trained model to a file.
	#also saves the relevant hyperparameters.

	out_folder = Path(out_folder)

	K = len(PCA_data.test_var[0]) #number of PCA components
	MSE = np.zeros(K)

	pred_var = model.predict(PCA_data.test_theta)
	for i in range(K):
		MSE[i]=np.sum( np.square(PCA_data.test_var[:,i]-pred_var[:,i]) / pred_var.shape[0])

	PC_comp_string = "".join(str(x) for x in PCA_data.PC_comp)

	residual_str = '_residual' if residual else ''
	fit_info_name = "Model_fit_info_{}{}.txt".format(PC_comp_string, residual_str)
	f = open(out_folder/fit_info_name, 'w')
	f.write("Model for " + PC_comp_string + 'PCs for '+PCA_data.quantity+'\n')
	if PCA_data_loc != None: f.write("Trained on dataset: {}\n".format(str(PCA_data_loc)))
	f.write("created model with params: \n")
	f.write("layer_list : ["+",".join([str(x) for x in hyperparameters['layer_list']])+']\n')
	f.write("optimizer : " + hyperparameters['optimizers'].name + " with learning rate " + str(hyperparameters['optimizers'].lr)+'\n')
	f.write("activation : " + hyperparameters['activation'] + '\n')
	f.write("batch_size : " + str(hyperparameters['batch_size']) + "\n")
	f.write("schedulers : " + hyperparameters['schedulers'].name + " with decay rate " + str(hyperparameters['schedulers'].exp)+'\n')

	for i in range(K):
		f.write("The MSE of principal component "+ str(i) + " of the fit is: "+ str(MSE[i])+"\n")

	f.close()

	plt.figure('lossfunction')
	plt.title('Loss function of '+PCA_data.quantity)
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.yscale('log')
	plt.legend()
	plt.savefig(out_folder/'lossfunction_{}{}.png'.format(PC_comp_string, residual_str))
	plt.close(fig='lossfunction')
	savemodel_file = '{}_weights_{}{}.keras'.format(PCA_data.quantity, PC_comp_string, residual_str)
	model.save(out_folder/savemodel_file)
	if residual:
		coefficients = np.genfromtxt(PCA_data_loc/'residual_coefficients_{}'.format(PC_comp_string))
		np.savetxt(out_folder/'residual_coefficients_{}'.format(PC_comp_string), coefficients)
	return

def tune_model(out_folder, project_name, quantity, PCA_data_loc, PC_to_fit, hyperparameters, max_epochs=1000, init_trials=None, trials = 10):
	#this function does the tuning of a model on a dataset. The input should be: 
	#- folder to which to save the tuning results
	#- the principal components that you want to fit
	#- the hyperparameters you want to tune
	#- other parameters that are necessary for tuning

	PCA_data = PcaData(PCA_data_loc, PC_to_fit, quantity)

	loss_weights = np.sqrt(np.array(PCA_data.pca.PCA_params[2]))[PCA_data.PC_comp] 
	loss_weights = loss_weights / min(loss_weights)
	
	if init_trials == None: init_trials = 3*len(hyperparameters)
	
	tuner = BayesianOptimization(
		NN_HyperModel(PCA_data.train_var.shape[1], hyperparameters, loss_weights),
		objective='val_loss',
		max_trials=trials,
		num_initial_points=init_trials,
		overwrite=True,
		directory=out_folder,
		project_name=project_name
	)

	callback_list = [EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
		LearningRateScheduler(Schedulers('exponential', exp=-0.0003).scheduler)]

	tuner.search(PCA_data.train_theta, PCA_data.train_var, epochs=max_epochs,
			validation_data=(PCA_data.test_theta, PCA_data.test_var),
			batch_size=128, callbacks=callback_list)
	return tuner

def analyse_tuner_results(file_loc, save_loc=None):
	data = []
	if not isinstance(file_loc, Path): file_loc = Path(file_loc)

	for x in os.listdir(file_loc):
		if x.startswith('trial'):
			with open(file_loc/"{}/trial.json".format(x), "r") as f:
				cur_data = json.load(f)
				if cur_data['score'] != None: #tuner is not finished yet
					data.append([x[6:],cur_data['score'],cur_data['hyperparameters']['values']])
	if len(data)==0:
		raise ValueError("Unable to load any model from the given folder '{}'".format(file_loc))

	hyperparams = data[0][2].keys()
	for param in hyperparams:
		x = plt.figure(param)
		plt.title(param+" vs loss")
		plt.scatter([data[i][2][param] for i in range(len(data))],
					[data[i][1] for i in range(len(data))])
		plt.yscale('log')
		if param in ['learning rate']: plt.xscale('log')
		if save_loc != None: plt.savefig(save_loc+"/"+param+".png")
	plt.show()

	data.sort(key=lambda x : x[1])
	
	print("Top 50 hyperparameters are: \n")
	for i in range(50):
		print(data[i][2], '\n\twith a score of ', data[i][1])
	return

def fit_NN(fit_type, in_folder, out_folder, hyperparameters, N_train = None, comp_to_fit = None, features = None, epochs = 2000, verbose = 1, residual=False):
	"""
	Fit a NN model for the selected PC's of the PCA dataset
	It loads a PCA dataset from in_folder and fits the regression
	
		theta = (q,s1,s2) ---> PC_pojection(theta)
	
	Outputs the fitted model to out_folder
	
		amp(ph)_PCs.keras (weights of the model)
	
		info.txt
	
	Furthermore it copies PCA models and times files to the out folder.
	Several of these NNs, for both amplitude and phase, can be combined with the NN_gather method (STILL HAEV TO IMPLEMENT!!!), which can then be inputted to mlgw.GW_generator.GW_generator.
	User can choose some fitting hyperparameters.

	Input:
		fit_type: str
			whether to fit the model for amplitude or phase ("amp" or "ph")
		in_folder: str
			path to folder with the PCA dataset. It must have the format of mlgw.fit_model.create_PCA_dataset
		out_folder: str
			path to folder to save models to. Several folder locations can be inputted to NN_gather which can then be used by mlgw.GW_generator.mode_generator_NN
		hyperparameters: dict
			dictionary of the hyperparameters used for the NN structure. If none, default parameters are used.
		N_train: int
			integer for how many training samples to use. If none, all are used.
		comp_to_fit: listl
			PCs to fit. If None, all components will be fitted. If int, it denotes the maximum PC order to be fitted.
		features: list
			Strings or tuples for adding features. Note that only features implemented in PcaData_v2.PcaData.augment_features can be used. If None, no extra features will be added: see :func:`add_extra_features`
		epochs: int
			integer specifying for how many iterations (epochs) the NN should train
		verbose: bool
			whether to display NN iteration messages
		residual: bool
			whether this is a model for the residual 
	"""
	assert fit_type in ["amp","ph"], "Data type for fit_type not understood. Required 'amp' or 'ph but {} given.".format(fit_type)

	os.makedirs(out_folder, exist_ok = True)

	out_folder = Path(out_folder)
	in_folder = Path(in_folder)

	if features is None:
		features = []
	if not isinstance(features, list):
		raise RuntimeError("Features to use for regression must be given as list. Type "+str(type(features))+" given instead")
		return
	
	if not (isinstance(N_train, int) or N_train is None):
		raise RuntimeError("Number of training point to use must be be an integer. Type {} given instead".format(type(N_train)))

		#loading data
	PCA_data = PcaData(in_folder, comp_to_fit, fit_type, features=features, N=N_train)
	#print("Training parameters: ", PCA_data.train_var[0])
	#print("Features used: ", PCA_data.features)
	print("Using "+str(PCA_data.train_var.shape[0])+" train data")
	
	D = PCA_data.train_theta.shape[1] #dimensionality of input space for NN
	if N_train == None: N_train = PCA_data.train_theta.shape[0]

	mse_train_list = [] #list for holding mse of every PCs
	mse_test_list = [] #list for holding mse of every PCs
	
		#starting fit procedure TODO: implement the training of neural network, and the tests and saving to files.
	PC_comp_string = "".join(str(x) for x in PCA_data.PC_comp)
	print("Starting fitting components ", PC_comp_string)
	if hyperparameters == None:
		warnings.warn("Default hyperparameters are being used for the NN (see Model_fit_info.txt in the out_folder)")
		hyperparameters = {'layer_list' : [20,20,15,15], 
				'optimizers' : Optimizers("Nadam",0.002), 
				'activation' : "sigmoid", 
				'batch_size' : 64, 
				'schedulers' : Schedulers('exponential', exp=-0.0005)}
	
	loss_weights = np.sqrt(np.array(PCA_data.pca.PCA_params[2]))[PCA_data.PC_comp]
	loss_weights = loss_weights / min(loss_weights)
	print("Using loss function weights: ", loss_weights)
	
	#model = keras.Sequential()
	model = mlgw_NN(name = 'nn_{}_{}'.format(fit_type, PC_comp_string),features=features)
	model.add(Dense(hyperparameters['layer_list'][0],
						activation=hyperparameters['activation'],
						input_shape=(D,)))
	for units in hyperparameters['layer_list'][1:]:
		model.add(Dense(units,
					activation=hyperparameters['activation']))
	
	model.add(Dense(PCA_data.test_var.shape[1], activation='linear'))

	model.compile(loss=LossFunctions('custom_mse', weights=loss_weights).LF,
					optimizer=hyperparameters['optimizers'].opt)
	
	callback_list = []
	early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
	callback_list.append(early_stopping)

	LR_scheduler = LearningRateScheduler(hyperparameters['schedulers'].scheduler)
	callback_list.append(LR_scheduler)

	history = model.fit(x=PCA_data.train_theta,  y=PCA_data.train_var, batch_size=hyperparameters['batch_size'], validation_data=(PCA_data.test_theta,PCA_data.test_var), epochs=epochs, verbose=verbose,callbacks=callback_list)
	
	print("Successfuly trained model!")
		#doing some test
		
	y_pred = model.predict(PCA_data.test_theta) #y_pred is now an ~(N,K) array
	for i in range(len(PCA_data.PC_comp)):
		mse_test_list.append(np.sum(np.square(y_pred[:,i]-PCA_data.test_var[:,i]))/(y_pred.shape[0]))
	
	print("Test square loss for each component: ", mse_test_list)

	#Not neccessary to save PCA model right?
	#copyfile(in_folder+fit_type+"_PCA_model", out_folder+fit_type+"_PCA_model")
	#copyfile(in_folder+"times.dat", out_folder+"times.dat")
	
		#saving NN model
	save_model(model, history, out_folder, hyperparameters, PCA_data, in_folder, residual=residual)
	print("Succesfully saved features and weights")

	return

def create_residual_PCA(pca_data_loc, base_model_file, save_loc, quantity, components, savefigs=True):
	pca_data_loc = Path(pca_data_loc)
	save_loc = Path(save_loc)
	os.makedirs(save_loc, exist_ok = True)
	
	assert quantity.lower()  in ['amp', 'ph'], "Quantity must be either amplitude or phase"

	#load in the PCA dataset
	data = PcaData(pca_data_loc, components, quantity)
	if isinstance(components, int): components = list(range(components))
	PC_comp_string = "".join(str(x) for x in components)

	#load in the model
	M = mlgw_NN.load_from_file(base_model_file)

	#do the predictions with the model

	train_pred = M.predict(data.train_theta)[:,components]
	test_pred = M.predict(data.test_theta)[:,components]

	#print MSE for debugging purposes
	for c in components:
		print("MSE PC ", c, ' is ',  np.sum( np.square(test_pred[:,c] - data.test_var[:,c])) / test_pred.shape[0])

	#subtract the predictions from the real data
	new_train_var = data.train_var[:,components] - train_pred
	new_test_var = data.test_var[:,components] - test_pred

	#normalize the new data
	#FIXME: do this without the for loop
	norm_coef = []
	for c in components:
		test_max = np.max(abs(new_test_var[:,c]))
		new_train_var[:,c] /= test_max
		new_test_var[:,c] /= test_max
		norm_coef.append(test_max)

	#save the new data
	np.savetxt(save_loc/"times.dat", data.times)
	np.savetxt(save_loc/"PCA_train_theta.dat", data.train_theta)
	np.savetxt(save_loc/"PCA_test_theta.dat", data.test_theta)
	data.pca.save_model(save_loc/"ph_PCA_model.dat")
	np.savetxt(save_loc/"residual_coefficients_{}".format(PC_comp_string), norm_coef)

	np.savetxt(save_loc/"PCA_train_{}.dat".format(quantity), new_train_var)
	np.savetxt(save_loc/"PCA_test_{}.dat".format(quantity), new_test_var)

	if savefigs:
		for i,c in enumerate(components):
			plt.figure()
			plt.title('delta pca/pred for test data as function of mass ratio')
			plt.scatter(data.test_theta[:,0], data.test_var[:,c] - test_pred[:,i])
			plt.savefig(save_loc/'delta pca-pred comp{}.png'.format(c+1))
			plt.xlabel('mass ratio')
			plt.close() 

	return

def compute_mismatch_WFS(ph_rec, amp_rec, ph_pca, amp_pca, time_grid, size, dt = 0.00001, plot = False):
	F = np.zeros((size))
	time_grid = time_grid[:]


	num = round( (max(time_grid)-min(time_grid)) / dt)
	new_x_grid = np.linspace(min(time_grid), max(time_grid),num)

	batch_size = min(100,size) #should divide size, but not too large because memory issues

	for j in range(size // batch_size):
		new_ph_rec = np.empty((batch_size,num),dtype=float)
		new_amp_rec = np.empty((batch_size,num),dtype=float)
		new_ph_pca = np.empty((batch_size,num),dtype=float)
		new_amp_pca = np.empty((batch_size,num),dtype=float)

		for i in range(batch_size):
			new_ph_rec[i,:] = np.interp(new_x_grid, time_grid, ph_rec[j*batch_size+i,:], left=0, right=0)
			new_amp_rec[i,:] = np.interp(new_x_grid, time_grid, amp_rec[j*batch_size+i,:], left=0, right=0)
			new_ph_pca[i,:] = np.interp(new_x_grid, time_grid, ph_pca[j*batch_size+i,:], left=0, right=0)
			new_amp_pca[i,:] = np.interp(new_x_grid, time_grid, amp_pca[j*batch_size+i,:], left=0, right=0)

		rec_WFs = PcaData.compute_WF(new_amp_rec,new_ph_rec,ratio=1)
		pca_WFs = PcaData.compute_WF(new_amp_pca,new_ph_pca,ratio=1)

		F_model,phase_shift_model = compute_optimal_mismatch(rec_WFs, pca_WFs)
		F[j*batch_size:(j+1)*batch_size] = F_model


	if plot:
		plt.figure()
		plt.hist(np.log(F)/np.log(10))

	return F

def check_NN_performance(data_loc, amp_model_locs, ph_model_locs, save_loc, mismatch_N=0):
	#TODO: seperate loading in the models from different files and checking the performance in to two different functions

	#computes the MSE and optionally mismatch for inputted models on dataset
	#if mismatch_N > 0, it will return the mismatches.

	ph_data = PcaData(data_loc, None, 'ph')
	amp_data = PcaData(data_loc, None, 'amp')

	testing_for_ph, testing_for_amp, ph_models, ph_models_res, amp_models, amp_modeled_comps, ph_modeled_comps, ph_modeled_comps_res = load_models_from_directories(amp_model_locs, ph_model_locs)

	#do the predictions, if not testing for either amplitude or phase, just use the "perfect" PCA coeffcients
	if not testing_for_ph: ph_pred = ph_data.test_var
	else:
		ph_pred = np.zeros(ph_data.test_var.shape)
		for comps in ph_models.keys():
			ph_pred[:,list(comps)] = ph_models[comps].predict(ph_data.test_theta)

		for comps in ph_models_res.keys():
			cur_pred = ph_models_res[comps].predict(ph_data.test_theta)
			for k in comps:
				ph_pred[:,k] += cur_pred[:,k]*ph_modeled_comps_res[k]

	if not testing_for_amp: amp_pred = amp_data.test_var
	else:
		amp_pred = np.zeros(amp_data.test_var.shape)

		for comps in amp_models.keys():
			amp_pred[:,list(comps)] = amp_models[comps].predict(amp_data.test_theta)

	print("Predictions made successfully!")

	if testing_for_ph:
		print("MSE for phase: ")
		N,K = ph_data.test_var.shape
		for k in range(min(K, max(ph_modeled_comps)+1)):
			print( np.sum( np.square(ph_pred[:,k]-ph_data.test_var[:,k]) / N) )

	if testing_for_amp:
		print("MSE for phase: ")
		N,K = amp_data.test_var.shape
		for k in range(min(K, max(amp_modeled_comps)+1)):
			print( np.sum( np.square(amp_pred[:,k]-amp_data.test_var[:,k]) / N) )

	if mismatch_N == 0: return 

	ph_rec = ph_data.pca.reconstruct_data(ph_pred,K=ph_pred.shape[1])
	amp_rec = amp_data.pca.reconstruct_data(amp_pred,K=amp_pred.shape[1])

	ph_pca = ph_data.pca.reconstruct_data(ph_data.test_var,K=ph_data.test_var.shape[1])
	amp_pca = amp_data.pca.reconstruct_data(amp_data.test_var,K=amp_data.test_var.shape[1])

	x_grid = ph_data.times

	F = compute_mismatch_WFS(ph_rec, amp_rec, ph_pca, amp_pca, x_grid, mismatch_N)

	print('median mismatch is: ', np.median(F))

	return F

def gather_NN(mode, pca_data_location, amp_model_locations, ph_model_locations, out_folder):
	"""
	Combines ampltidude and phase models for a specific mode and formats them in a folder inside the out_folder which can then be inputted 	in the mode_generator class. It assumes the folders in amp_model_locations are formatted as outputted by fit_NN.
	mode : string "lm" that refers to the mode 
	amp_model_locations : list of strings containing the model locations for amplitude
	ph_model_locations : list of strings containing the model locations for phase
	out_folder : string contaning the folder to which combined model should be saved.
	"""
	pca_data_location = Path(pca_data_location)
	out_folder = Path(out_folder)
	if not os.path.isdir(out_folder): #check if out_folder exists
			os.makedirs(out_folder)

	out_folder = out_folder/mode
	os.makedirs(out_folder)
	
	copy2(pca_data_location/'times.dat', out_folder)
	copy2(pca_data_location/'amp_PCA_model.dat', out_folder/'amp_PCA_model')
	copy2(pca_data_location/'ph_PCA_model.dat', out_folder/'ph_PCA_model')

	for i,amp_loc in enumerate(amp_model_locations):
		for file in os.listdir(amp_loc):
			if not file.startswith("amp") and not file.startswith('coefficients'): continue #not a relevant file
			copy2(amp_loc+'/'+file, out_folder)

	for i,ph_loc in enumerate(ph_model_locations):
		for file in os.listdir(ph_loc):
			if not file.startswith("ph") and not file.startswith('coefficients'): continue
			copy2(ph_loc+'/'+file, out_folder)
	
	print("Neural Networks gathered successfully in folder {}".format(out_folder))
	return
