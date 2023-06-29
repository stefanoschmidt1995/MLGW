import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import time

from .ML_routines import PCA_model

class PcaData:
	def __init__(self, PCA_data_location, PC_comp, quant, modes=[], features=[], ratio=1):
		'''
		PC should be an integer/list refering to the amount of PC components to be fitted
		Option to give extra val_data set, should have same amount of components as training data
		
		Currently not yet added:
			multiple modes
			combining multiple datasets
		'''
		
		
		self.data_loc = PCA_data_location
		
		self.times = np.genfromtxt(PCA_data_location+"times.dat")
		self.features = features
		self.PC_comp = PC_comp
		self.quantity = quant
		
		train_theta = np.genfromtxt(PCA_data_location+"PCA_train_theta.dat")
		test_theta = np.genfromtxt(PCA_data_location+"PCA_test_theta.dat")
		
		if quant == 'ph':
			self.pca = PCA_model(PCA_data_location+"ph_PCA_model.dat")
			train_var = np.genfromtxt(PCA_data_location+"PCA_train_ph.dat")
			test_var = np.genfromtxt(PCA_data_location+"PCA_test_ph.dat")
			if len(train_var.shape) == 1:
				train_var = np.reshape(train_var, (train_var.shape[0],1))
				test_var = np.reshape(test_var, (test_var.shape[0],1))
				print(train_var.shape)
		if quant == 'amp':
			self.pca = PCA_model(PCA_data_location+"amp_PCA_model.dat")
			train_var = np.genfromtxt(PCA_data_location+"PCA_train_amp.dat")
			test_var = np.genfromtxt(PCA_data_location+"PCA_test_amp.dat")
		
		if isinstance(PC_comp,int): PC_comp = list(range(PC_comp))
		if ratio == 1:
			self.train_theta = train_theta
			self.test_theta = test_theta
			self.train_var = train_var[:,PC_comp]
			self.test_var = test_var[:,PC_comp]
		if ratio < 1:
			train_indc = random.sample(range(len(train_theta)),round(ratio*len(train_theta)))
			test_indc = random.sample(range(len(test_theta)),round(ratio*len(test_theta)))
			self.train_theta = train_theta[train_indc]
			self.test_theta = test_theta[test_indc]
			self.train_var = train_var[train_indc][:,PC_comp]
			self.test_var = test_var[test_indc][:,PC_comp]

		self.augment_features()
	
	def augment_features(self, theta = [], features = []):
		ret = True
		if len(theta)==0: #augment features of theta_vectors of self
			train_theta = self.train_theta
			test_theta = self.test_theta
			ret = False
		else:
			train_theta = theta
			test_theta = theta #dummy array
		
		if len(features) == 0:
			features = self.features

		#BEWARE OF ORDER
		if "2nd_poly" in features:
			for x in ["00","11","22","01","02","12"]:
				train_theta = np.c_[train_theta, train_theta[:,int(x[0])]*train_theta[:,int(x[1])]]
				test_theta = np.c_[test_theta, test_theta[:,int(x[0])]*test_theta[:,int(x[1])]]
				
		if "sym_mas" in features:
			train_theta = np.c_[train_theta, train_theta[:,0] / (1+train_theta[:,0]**2)]
			test_theta = np.c_[test_theta,  test_theta[:,0] / (1+test_theta[:,0]**2)]
		
		if "eff_spin" in features:
			train_theta = np.c_[train_theta, (train_theta[:,1] + train_theta[:,0]*train_theta[:,2]) / (1 + train_theta[:,0])]
			test_theta = np.c_[test_theta, (test_theta[:,1] + test_theta[:,0]*test_theta[:,2]) / (1 + test_theta[:,0])]
			
		if "eff_spin_powers" in features:
			for x in [1,2,3]:
				train_theta = np.c_[train_theta, ( (train_theta[:,1] + train_theta[:,0]*train_theta[:,2]) / (1 + train_theta[:,0]) )**x ]
				test_theta = np.c_[test_theta, ( (test_theta[:,1] + test_theta[:,0]*test_theta[:,2]) / (1 + test_theta[:,0]) )**x ]
		
		if "sym_mas_powers" in features:
			for x in [1,2,3,4]:
				train_theta = np.c_[train_theta, (train_theta[:,0] / (1+train_theta[:,0]**2))**x]
				test_theta = np.c_[test_theta,  (test_theta[:,0] / (1+test_theta[:,0]**2))**x]
		
		if "eff_spin_sym_mas_2nd_poly" in features:
			eff_spin_train = (train_theta[:,1] + train_theta[:,0]*train_theta[:,2]) / (1 + train_theta[:,0])
			eff_spin_test = (test_theta[:,1] + test_theta[:,0]*test_theta[:,2]) / (1 + test_theta[:,0])
			sym_mas_train = train_theta[:,0] / (1+train_theta[:,0]**2)
			sym_mas_test = test_theta[:,0] / (1+test_theta[:,0]**2)
			train = np.c_[sym_mas_train, eff_spin_train]
			test = np.c_[sym_mas_test, eff_spin_test]
			for x in ["00","11","01"]:
				train_theta = np.c_[train_theta, train[:,int(x[0])] * train[:,int(x[1])]]
				test_theta = np.c_[test_theta, test[:,int(x[0])] * test[:,int(x[1])]]
		
		if "eff_spin_sym_mas_3rd_poly" in features:
			eff_spin_train = (train_theta[:,1] + train_theta[:,0]*train_theta[:,2]) / (1 + train_theta[:,0])
			eff_spin_test = (test_theta[:,1] + test_theta[:,0]*test_theta[:,2]) / (1 + test_theta[:,0])
			sym_mas_train = train_theta[:,0] / (1+train_theta[:,0]**2)
			sym_mas_test = test_theta[:,0] / (1+test_theta[:,0]**2)
			train = np.c_[sym_mas_train, eff_spin_train]
			test = np.c_[sym_mas_test, eff_spin_test]
			for x in ["000","111","001","011"]:
				train_theta = np.c_[train_theta, train[:,int(x[0])] * train[:,int(x[1])] * train[:,int(x[2])]]
				test_theta = np.c_[test_theta, test[:,int(x[0])] * test[:,int(x[1])] * test[:,int(x[2])]]
		
		if "eff_spin_sym_mas_4th_poly" in features:
			eff_spin_train = (train_theta[:,1] + train_theta[:,0]*train_theta[:,2]) / (1 + train_theta[:,0])
			eff_spin_test = (test_theta[:,1] + test_theta[:,0]*test_theta[:,2]) / (1 + test_theta[:,0])
			sym_mas_train = train_theta[:,0] / (1+train_theta[:,0]**2)
			sym_mas_test = test_theta[:,0] / (1+test_theta[:,0]**2)
			train = np.c_[sym_mas_train, eff_spin_train]
			test = np.c_[sym_mas_test, eff_spin_test]
			for x in ["0000","0001","0011","0111","1111"]:
				train_theta = np.c_[train_theta, train[:,int(x[0])] * train[:,int(x[1])] * train[:,int(x[2])] * train[:,int(x[3])]]
				test_theta = np.c_[test_theta, test[:,int(x[0])] * test[:,int(x[1])] * test[:,int(x[2])] * test[:,int(x[3])]]
		
		if "chirp" in features:
			train_theta = np.c_[train_theta, (train_theta[:,0] / (1+train_theta[:,0]**2))**(3/5)]
			test_theta = np.c_[test_theta,  (test_theta[:,0] / (1+test_theta[:,0]**2))**(3/5)]
		
		if "1_inverse" in features:
			for x in ["0","1","2"]:
				train_theta = np.c_[train_theta, 1/train_theta[:,int(x[0])]]
				test_theta = np.c_[test_theta, 1/test_theta[:,int(x[0])]]
				
		if "q_cube" in features:
			train_theta = np.c_[train_theta, train_theta[:,0]**3]
			test_theta = np.c_[test_theta, test_theta[:,0]**3]
		
		if "q_quart" in features:
			train_theta = np.c_[train_theta, train_theta[:,0]**4]
			test_theta = np.c_[test_theta, test_theta[:,0]**4]
		
		if "q_min1inverse" in features:
			for x in ["0"]:
				train_theta = np.c_[train_theta, 1/(train_theta[:,int(x[0])] - 1)]
				test_theta = np.c_[test_theta, 1/(test_theta[:,int(x[0])] - 1)]
		
		if "q_squared" in features:
			for x in ["00"]:
				train_theta = np.c_[train_theta, train_theta[:,int(x[0])]*train_theta[:,int(x[1])]]
				test_theta = np.c_[test_theta, test_theta[:,int(x[0])]*test_theta[:,int(x[1])]]
		
		if "q_inverse" in features:
			for x in ["0"]:
				train_theta = np.c_[train_theta, 1/train_theta[:,int(x[0])]]
				test_theta = np.c_[test_theta, 1/test_theta[:,int(x[0])]]
		
		if "log" in features:
			train_theta = np.c_[train_theta, np.log(train_theta[:,0])]
			test_theta = np.c_[test_theta, np.log(test_theta[:,0])]
		
		if "tan" in features:
			train_theta = np.c_[train_theta, np.tan((np.pi/2) * train_theta[:,1])]
			train_theta = np.c_[train_theta, np.tan((np.pi/2) * train_theta[:,2])]
			test_theta = np.c_[test_theta, np.tan((np.pi/2) * test_theta[:,1])]
			test_theta = np.c_[test_theta, np.tan((np.pi/2) * test_theta[:,2])]

		if ret:
			return train_theta
		else:
			self.train_theta = train_theta
			self.test_theta = test_theta

	def augment_features_2(theta = [], features = []):
		theta = np.atleast_2d(np.asarray(theta))
		if "2nd_poly" in features:
			for x in ["00","11","22","01","02","12"]:
				theta = np.c_[theta, theta[:,int(x[0])]*theta[:,int(x[1])]]
				
		if "sym_mas" in features:
			theta = np.c_[theta, theta[:,0] / (1+theta[:,0]**2)] #not calculated correctly
		
		if "eff_spin" in features:
			theta = np.c_[theta, (theta[:,1] + theta[:,0]*theta[:,2]) / (1 + theta[:,0])]
			
		if "eff_spin_powers" in features:
			for x in [1,2,3]:
				theta = np.c_[theta, ( (theta[:,1] + theta[:,0]*theta[:,2]) / (1 + theta[:,0]) )**x ]
		
		if "sym_mas_powers" in features:
			for x in [1,2,3,4]:
				theta = np.c_[theta, (theta[:,0] / (1+theta[:,0]**2))**x]
		
		if "eff_spin_sym_mas_2nd_poly" in features:
			eff_spin = (theta[:,1] + theta[:,0]*theta[:,2]) / (1 + theta[:,0])
			sym_mas = theta[:,0] / (1+theta[:,0]**2)
			train = np.c_[sym_mas, eff_spin]
			for x in ["00","11","01"]:
				theta = np.c_[theta, train[:,int(x[0])] * train[:,int(x[1])]]
		
		if "eff_spin_sym_mas_3rd_poly" in features:
			eff_spin = (theta[:,1] + theta[:,0]*theta[:,2]) / (1 + theta[:,0])
			sym_mas = theta[:,0] / (1+theta[:,0]**2)
			train = np.c_[sym_mas, eff_spin]
			for x in ["000","111","001","011"]:
				theta = np.c_[theta, train[:,int(x[0])] * train[:,int(x[1])] * train[:,int(x[2])]]
		
		if "eff_spin_sym_mas_4th_poly" in features:
			eff_spin = (theta[:,1] + theta[:,0]*theta[:,2]) / (1 + theta[:,0])
			sym_mas = theta[:,0] / (1+theta[:,0]**2)
			train = np.c_[sym_mas, eff_spin]
			for x in ["0000","0001","0011","0111","1111"]:
				theta = np.c_[theta, train[:,int(x[0])] * train[:,int(x[1])] * train[:,int(x[2])] * train[:,int(x[3])]]
		
		if "chirp" in features:
			theta = np.c_[theta, (theta[:,0] / (1+theta[:,0]**2))**(3/5)]
		
		if "1_inverse" in features:
			for x in ["0","1","2"]:
				theta = np.c_[theta, 1/theta[:,int(x[0])]]
				
		if "q_cube" in features:
			theta = np.c_[theta, theta[:,0]**3]
		
		if "q_quart" in features:
			theta = np.c_[theta, theta[:,0]**4]
		
		if "q_min1inverse" in features:
			for x in ["0"]:
				theta = np.c_[theta, 1/(theta[:,int(x[0])] - 1)]
		
		if "q_squared" in features:
			for x in ["00"]:
				theta = np.c_[theta, theta[:,int(x[0])]*theta[:,int(x[1])]]
		
		if "q_inverse" in features:
			for x in ["0"]:
				theta = np.c_[theta, 1/theta[:,int(x[0])]]
		
		if "log" in features:
			theta = np.c_[theta, np.log(theta[:,0])]
		
		if "tan" in features:
			theta = np.c_[theta, np.tan((np.pi/2) * theta[:,1])]
			theta = np.c_[theta, np.tan((np.pi/2) * theta[:,2])]
		
		return theta

	
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
		indc_train = random.sample(range(N1_train+N2_train),N1_train+N2_train)
		N1_test,N2_test = test_ph_1.shape[0], test_ph_2.shape[0]
		indc_test = random.sample(range(N1_test+N2_test),N1_test+N2_test)
		
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

class LossFunctions:
	def __init__(self, name, weights = [], time_evo=0, exp=2):
		self.weights = weights
		if name == "mean_squared_error":
			self.name = 'mean_squared_error'
			self.LF = tf.keras.losses.mean_squared_error
			self.time_evo = time_evo
		if name == "mean_absolute_error":
			self.name = 'mean_absolute_error'
			self.LF = tf.keras.losses.mean_absolute_error
			self.time_evo = time_evo
		if name == 'mean_squared_logarithmic_error':
			self.name = 'mean_squared_logarithmic_error'
			self.LF = tf.keras.losses.msle
			self.time_evo = time_evo
		if name == 'custom_mse':
			self.name = 'custom_mse'
			self.LF = CustomLoss.custom_MSE_loss(weights)
			self.time_evo=time_evo
		if name == "custom_exp":
			self.name = "custom_exp"
			self.LF = CustomLoss.custom_exp_loss(exp, weights)
			self.time_evo = time_evo

class Schedulers:
	def __init__(self, name, exp=0, decay_epoch = 500):
		def scheduler(epoch, lr):
			if epoch < decay_epoch:
				return lr
			else:
				return lr * math.exp(exp) #exp 0 is just no decay
		
		self.scheduler = scheduler
		self.exp = exp
		self.name = name
		self.decay_epoch = decay_epoch

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
			
			loss = tf.abs((y_true - y_pred) * weights )**exp
			
			return tf.math.reduce_mean(loss, axis=-1)
		
		return loss_function




class NeuralNetwork:
	def __init__(self, PCA_data, quantity, layers_nodes=[10,10,10,10,10], activation = 'sigmoid',
				 optimizer = ('Adam',0), initializer='glorot_uniform', loss_function = ('mean_squared_error',[],0)):
		
		#utility
		self.time = 0 # time it takes to train the model
		self.quantity = quantity
		self.PCA_data = PCA_data
		
		#architecture
		self.layers_nodes = layers_nodes
		self.activation = activation
		self.optimizer = Optimizers(optimizer[0],lr=optimizer[1])
		self.loss_function = LossFunctions(loss_function[0], weights=loss_function[1], time_evo=loss_function[2], exp=loss_function[3])
		
		'''
		self.weights_list = [0]*len(self.loss_function.weights) #make a list of weights like this so they are changeable in callbacks
		for i,x in enumerate(self.loss_function.weights):
			self.weights_list[i] = tf.keras.backend.variable(x)
		'''
		
		K = len(self.PCA_data.train_var[0]) #number of PCA components as output
		L = len(self.PCA_data.train_theta[0]) #number of parameters as input
		
		#build regression model
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(layers_nodes[0],input_shape=(L,), kernel_initializer=initializer, activation=activation))
		for i in range(1,len(layers_nodes)):
			self.model.add(tf.keras.layers.Dense(layers_nodes[i], kernel_initializer=initializer, activation=activation))
		self.model.add(tf.keras.layers.Dense(K,kernel_initializer=initializer,activation='linear'))
		self.model.compile(loss=self.loss_function.LF, optimizer=self.optimizer.opt)
		

	def fit_model(self, max_epochs=5000, Batch_size=500, LRscheduler=Schedulers('exponential')):
		self.batch_size = Batch_size
		self.scheduler = LRscheduler
		
		start = time.time()
		callback_list = []
		early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
		callback_list.append(early_stopping)
		
		LR_scheduler =  tf.keras.callbacks.LearningRateScheduler(LRscheduler.scheduler)
		callback_list.append(LR_scheduler)
		'''
		#don't use for now, does not improve results
		if self.loss_function.time_evo != 0:
			#create callback for changing loss_function
			def call_back_LF(self,epoch):
				if epoch <= 500:
					#for i in range(1,len(self.loss_function.weights)):
					#	tf.keras.backend.set_value(self.weights_list[0],tf.keras.backend.get_value(self.weights_list[0])-1/10000)
					tf.keras.backend.set_value(self.weights_list[0],tf.keras.backend.get_value(self.weights_list[0])*1.011)
			
			callback_list.append(tf.keras.callbacks.LambdaCallback(on_epoch_begin= lambda epoch, log : call_back_LF(self,epoch)))
			print("Succesfully added callback for loss_weights evolution")
		'''	  
		
		self.history = self.model.fit(x=self.PCA_data.train_theta,y=self.PCA_data.train_var,batch_size=Batch_size,
								 validation_data=(self.PCA_data.test_theta,self.PCA_data.test_var),
								 epochs=max_epochs, verbose=0,callbacks=callback_list)
		
		pred_var = self.model.predict(self.PCA_data.test_theta)
		MSE = mean_squared_error(self.PCA_data.test_var,pred_var)
		MSE_PC1 = mean_squared_error(self.PCA_data.test_var[:,0],pred_var[:,0])
		print("The MSE for all principal components of the fit is: ", MSE)
		print("The MSE of first principal component of the fit is: ", MSE_PC1)
		self.time = time.time()-start
		
	def save_model(self, location, dirname, quantity):
		K = len(self.PCA_data.test_var[0]) #number of PCA components
		self.MSE = np.zeros(K)
		os.mkdir(location+dirname)
		
		pred_var = self.model.predict(self.PCA_data.test_theta)
		for i in range(K):
			self.MSE[i]=mean_squared_error(self.PCA_data.test_var[:,i],pred_var[:,i])
		
		self.model_loc = location+dirname
		f = open(location+dirname+"/Model_fit_info.txt", 'x')
		f.write("Generation 2 model for: " + self.PCA_data.quantity+'\n')
		f.write("Trained on dataset: "+self.PCA_data.data_loc+'\n')
		f.write("Features used: ["+", ".join(self.PCA_data.features)+']\n')
		f.write("created model with params: \n")
		f.write("layer_list : ["+",".join([str(x) for x in self.layers_nodes])+']\n')
		f.write("optimizer : " + self.optimizer.name + " with learning rate " + str(self.optimizer.lr)+'\n')
		f.write("activation : " + self.activation + '\n')
		f.write("batch_size : " + str(self.batch_size) + "\n")
		f.write("schedulers : " + self.scheduler.name + " with decay rate " + str(self.scheduler.exp)+'\n')
		f.write("loss_function : " + self.loss_function.name + " with weights " +"["+",".join([str(x) for x in self.loss_function.weights])+"] and time evolution: "+str(self.loss_function.time_evo)+"\n")
		
		for i in range(K):
			f.write("The MSE of principal component "+ str(i) + " of the fit is: "+ str(self.MSE[i])+"\n")
			
		f.write("Time taken for fit: "+str(self.time))
		f.close()
		
		plt.figure('lossfunction')
		plt.title('Loss function of '+self.PCA_data.quantity)
		plt.plot(self.history.history['loss'], label='train')
		plt.plot(self.history.history['val_loss'], label='test')
		plt.yscale('log')
		plt.legend()
		plt.savefig(location+dirname+'/lossfunction.png')
		plt.close(fig='lossfunction')
		self.model.save(location+dirname+'/model.h5') #changed to h5, because else it does not work on cluster
	
	def load_model2(self, weights_location): #don't use
		self.model_loc = "".join(weights_location.split('/')[-2])
		self.model.load_weights(weights_location)
	
	def load_model(model_location, custom_objects = None, as_h5=False):
		if not as_h5:
			#if custom_objects == None: tf.keras.models.load_model(model_location, compile = False)
			return tf.keras.models.load_model(model_location, custom_objects=custom_objects, compile=False)
		else:
			#if custom_objects == None: tf.keras.models.load_model(model_location+'.h5', compile = False)
			return tf.keras.models.load_model(model_location+'.h5', custom_objects=custom_objects, compile=False)


	
	def create_single_model(data_loc, save_loc, dirname, quantity, param_list):
		D = PCAdata_v2.PcaData(data_loc)
		M = NeuralNetwork(D,quantity)
		M.fit_model()
		M.save_model(save_loc,dirname,quantity)
		
	def HyperParametertesting(data_loc, save_loc, quantity, param_dict, PC_comp, epcs = 5000, feat = []):
		'''
			param_dict is a dictionary that can store several types of hyper_parameters
			param_dict = {'layer_list' : [], 'optimizers' : [], 'activation' : [],
						  'batch_size' : [], 'schedulers' : [], 'loss_functions' : []}
			
			if len(param_dict[HP]) == 0, then it will use default value
			if len(param_dict[HP]) == 1, then it will use that value for all models
			if len(param_dict[HP]) > 2, then it will loop over the specified values
			
			code creates a param_list that is essentially a list of lists of length 6
			that indicate which params to be used
		'''
		
		default = {'layer_list' : [10,10,10,10,10],
				   'optimizers' : ('Adam',0),
				   'activation' : 'sigmoid',
				   'batch_size' : 500,
				   'schedulers' : PCAdata_v2.Schedulers('exponential'),
				   'loss_functions' : ('mean_squared_error',[],0)}
		for key in param_dict:
			if len(param_dict[key]) == 0:
				param_dict[key] = [default[key]]
		
		param_list = []
		for a in param_dict['layer_list']:
			for b in param_dict['optimizers']:
				for c in param_dict['activation']:
					for d in param_dict['batch_size']:
						for e in param_dict['schedulers']:
							for f in param_dict['loss_functions']:
								param_list.append([a,b,c,d,e,f])
		
		
		
		for (a,b,c,d,e,f) in param_list:
			print("started creating model with params: \n")
			print("layer_list : ["+",".join([str(x) for x in a])+']')
			print("optimizer : " + b[0] + " with learning rate " + str(b[1]))
			print("activation : " + c)
			print("batch_size : " + str(d))
			print("schedulers : " + e.name + " with decay rate " + str(e.exp))
			print("loss_function : " + f[0] + " with weights " +"["+",".join([str(x) for x in f[1]])+"] and time evolution: "+str(f[2])+"\n")
			dirname = quantity+'_'+"["+",".join([str(x) for x in a])+"]"
			D = PCAdata_v2.PcaData(data_loc, PC_comp, quantity, features=feat)
			M = NeuralNetwork(D, quantity ,layers_nodes=a, activation = c, optimizer = b, loss_function = f)
			M.fit_model(Batch_size = d, LRscheduler = e, max_epochs = epcs)
			M.save_model(save_loc, dirname, quantity)
			del D
			del M
	
	def ContinueTraining(data_loc, save_loc, quantity, PC, model_loc, model_param_dict, feat=[], epcs = 5000):
		'''
			model_param_dict is of the form:
				{layer_list : [], activation : str, optimizer : (str,float),
				 loss_function : (str,[]), batch_size : int, scheduler : PCAdataV2.loss_functions('exponential')}
			and should correspond to the model that is continued.
			
			should implement: concatenation of new training set and old data set (what do do with PCA?)
			
		'''
		
		D = PCAdata_v2.PcaData(data_loc, PC, quantity, features=feat)
		M = NeuralNetwork(D,quantity,
						  layers_nodes=model_param_dict['layer_list'],
						  activation=model_param_dict['activation'],
						  optimizer=model_param_dict['optimizer'],
						  loss_function=model_param_dict['loss_function'])
		M.model = NeuralNetwork.load_model(model_loc)
		print("model loaded and initialized, now starting fit")
		M.fit_model(Batch_size = model_param_dict['batch_size'],
					LRscheduler = model_param_dict['scheduler'],
					max_epochs=epcs)
		M.save_model(save_loc, model_loc.split('/')[-2]+"_updated",quantity)
		print('model ftted and saved')

	def CreateResidualPCAsets(PCA_data, pred, model_loc, save_loc, q, components=1):
		times = PCA_data.times
		train_theta = PCA_data.train_theta[:,:3]
		test_theta = PCA_data.test_theta[:,:3]
		train_pred = pred[0][:,:components]
		test_pred = pred[1][:,:components]
		
		new_train_var = PCA_data.train_var[:,:components] - train_pred
		new_test_var = PCA_data.test_var[:,:components] - test_pred
		
		#normalize the datasets
		norm_coef = []
		for i in range(components):
			test_max = np.max(abs(new_test_var[:,i]))
			new_train_var[:,i] /= test_max
			new_test_var[:,i] /= test_max
			norm_coef.append(test_max)

		
		if q == 'ph':
			PCA_data.pca.save_model(save_loc+"ph_PCA_model.dat")
			np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
			np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
			np.savetxt(save_loc+"PCA_train_ph.dat", new_train_var)
			np.savetxt(save_loc+"PCA_test_ph.dat", new_test_var)
			
		if q == 'amp':
			PCA_data.pca.save_model(save_loc+"amp_PCA_model.dat")
			np.savetxt(save_loc+"PCA_train_theta.dat", train_theta)
			np.savetxt(save_loc+"PCA_test_theta.dat", test_theta)
			np.savetxt(save_loc+"PCA_train_amp.dat", new_train_var)
			np.savetxt(save_loc+"PCA_test_amp.dat", new_test_var)
		
		np.savetxt(save_loc+"times.dat", times)
		
		f = open(save_loc+"/info.txt", 'x')
		f.write("Model location: "+model_loc+"\n")
		f.write("Test scale coefficients: "+", ".join(str(x) for x in norm_coef))
		f.close()
		
		plt.figure('new_pca PC1')
		plt.title('delta pca/pred for test data, PC1')
		plt.scatter(test_theta[:,0], new_test_var[:,0])
		plt.savefig(save_loc+'/delta pca-pred.png')
		plt.close(fig='new_pca') 
