"""
Train a simple ML model to make a regression for the angles:

To fit the model:
	python train_reduced_angles_NN.py --run-name only_q --layers 256 64 --n-cosines 32 --fit --epochs 1000

To just make some plots:
	python train_reduced_angles_NN.py --run-name only_qt1


The general case:
	python train_reduced_angles_NN.py --run-name only_qs1s2t1t2phi1 --dataset dataset/angle_dataset_only_qs1s2t1t2phi1.dat --progress-to-file --layers 256 128 64 32 16 --epochs  10000 --fit --vars-to-fit qs1s2t1t2phi1 --augment-data
	
	python train_reduced_angles_NN.py --run-name only_qs1s2t1t2phi1  --vars-to-fit qs1s2t1t2phi1 --augment-data

"""

import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.NN_model import CustomLoss, Schedulers
from mlgw.precession_helper import residual_network_angles, angle_params_keeper, angle_manager, get_S_effective_norm, get_alpha0_beta0_gamma0, CosinesLayer, augment_for_angles
import numpy as np
import tensorflow as tf
import tensorflow.data
import time
import numpy as np
import os
import sys
from keras.callbacks import EarlyStopping, LearningRateScheduler
	#Interesting: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
	#If this proves to be useful, you need to implement this by yourself
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer
from itertools import combinations, permutations, product
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import joblib

import argparse

import psutil
import gc

######################################################################################

class MemoryUsageCallback(tf.keras.callbacks.Callback):
  '''Monitor memory usage on epoch begin and end.'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss*1e-9))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss*1e-9))

class GarbageRemoval(tf.keras.callbacks.Callback):
	
	def on_epoch_end(self, epoch, logs=None):
		tf.keras.backend.clear_session()
		gc.collect()

def plot_contour(test_theta, values, values_label, x_labels, bins):
	fsize = 4*test_theta.shape[1]-1
	fs = 10
	fig, axes = plt.subplots(test_theta.shape[1]-1, test_theta.shape[1]-1, figsize = (fsize, fsize))
	fig.suptitle('{}: {}'.format(args.run_name, values_label))
	if test_theta.shape[1]-1 == 1:
		axes = np.array([[axes]])
	for i,j in permutations(range(test_theta.shape[1]-1), 2):
		if i<j:	axes[i,j].remove()

		#Plot the datapoints
	for ax_ in combinations(range(test_theta.shape[1]), 2):
		currentAxis = axes[ax_[1]-1, ax_[0]]
		ax_ = list(ax_)
		
		stat, x_edges, y_edges, binnumber = binned_statistic_2d(test_theta[:,ax_[0]], test_theta[:,ax_[1]], values = values,
			statistic = 'mean', bins = bins)
		X, Y = np.meshgrid(x_edges,y_edges)
		mesh = currentAxis.pcolormesh(X, Y, stat.T)
		
		cbar = plt.colorbar(mesh, ax = currentAxis)
		if values_label:
			cbar.set_label(r'$\log_{10}(\mathrm{'+values_label.replace('_', '\_')+r'})$', rotation=270, labelpad = 15)

		if ax_[0] == 0:
			currentAxis.set_ylabel(x_labels[ax_[1]], fontsize = fs)
		else:
			currentAxis.set_yticks([])
		if ax_[1] == test_theta.shape[1]-1:
			currentAxis.set_xlabel(x_labels[ax_[0]], fontsize = fs)
		else:
			currentAxis.set_xticks([])
		currentAxis.tick_params(axis='x', labelsize=fs)
		currentAxis.tick_params(axis='y', labelsize=fs)

######################################################################################

class TrendAmpPhaseLayer(tf.keras.layers.Layer):
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim

	def call(self, inputs):
		assert inputs.shape[-1]==3*self.output_dim, "Wrong shape for the inputs! It must be 3*{}".format(self.output_dim)
		return inputs[...,:self.output_dim] + inputs[...,self.output_dim:2*self.output_dim]*tf.math.cos(inputs[...,2*self.output_dim:])

def get_model(layers, in_shape, out_shape, cosines = None, lr = 1e-3, frequencies = (1,10)):
	layers_regression = [tf.keras.Input((in_shape,))]

	if cosines: layers_regression.append(CosinesLayer(units = cosines, frequencies = frequencies))

	layers_regression.extend([tf.keras.layers.Dense(units= l, activation='sigmoid') for l in layers])
	
	layers_regression.append(tf.keras.layers.Dense(units= out_shape, activation = 'linear'))
	
	#layers_regression.append(tf.keras.layers.Dense(units= 3*out_shape, activation = 'linear'))
	#layers_regression.append(TrendAmpPhaseLayer(output_dim = out_shape))
	
	#layers_regression.append(tf.keras.layers.Normalization(axis=-1, mean=None, variance=None, invert=True))

	model = tf.keras.Sequential(layers_regression)
	
	print(model.summary())

	loss = tf.keras.losses.MeanSquaredError() #tf.keras.losses.MeanAbsoluteError())	
		
	optimizer = tf.keras.optimizers.Adam(lr)
	metric = None if isinstance(loss, tf.keras.losses.MeanSquaredError) else 'mse'
	model.compile(optimizer=optimizer, loss = loss, metrics = metric)

	return model

########################################################################

if __name__ == '__main__':

	parser = argparse.ArgumentParser(__doc__)

	parser.add_argument(
		"--dataset-file", type = str, required = False,
		help="File for the dataset")
	parser.add_argument(
		"--dirname", type = str, required = False, default = 'NN_models',
		help="Name of the directory where to save the models")
	parser.add_argument(
		"--run-name", type = str, required = True,
		help="Name of the run")
	parser.add_argument(
		"--residual", action = 'store_true',
		help="Whether to use a residual model. If the model is not available, it will be created")
	parser.add_argument(
		"--fit", action = 'store_true',
		help="Whether to train the trend model")
	parser.add_argument(
		"--layers", type = int, required = False, default = [100, 150, 250], nargs = '+',
		help="Layers of the ANN")
	parser.add_argument(
		"--n-cosines", type = int, required = False, default = 0,
		help="Number of cosines in the first layer of the residual model")
	parser.add_argument(
		"--residual-layers", type = int, required = False, default = [100, 150, 250], nargs = '+',
		help="Layers of the residual ANN")
	parser.add_argument(
		"--epochs", type = int, required = False, default = 800,
		help="Number of epochs")
	parser.add_argument(
		"--limit-clustering", action = 'store_true',
		help="Whether to limit the parameter space using the clustering model (experimental)")
	parser.add_argument(
		"--limit-parameters-cut", action = 'store_true',
		help="Whether to limit the parameter space with some cuts (experimental)")
	parser.add_argument(
		"--progress-to-file", action = 'store_true',
		help="Whether to redirect the progress bar to file")
	parser.add_argument(
		"--augment-data", action = 'store_true',
		help="Whether to augment the data with some extra handy features")
	parser.add_argument(
		"--vars-to-fit", type = str, required = True,
		help="Variables to include in the fit, e.g. qt1 or qs1s2t1t2phi1")
	parser.add_argument(
		"--include-fref", action = 'store_true',
		help="Include reference phase among the input features.")
	args = parser.parse_args()

	#TODO: plot mse in the angle reconstruction!!

	#ids_to_fit = [0, 3, 7]
	#ids_to_fit = [0, 1, 2, 3, 4, 7]
	#ids_to_fit = [0, 1, 2, 3, 4, 7]
	
	dict_ids_to_fit = {
		'q': [0],
		'qt1': [0, 1],
		'qs1s2t1t2': [0, 1, 2, 3, 4],
		'qs1s2t1t2phi1': [0, 1, 2, 3, 4, 5],
		'qs1s2t1t2phi1phi2': [0, 1, 2, 3, 4, 5, 6],
	}
	ids_to_fit = dict_ids_to_fit[args.vars_to_fit]
	if args.include_fref: ids_to_fit.append(7)

	labels = ['q', 's1', 's2', 't1', 't2', 'phi1', 'phi2', 'fref']
	labels = [labels[i] for i in ids_to_fit]
	print('Fitting: ', labels)

	dirname = args.dirname
	os.makedirs('{}/model_{}'.format(dirname, args.run_name), exist_ok = True)
	test_angles_error =  True

	dataset_file = args.dataset_file if args.dataset_file else 'datasets/angle_dataset_{}.dat'.format(args.run_name)

	train_fraction = 0.8

	theta, targets, _ = load_dataset(dataset_file, N_entries = 1, N_grid = None, shuffle = False, n_params = 8)
	targets = targets[:,[0,1,3]] #removing residuals for beta & alpha0 & gamma0
	base_theta, frefs = np.array(theta[0]), np.array(theta[:,7])
	
	if args.limit_clustering:
		
		assert targets.shape[-1] == 4
		
		#Doing clustering reduction! Very very unphysical: just to see how it goes...
		print(dataset_file)
		model = joblib.load(dirname+'/cluster_model_full_K4.gz')
		
		X = np.concatenate([theta, targets], axis = 1)
		
		labels = model.predict(X)
		
		#print([sum(labels == i) for i in range(4)])
		
		ids_, = np.where(labels != 3)
		del X
		old_len = len(theta)
		theta, targets = theta[ids_], targets[ids_]
		
		print('Limiting with clustering from {} to {}'.format(old_len, len(theta)) )

	if args.limit_parameters_cut:
	
		chi_P_2D_norm, ids_s1_p, ids_s2_p = get_S_effective_norm(theta)
		ids_, = np.where(theta[ids_s1_p,1]>0.1) #s1> 0.1 and ids_s1_p
		ids_ = ids_s1_p[ids_]
		old_len = len(theta)
		theta, targets = theta[ids_], targets[ids_]
		#theta, targets = np.delete(theta, ids_, axis = 0), np.delete(targets, ids_, axis = 0)
		print('Limiting with parameter space cuts from {} to {}'.format(old_len, len(theta)) )

	theta = theta[:, ids_to_fit]
	if args.augment_data: theta = augment_for_angles(theta)

	N_train = int(train_fraction*len(theta))

	train_theta, train_targets = theta[:N_train], targets[:N_train]
	test_theta, test_targets = theta[N_train:], targets[N_train:]

	if args.fit:
		scaler = RobustScaler(quantile_range=(10.0, 90.0))
		#scaler = MinMaxScaler()
		#scaler = QuantileTransformer()
		scaler.fit(train_targets)
	else:
		scaler = joblib.load('{}/model_{}/scaler.gz'.format(dirname, args.run_name))

	train_targets, test_targets = scaler.transform(train_targets), scaler.transform(test_targets)

	print("Train | test data = {0} | {2}\nInput | output features {1} | {3}".format(*train_theta.shape, test_theta.shape[0], train_targets.shape[-1]))
	
	#pip install tensorflow==2.12.0


	if args.fit:
		
		model = get_model(args.layers, train_theta.shape[1], targets.shape[1], cosines = args.n_cosines, frequencies = (1,10), lr = 1e-3)
		
		#with tf.keras.utils.custom_object_scope({'CosinesLayer': CosinesLayer}):
		#	model = tf.keras.models.load_model('{}/model_{}/model.keras'.format(dirname, args.run_name)); print('Loading the model')
		
		fh = open('{}/model_{}/train_output.err'.format(dirname, args.run_name), 'w') if args.progress_to_file else sys.stderr
		
		history = model.fit(train_theta, train_targets,
				epochs=args.epochs,
				shuffle=True,  verbose = 0,
				batch_size = 2048*4,
				callbacks = [EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
						LearningRateScheduler(Schedulers('exponential', exp=-0.001, decay_epoch = 30, min_lr = 1e-4).scheduler),
						TqdmCallback(verbose = 0, file = fh), GarbageRemoval()],#, MemoryUsageCallback()],
				validation_data=(test_theta, test_targets))

		os.makedirs('{}/model_{}'.format(dirname, args.run_name), exist_ok = True)
		model.save('{}/model_{}/model.keras'.format(dirname, args.run_name))
		joblib.dump(scaler, '{}/model_{}/scaler.gz'.format(dirname, args.run_name))


	else:

		with tf.keras.utils.custom_object_scope({'CosinesLayer': CosinesLayer}):
			model = tf.keras.models.load_model('{}/model_{}/model.keras'.format(dirname, args.run_name))
			
		#model.set_weights([np.array([[ 3. , 0.08,  15 ]], dtype = np.float32),
		#	np.array([0., 0., 0.1 ], dtype = np.float32)])
		

		#model.load_weights('{}/model_{}/model'.format(dirname, args.run_name))

	#for layer in model.layers: print(layer.get_config(), layer.get_weights())

	if args.residual:
		#train_theta_aug = augment_for_angles(train_theta)
		#test_theta_aug = augment_for_angles(test_theta)
		
		res_model_file = '{}/model_{}/residual_model.keras'.format(dirname, args.run_name)
		
		if os.path.isfile(res_model_file):
			with tf.keras.utils.custom_object_scope({'CosinesLayer': CosinesLayer}):
				model_res = tf.keras.models.load_model(res_model_file)
			residual_scaler = joblib.load('{}/model_{}/residual_scaler.gz'.format(dirname, args.run_name))
		else:	

			residual_scaler = RobustScaler(quantile_range=(5.0, 95.0))
			residual_scaler.fit(train_targets - model(train_theta).numpy())
			joblib.dump(residual_scaler, '{}/model_{}/residual_scaler.gz'.format(dirname, args.run_name))
		
			model_res = get_model(args.residual_layers, test_theta.shape[1], targets.shape[1], cosines = args.n_cosines, frequencies = (1,10), lr = 1e-3)

			model_res.fit(train_theta, residual_scaler.transform(train_targets - model(train_theta).numpy()),
				epochs=args.epochs,
				shuffle=True, verbose = 0,
				batch_size = 2048*4,
				callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
						LearningRateScheduler(Schedulers('exponential', exp=-0.0003, decay_epoch = 800,  min_lr = 5e-5).scheduler),
						TqdmCallback(verbose = 0)],
				validation_data=(test_theta, residual_scaler.transform(test_targets - model(test_theta).numpy())))

			model_res.save(res_model_file)

		#print('freqs: ', model_res.layers[0].get_weights()[0])

		rec_test_targets =  model(test_theta).numpy() + residual_scaler.inverse_transform(model_res(test_theta).numpy())

	else:
		rec_test_targets =  model(test_theta).numpy()

	#print("MSE Train | Test = {0} | {1}".format(np.mean(np.square(model(train_theta) - train_targets)),
	#		np.mean(np.square(rec_test_targets - test_targets))))
	#print("Non-scaled MSE Train | Test = {0} | {1}".format(
	#	np.mean(np.square(scaler.inverse_transform(model(train_theta)) - scaler.inverse_transform(train_targets))),
	#	np.mean(np.square(scaler.inverse_transform(rec_test_targets) - scaler.inverse_transform(test_targets)))))

	test_targets, rec_test_targets = scaler.inverse_transform(test_targets), scaler.inverse_transform(rec_test_targets)

	Psi = angle_params_keeper(test_targets)
	rec_Psi = angle_params_keeper(rec_test_targets)
	
	if test_angles_error:
		t_coal, mtot = 2., 20.
		gen = mlgw.GW_generator()
		times = np.linspace(-t_coal*mtot, 0.01, int(t_coal*mtot+0.01)*4096)
		manager = angle_manager(gen, times, 5, 5, beta_residuals = True)
		
		mse_alpha, mse_beta, mse_gamma = [], [], []
		
		for i, (t_, t_rec_, theta_) in enumerate(zip(tqdm(test_targets), rec_test_targets, test_theta)):
		
			#if i>100: break
		
			base_theta[ids_to_fit] = theta_[:len(ids_to_fit)]
			if 7 not in ids_to_fit: base_theta[7] = frefs[N_train:][i] #setting fref appropriately
			print(base_theta.tolist())

			q, s1, s2, t1, t2, phi1, phi2, fstart = base_theta
			s1x, s1y, s1z = s1*np.sin(t1)*np.cos(phi1), s1*np.sin(t1)*np.sin(phi1), s1*np.cos(t1)
			s2x, s2y, s2z = s2*np.sin(t2)*np.cos(phi2), s2*np.sin(t2)*np.sin(phi2), s2*np.cos(t2)
			
			theta_ = np.array([20*q/(1+q), 20/(1+q), s1x, s1y, s1z, s2x, s2y, s2z])

			manager.fref, manager.fstart = fstart, fstart

			L, _ = manager.get_L(theta_)
			
			alpha_true, beta_true, gamma_true = manager.get_alpha_beta_gamma(theta_, t_)
			alpha_rec, beta_rec, gamma_rec = manager.get_alpha_beta_gamma(theta_, t_rec_)
			
			mse_alpha.append(np.mean(np.square(alpha_true- alpha_rec)))
			mse_beta.append(np.mean(np.square(beta_true- beta_rec)))
			mse_gamma.append(np.mean(np.square(gamma_true- gamma_rec)))


			if not False:		
				print('###########')
				print('\ttheta, fstart = ', theta_.tolist(), fstart)
				print('\tPsi true: ', t_.tolist())
				print('\tPsi rec: ', t_rec_.tolist())
				#print([(a,b) for a,b, in zip(t_, t_rec_)])
				print('\tmse alpha, beta, gamma: ',mse_alpha[-1], mse_beta[-1], mse_gamma[-1])

				fig, axes = plt.subplots(3,1, sharex=True)
				axes[0].set_title('alpha')
				axes[0].plot(times, alpha_true, label = 'true')
				axes[0].plot(times, alpha_rec, label = 'rec')
				axes[1].set_title('Residual alpha')
				axes[1].plot(times, alpha_rec - alpha_true, c= 'k')
				axes[2].set_title('Beta')
				axes[2].plot(times, beta_true, label = 'true')
				axes[2].plot(times, beta_rec, label = 'rec')
				axes[2].legend()
				
				plt.tight_layout()
				
				plt.show()

		for values, y_label in zip([mse_alpha, mse_beta, mse_gamma], ['alpha', 'beta', 'gamma']):
			values = np.log10(values)
			plot_contour(test_theta[:len(values)], values, y_label, labels, 100)
		

	if train_theta.shape[1] == 1:

		ids_sort = np.argsort(test_theta[:,0])
		#for attr in ['A_beta', 'B_beta', 'A_alpha', 'B_alpha', 'amp_beta', 'A_ph_beta', 'B_ph_beta', 'ph0_beta', 'alpha0']:
		for attr in ['A_beta', 'A_alpha', 'B_alpha']:
		#for attr in ['amp_beta', 'A_ph_beta', 'B_ph_beta', 'ph0_beta']:
			fig, axes = plt.subplots(2,1, sharex = True)
			axes[0].plot(test_theta[ids_sort, 0], getattr(Psi, attr)[ids_sort], lw = 1,  label = 'true')
			axes[0].plot(test_theta[ids_sort, 0], getattr(rec_Psi, attr)[ids_sort], lw = 1,  label = 'fit')
			axes[0].set_ylabel(attr)
			axes[1].plot(test_theta[ids_sort, 0], getattr(Psi, attr)[ids_sort] - getattr(rec_Psi, attr)[ids_sort], lw = 1,  label = 'true')
			axes[1].set_ylabel(attr+' - residuals')
			axes[1].set_xlabel(labels[0])
			axes[0].legend()

	else:
		fsize = 4*test_theta.shape[1]-1
		fs = 10
		plot_dir = '{}/model_{}/plots/'.format(dirname, args.run_name)
		os.makedirs(plot_dir, exist_ok = True)
		for attr in ['A_beta', 'A_alpha', 'B_alpha']:
			val = getattr(Psi, attr)
			rec_val = getattr(rec_Psi, attr)
		
			values = np.log10(np.abs((rec_val - val)/val))
			
			plot_contour(test_theta[:,:len(ids_to_fit)], values, attr, labels, 100)
			plt.savefig(plot_dir+attr+'.pdf', transparent = True)
		
	plt.show()
	quit()


	keys = [('A_ph_beta', 'A_alpha'), ('B_ph_beta', 'B_alpha')]
	for k in keys:
		plt.figure()
		plt.scatter(getattr(Psi, k[0]), getattr(Psi, k[1]), s = 3, label = 'true')
		plt.scatter(getattr(rec_Psi, k[0]), getattr(rec_Psi, k[1]), s = 3, label = 'fit')
		plt.xlabel(str(k[0]))
		plt.ylabel(str(k[1]))
		plt.legend()

