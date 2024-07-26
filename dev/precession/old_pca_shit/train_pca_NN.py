import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.NN_model import CustomLoss, Schedulers
from mlgw.precession_helper import residual_network_angles
import numpy as np
import tensorflow as tf
import tensorflow.data
import time
import numpy as np
import os
from keras.callbacks import EarlyStopping, LearningRateScheduler


def resnet_loss(weights, epsilon = 0.1):

	def loss_function(y_true, y_pred):

		y_pred = (tf.matmul(tf.multiply(red_y_pred, scale), W) + mu)
		
		#plt.figure()
		#plt.plot(y_true[0])
		#plt.plot(y_pred[0].numpy())
		#plt.show()
		
		#return tf.math.reduce_mean(tf.abs(y_true - y_pred), axis = -1)
		
		return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)
		
	return loss_function

def augment_1d(theta):
	#TODO: put this into a keras layer
#	return np.column_stack([theta[:,0], theta[:,7], theta[:,8], theta[:,0]**2, theta[:,0]**3, np.log(theta[:,0])])
	#return theta[:,[0]]#,7,8]]
	
	q = theta[:,0]
	frequencies = np.logspace(0, 1.4, 10)
	
	return np.column_stack([q, *[np.cos(f*q) for f in frequencies]])

def augment_5d(theta):
	
	frequencies = np.logspace(0, 1.4, 100)
	q = theta[:,0]
	s1 = theta[:,1]
	
	extra_features = np.array([[np.cos(f*var) for f in frequencies] for var in theta[:,:5].T]).T
	extra_features = extra_features.reshape((extra_features.shape[0], extra_features.shape[1]*extra_features.shape[2]))
	
	return np.column_stack([theta[:,:5], *extra_features.T])

#TODO:
#	- Serious feature augmentation
#		Expand the number of frequencies
#		Learn the frequencies...
#	- New architecture:
#		PC(x) = B(x) + A(x)*cos(phi(x))
#		where A,B, phi are NNs
#		
#		Now is PC(x) = B(x)

#	- Waveform prediction
#		WF(t, x) = B(t, x) + A(t, x)*cos(phi(t, x))
#		using cos(t) is a good idea
#		WF(t,x) = B(t,x)


#########################

#SOLUTION TO ALL OF YOUR PROBLEMS:
# f_ref = f_start
# For some reason, this avoid any problem you may have with discontinuities!! Not sure exactly why, but since we don't care about which f_ref, this is totally fine for us :D

#TAKE-AWAYS
# - Using cos(q) features solves all of your problems! Maybe you may also consider going to high PC dimensions
# - This can scale up to several dimension?
# - You can also write the fwd methdod as:
#		f(x) = b(x) + A(x)*cos(phi(x))
#	This is the generalization of the thing. If the 5d model does not work, you can explore this in 1D


dirname = 'pca_model_1d'
dirname = 'pca_model_1d_alpha_aligned'
dirname = 'pca_model_1d_f_ISCO_fstart_const'
dirname = 'pca_model_1d_fref_eq_fstart_const'
#dirname = 'pca_model_1d_fref_eq_fstart_const_lots_of_PC'
#dirname = 'pca_model_qs1s2t1t2'

angle_name = 'beta'
load =  not True
fit = not load
plot = True

train_theta =  np.loadtxt('{}/PCA_train_theta_angles.dat'.format(dirname))
test_theta = np.loadtxt('{}/PCA_test_theta_angles.dat'.format(dirname))
train_angle = np.loadtxt('{}/PCA_train_{}.dat'.format(dirname, angle_name))#[:,[0,1,2]]
test_angle = np.loadtxt('{}/PCA_test_{}.dat'.format(dirname, angle_name))#[:,[0,1,2]]

if False:
	train_angle = train_angle[train_theta[:,0]>1.5]
	train_theta = train_theta[train_theta[:,0]>1.5]
	test_angle = test_angle[test_theta[:,0]>1.5]
	test_theta = test_theta[test_theta[:,0]>1.5]

train_theta = augment_1d(train_theta)
test_theta = augment_1d(test_theta)

	###########
	# Model generation

	#Making the model by hand
layers = [50, 100, 200, 500]
layers_regression = [tf.keras.Input((train_theta.shape[1],))]
layers_regression.extend([tf.keras.layers.Dense(units= l, activation='relu') for l in layers])
layers_regression.append(tf.keras.layers.Dense(units= train_angle.shape[1]))
model = tf.keras.Sequential(layers_regression)

pca = PCA_model()
pca.load_model('{}/pca_{}'.format(dirname, angle_name))
loss_weights = np.sqrt(np.array(pca.get_eigenvalues()))
loss_weights = loss_weights / min(loss_weights)
#loss = CustomLoss.custom_MSE_loss(loss_weights)
loss = tf.keras.losses.MeanSquaredError() #tf.keras.losses.MeanAbsoluteError())	
	
optimizer = tf.keras.optimizers.Adam(1e-4)
optimizer_res = tf.keras.optimizers.Adam(1e-4)
metric = None if isinstance(loss, tf.keras.losses.MeanSquaredError) else 'mse'
#model.compile(optimizer=optimizer, optimizer_res=optimizer_res, loss = loss, loss_res = tf.keras.losses.MeanSquaredError(), metrics = metric)
model.compile(optimizer=optimizer, loss = loss, metrics = metric)

if load:
	#model = tf.keras.model.load_model('{}/model_keras'.format(dirname))
	model.load_weights('{}/model_{}/model'.format(dirname, angle_name))

if fit:
	model.fit(train_theta, train_angle,
			epochs=80,
			shuffle=True,
			batch_size = 2048,
			callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
					LearningRateScheduler(Schedulers('exponential', exp=-0.003, min_lr = 5e-5).scheduler)],
			validation_data=(test_theta, test_angle))

if not load: model.save_weights('{}/model_{}/model'.format(dirname, angle_name))


#########
if plot:
	K_to_plot = 5 #train_angle.shape[1]

	fig, axes = plt.subplots(K_to_plot, 1, sharex = True, figsize = (6.4, 4.8*K_to_plot/3))
	fig_res, axes_res = plt.subplots(K_to_plot, 1, sharex = True, figsize = (6.4, 4.8*K_to_plot/3))
	
	if train_angle.shape[1] == 1:
		axes = [axes]
		axes_res = [axes_res]
	fig.suptitle("Angle {}".format(angle_name))
	fig_res.suptitle("Residual {}".format(angle_name))
	
	rec_test_angle =  model(test_theta).numpy()
	
	ids_ = np.argsort(test_theta[:,0])
	
	for k, (ax, ax_res, angle, rec_angle) in enumerate(zip(axes, axes_res, test_angle.T, rec_test_angle.T)):
		mse = np.mean(np.square(rec_angle[ids_] - angle[ids_]))
		ax.set_title('K = {} - mse = {}'.format(k, mse))
		ax_res.set_title('K = {} - mse = {}'.format(k, mse))
		
		ax.scatter(test_theta[ids_,0], angle[ids_], label = 'true', s = 1)
		ax.scatter(test_theta[ids_,0], rec_angle[ids_], label = 'rec', s = 1)
		
		ax_res.scatter(test_theta[ids_,0], angle[ids_] - rec_angle[ids_], label = 'residuals', s = 1)
		
	axes[-1].legend()
	axes[-1].set_xlabel('q')
	fig.tight_layout()

	axes_res[-1].legend()
	axes_res[-1].set_xlabel('q')
	fig_res.tight_layout()
	
	pca_model = PCA_model()
	pca_model.load_model('{}/pca_{}'.format(dirname, angle_name))
	time_grid = np.loadtxt('{}/time_grid.dat'.format(dirname))

	rec_angle_time = pca_model.reconstruct_data(rec_test_angle)
	true_angle_time = pca_model.reconstruct_data(test_angle)
	mse = np.mean(np.square(true_angle_time - rec_angle_time))
	
	fig, axes = plt.subplots(2,1, sharex = True)
	plt.suptitle("Rec angles\nalphamse for angle {}: {}".format(angle_name, mse))
	for i in range(10):
		axes[0].plot(time_grid, true_angle_time[i,:], c = 'orange', label = 'true')
		axes[0].plot(time_grid, rec_angle_time[i,:], c = 'blue', label = 'rec')
		axes[1].plot(time_grid, rec_angle_time[i,:] - true_angle_time[i,:], c = 'green', label = 'rec - true')
		if not i: axes[0].legend()
	plt.tight_layout()
	
	plt.figure()
	plt.title('angle[0]')
	plt.plot(test_theta[ids_,0], true_angle_time[ids_,0])
	plt.plot(test_theta[ids_,0], rec_angle_time[ids_,0], label = 'rec')
	plt.legend()
	
	plt.show()
		

















