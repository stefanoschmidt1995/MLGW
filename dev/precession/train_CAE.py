import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.NN_model import CustomLoss, Schedulers
import numpy as np
import tensorflow as tf
import tensorflow.data
import time
import numpy as np
import os
from keras.callbacks import EarlyStopping, LearningRateScheduler
from CAE import CAE_1D

def plot_mse_as_k(pca_model, dataset, K_min, K_max, time_grid = None, savedir = None):
	reduced_dataset = pca_model.reduce_data(dataset)
	
	plt.figure()
	for k in range(K_min, K_max+1):
		rec_dataset = pca_model.reconstruct_data(reduced_dataset, K = k)
		mse = np.mean(np.square(rec_dataset - dataset))
		plt.scatter(k, mse)

	plt.xlabel('# PC')
	plt.ylabel('Validation mse')
	plt.yscale('log')
	if savedir: plt.savefig('{}/pca_residuals.png'.format(savedir))
	
	if time_grid is not None:
		fig, axes = plt.subplots(2,1, sharex = True)
		plt.suptitle('K = {}'.format(K_max))
		for id_ in range(1,9):
			axes[0].plot(time_grid, rec_dataset[id_], c='orange')
			axes[0].plot(time_grid, dataset[id_], c='blue')
			axes[1].plot(time_grid, rec_dataset[id_]-dataset[id_])

		if savedir: plt.savefig('{}/pca_rec_beta.png'.format(savedir))

def PCA_loss(W, mu, scale = 1.):
	W, mu, scale = tf.transpose(tf.constant(W, dtype = tf.float32)), tf.constant(mu, dtype = tf.float32), tf.constant(scale, dtype = tf.float32)
	# returns the custom loss function given the weights
	def loss_function(y_true, red_y_pred):

		y_pred = (tf.matmul(tf.multiply(red_y_pred, scale), W) + mu)
		
		#plt.figure()
		#plt.plot(y_true[0])
		#plt.plot(y_pred[0].numpy())
		#plt.show()
		
		#return tf.math.reduce_mean(tf.abs(y_true - y_pred), axis = -1)
		
		return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)
		
	return loss_function

def mse_derivative_loss(time_grid, alpha, beta = 0):
	time_grid = tf.constant(time_grid, dtype = tf.float32)
	
	#FIXME: how to normalize the derivative???
	
	def loss_function(y_true, y_pred):

		dydx_true = tf.experimental.numpy.diff(y_true)#/tf.experimental.numpy.diff(time_grid)
		dydx_pred = tf.experimental.numpy.diff(y_pred)#/tf.experimental.numpy.diff(time_grid)
		
		d2ydx2_true = tf.experimental.numpy.diff(dydx_true)
		d2ydx2_pred = tf.experimental.numpy.diff(dydx_pred)
		
		
		mean_derivative = tf.math.reduce_mean(tf.square(dydx_true))
		#dydx_true /= mean_derivative
		#dydx_pred /= mean_derivative
		
		mse_ = tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)
		mse_derivative_ = tf.math.reduce_mean(tf.square(dydx_true - dydx_pred), axis=-1)
		mse_2nd_derivative_ = tf.math.reduce_mean(tf.square(d2ydx2_true - d2ydx2_pred), axis=-1)

		if False:
			print(mse_, mse_derivative_, mse_2nd_derivative_)
			print(mse_, alpha * mse_derivative_, beta * mse_2nd_derivative_)
			
			print(alpha, beta)
			quit()

			plt.figure()
			plt.title('Angle')
			plt.plot(y_pred[0], label = 'pred')
			plt.plot(y_true[0], label = 'true')
			plt.legend()
			
			plt.figure()
			plt.title('Derivative')
			plt.plot(dydx_pred[0], label = 'pred')
			plt.plot(dydx_true[0], label = 'true')
			plt.legend()
			plt.show()
		
		return mse_ + alpha * mse_derivative_ + beta * mse_2nd_derivative_
	return loss_function

################################################################################################################################

dirname = 'NN_convolution_only_q_mse_loss'
dirname = 'NN_PCA_residual'
load = False
os.makedirs(dirname, exist_ok = True)

dataset_file = 'datasets/angle_dataset_merged_10_11_12.dat'
dataset_file = 'tiny_angle_dataset.dat'

dataset_file = 'datasets_only_q/angle_dataset_only_q.dat'
dataset_file = 'datasets_only_q/merged_angle_dataset_only_q.dat'
dataset_file = 'angle_dataset_only_q.dat'

theta, alpha_dataset, cosbeta_dataset, time_grid = load_dataset(dataset_file, N_data=None, N_entries = 2, N_grid = None, shuffle = False, n_params = 9)
theta = theta[:,[0]]

	###########
	## CAE model for the betas

model_type = 'NN_PCA_residual'
#model_type = 'NN'

train_theta, test_theta, train_cosbeta, test_cosbeta = make_set_split(theta, cosbeta_dataset, train_fraction = .85, scale_factor = None)
input_dim = train_cosbeta.shape[-1]

#train_cosbeta = tf.data.Dataset.from_tensors(train_cosbeta[...,None]) #you may also use from_tensor_slices. which one is better depends on the loss
#test_cosbeta = tf.data.Dataset.from_tensors(test_cosbeta[...,None])

optimizer = tf.keras.optimizers.Adam(5e-4)

	#Training model
if model_type == 'autoencoder':
	model = CAE_1D(10, input_dim)
	model.compile(optimizer=optimizer, loss = tf.keras.losses.MeanSquaredError())
	model.fit(train_cosbeta, train_cosbeta,
		            epochs=3,
		            shuffle=True,
		            batch_size = 300,
		            callbacks = [EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)],
		            validation_data=(test_cosbeta, test_cosbeta))	

	final_mse = tf.keras.losses.MeanSquaredError()(test_cosbeta, model(test_cosbeta)).numpy()
	rec_test_cosbeta = model(test_cosbeta).numpy()
elif model_type == 'NN':
	#TODO: You shall do it convolutional later on. Maybe...
	#TODO: You should remove some features from the thetas
	
	model = tf.keras.Sequential(
		[
			tf.keras.layers.InputLayer(input_shape=(train_theta.shape[-1],)),
			tf.keras.layers.Dense(units= 64, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 1024, activation=tf.nn.relu),
			tf.keras.layers.Reshape(target_shape=(64, 1)),
			tf.keras.layers.Conv1DTranspose(
					filters=64, kernel_size=10, strides=2, padding='same',
					activation='relu'),
			tf.keras.layers.Conv1DTranspose(
					filters=32, kernel_size=100, strides=5, padding='same',
					activation='relu'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(input_dim),
		]
	)
	
	loss = tf.keras.losses.MeanSquaredError() #tf.keras.losses.MeanAbsoluteError())
	loss = mse_derivative_loss(time_grid, 10, 0)
	
	#t = tf.constant(train_cosbeta[None, 0,:], dtype = tf.float32)
	#loss(t, t*np.random.normal(1,0.00001, size = train_cosbeta.shape[1]))
	
	metric = None if isinstance(loss, tf.keras.losses.MeanSquaredError) else 'mse'
	model.compile(optimizer=optimizer, loss = loss, metrics = metric)
	
	if load:
		#model = tf.keras.model.load_model('{}/model_keras'.format(dirname))
		model.load_weights('{}/model.keras'.format(dirname))
		
		loss(tf.constant(test_cosbeta[None, 0, :], dtype = tf.float32), model(test_theta[0]))
		
		
		#model.save('tmp_model')
	else:
		model.fit(train_theta, train_cosbeta,
		            epochs=10,
		            shuffle=True,
		            batch_size = 64,
		            callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
					            LearningRateScheduler(Schedulers('exponential', exp=-0.0003, min_lr = 5e-5).scheduler)],
		            validation_data=(test_theta, test_cosbeta))

	final_mse = tf.keras.losses.MeanSquaredError()(test_cosbeta, model(test_theta)).numpy()

	rec_test_cosbeta = model(test_theta).numpy()

elif model_type == 'NN_PCA_residual':
	K = 2

	pca = PCA_model()
	pca_file = '{}/pca_model'.format(dirname)
	
	if os.path.isfile(pca_file):
		pca.load_model(pca_file)
	else:
		pca.fit_model(train_cosbeta[:4000], K = K)
		print('PCA eigenvalues: ', pca.get_eigenvalues())
		#plot_mse_as_k(pca, test_cosbeta[:1000], 0, K, time_grid, savedir = dirname)
		
		pca.save_model(pca_file)
	
	plot_mse_as_k(pca, test_cosbeta[:1000], 0, K, time_grid, savedir = dirname)


	if not False:
		train_cosbeta_ = np.repeat(train_cosbeta[[0,1]], 100, axis = 0)
		test_cosbeta = np.repeat(train_cosbeta[[0,1]], 20, axis = 0)
		train_theta_ = np.repeat(train_theta[[0,1]], 100, axis = 0)
		test_theta = np.repeat(train_theta[[0,1]], 20, axis = 0)
		train_theta, train_cosbeta = train_theta_, train_cosbeta_
		
		ids_ = np.random.permutation(len(test_theta))
		test_theta, test_cosbeta = test_theta[ids_], test_cosbeta[ids_]
		
		print(train_cosbeta.shape, train_theta.shape)
	
	
	train_res_cosbeta = (train_cosbeta - pca.reconstruct_data(pca.reduce_data(train_cosbeta)))*100
	test_res_cosbeta = (test_cosbeta - pca.reconstruct_data(pca.reduce_data(test_cosbeta)))*100
	
	if True:
		#This seems to perform much much better
		model = tf.keras.Sequential(
		[
			tf.keras.layers.InputLayer(input_shape=(train_theta.shape[-1],)),
			tf.keras.layers.Dense(units= 32, activation=tf.nn.relu),
			tf.keras.layers.Reshape(target_shape=(32, 1)),
			
			#tf.keras.layers.Dense(units= 32, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 64, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 256, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 256*2, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 1024, activation=tf.nn.relu),
			#tf.keras.layers.Dense(input_dim),

			#tf.keras.layers.LSTM(512),
			#tf.keras.layers.Reshape(target_shape=(512, 1)),
			tf.keras.layers.LSTM(input_dim),
			tf.keras.layers.Dense(input_dim),
		]
		)
	
	else:
		model = tf.keras.Sequential(
		[
			tf.keras.layers.InputLayer(input_shape=(train_theta.shape[-1],)),
			tf.keras.layers.Dense(units= 64, activation=tf.nn.relu),
			#tf.keras.layers.Dense(units= 1024, activation=tf.nn.relu),
			tf.keras.layers.Reshape(target_shape=(64, 1)),
			tf.keras.layers.Conv1DTranspose(
					filters=64, kernel_size=10, strides=2, padding='same',
					activation='relu'),
			tf.keras.layers.Conv1DTranspose(
					filters=32, kernel_size=100, strides=5, padding='same',
					activation='relu'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(input_dim),
		]
		)
	
	loss = tf.keras.losses.MeanSquaredError() #tf.keras.losses.MeanAbsoluteError())
	loss = mse_derivative_loss(time_grid, 0.5, .0001)
	
	metric = None if isinstance(loss, tf.keras.losses.MeanSquaredError) else 'mse'
	model.compile(optimizer=optimizer, loss = loss, metrics = metric)
	
	if load:
		#model = tf.keras.model.load_model('{}/model_keras'.format(dirname))
		model.load_weights('{}/model_weights.keras'.format(dirname))
		#model.save('tmp_model')
	#else:

	#loss(tf.constant(test_res_cosbeta[None, 0, :], dtype = tf.float32), model(test_theta[0])); quit()

	model.fit(train_theta, train_res_cosbeta,
		            epochs=100,
		            shuffle=True,
		            batch_size = 64,
		            callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
					            LearningRateScheduler(Schedulers('exponential', exp=-0.0003, min_lr = 5e-5).scheduler)],
		            validation_data=(test_theta, test_res_cosbeta))

	rec_test_cosbeta = model(test_theta).numpy()/100. + pca.reconstruct_data(pca.reduce_data(test_cosbeta))
	rec_test_res_cosbeta = model(test_theta).numpy()
	final_mse = tf.keras.losses.MeanSquaredError()(test_cosbeta, rec_test_cosbeta).numpy()
	final_res_mse = tf.keras.losses.MeanSquaredError()(test_res_cosbeta, rec_test_res_cosbeta).numpy()


elif model_type == 'PCA_NN':

	K = 2

	pca = PCA_model()
	pca_file = '{}/pca_model'.format(dirname)
	
	if os.path.isfile(pca_file):
		pca.load_model(pca_file)
	else:
		pca.fit_model(train_cosbeta[:4000], K = K)
		print('PCA eigenvalues: ', pca.get_eigenvalues())
		#plot_mse_as_k(pca, test_cosbeta[:1000], 0, K, time_grid, savedir = dirname)
		
		pca.save_model(pca_file)
	
	plot_mse_as_k(pca, test_cosbeta[:1000], 0, K, time_grid, savedir = dirname)
	
	train_red_cosbeta = pca.reduce_data(train_cosbeta)
	test_red_cosbeta = pca.reduce_data(test_cosbeta)
	
	loss_weights = np.sqrt(np.array(pca.get_eigenvalues()))
	loss_weights = loss_weights / min(loss_weights)
	loss_function = CustomLoss.custom_MSE_loss(loss_weights)

	#loss_function = PCA_loss(*pca.PCA_params[:3])
	
	#loss_function(test_cosbeta[:1], test_red_cosbeta[:1]) #test that it's fine...

	model = tf.keras.Sequential(
		[
			tf.keras.layers.InputLayer(input_shape=(train_theta.shape[-1],)),
			tf.keras.layers.Dense(units= 1000, activation=tf.nn.relu),
			tf.keras.layers.Dense(units= 500, activation=tf.nn.relu),
			tf.keras.layers.Dense(units= 250, activation=tf.nn.relu),
			tf.keras.layers.Dense(units= 100, activation=tf.nn.relu),
			tf.keras.layers.Dense(K),
		]
	)
	
	model.compile(optimizer=optimizer, loss = loss_function)
	
	if load:
		model.load_weights('{}/model.keras'.format(dirname))
	else:
		model.fit(train_theta, train_red_cosbeta,
		            epochs=10,
		            shuffle=True,
		            batch_size = 5,
		            callbacks = [EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True),
					            LearningRateScheduler(Schedulers('exponential', exp=-0.0003, min_lr = 5e-5).scheduler)],
		            validation_data=(test_theta, test_red_cosbeta))

	mse_ = np.mean(np.square(model(test_theta).numpy()- test_red_cosbeta), axis = 0)
	for i in range(K):
		ids_ = np.argsort(train_theta[:,0])
		plt.figure()
		plt.title('K = {}\nmse = {}'.format(i, mse_[i]))
		plt.scatter(test_theta[:,0], model(test_theta).numpy()[:,i], c = 'blue', label = 'pred')
		plt.scatter(test_theta[:,0], test_red_cosbeta[:,i], c = 'orange', label = 'true')
		plt.plot(train_theta[ids_,0], train_red_cosbeta[ids_,i], c = 'red')
		plt.legend()
		plt.savefig('{}/PC_comp_error_{}.png'.format(dirname, i))
	plt.show()

	rec_test_cosbeta = pca.reconstruct_data(model(test_theta).numpy()).astype(np.float64)
	final_mse = tf.keras.losses.MeanSquaredError()(test_cosbeta, rec_test_cosbeta).numpy()

else:
	raise ValueError("Model type not understood")

model.save_weights('{}/model_weights.keras'.format(dirname))
model.save('{}/model_keras'.format(dirname))

print('Final mse: {}'.format(final_mse))


N_plot = 10
plt.figure()
plt.title("Overall mse: {}".format(final_mse))
for cosbeta_rec, cosbeta in zip(rec_test_cosbeta[:N_plot], test_cosbeta[:N_plot]):
	plt.plot(cosbeta_rec, c='orange')
	plt.plot(cosbeta, c='blue')
plt.savefig('{}/reconstruction_beta.png'.format(dirname))

if model_type == 'NN_PCA_residual':
	
	plt.figure()
	plt.title("Residual cos(beta)\nres mse: {}".format(final_res_mse))
	for cosbeta_rec, cosbeta in zip(rec_test_res_cosbeta[:N_plot], test_res_cosbeta[:N_plot]):
		plt.plot(cosbeta_rec, c='orange')
		plt.plot(cosbeta, c='blue')
	plt.savefig('{}/reconstruction_beta_residuals.png'.format(dirname))

plt.show()


quit()

####################################################################################################################################


	###########
	## PCA model for the alphas??
	
pca = PCA_model()
pca.fit_model(alpha_dataset[:val_id], K = 20)
print(pca.get_eigenvalues())

rec_alpha = pca.reconstruct_data(pca.reduce_data(alpha_dataset[val_id:]))
mse = np.mean(np.square(rec_alpha - alpha_dataset[val_id:]))
print(mse)


plt.figure()
for k in range(1,20):
	rec_alpha = pca.reduce_data(alpha_dataset[val_id:])
	rec_alpha[:,k:] = 0.
	rec_alpha = pca.reconstruct_data(rec_alpha)
	mse = np.mean(np.square(rec_alpha - alpha_dataset[val_id:]))
	plt.scatter(k, mse)

	if False:
		fig, axes = plt.subplots(2,1, sharex = True)
		plt.suptitle('K = {}'.format(k))
		for id_ in range(1,5):
			axes[0].plot(time_grid, rec_alpha[id_], c='orange')
			axes[0].plot(time_grid, alpha_dataset[val_id+id_], c='blue')
			axes[1].plot(time_grid, rec_alpha[id_]-alpha_dataset[val_id+id_])
		plt.show()

plt.xlabel('# PC')
plt.ylabel('Validation mse')
plt.yscale('log')
plt.show()


N_plot = 20
fig, axes = plt.subplots(2,1, sharex = True)
for cosbeta_ in cosbeta_dataset[:N_plot]:
	axes[0].plot(time_grid, cosbeta_)
	axes[1].plot(time_grid, cosbeta_)

axes[1].set_ylim([0.9,1.01])
plt.xlabel(r'Time (s/M_sun)')
plt.tight_layout()
plt.show()

	

if not True:
	N_plot = 20
	fig, axes = plt.subplots(2,1, sharex = True)
	for alpha_, cosbeta_ in zip(alpha_dataset[:N_plot], cosbeta_dataset[:N_plot]):
		axes[0].plot(time_grid, alpha_)
		axes[1].plot(time_grid, cosbeta_)
		
	plt.xlabel(r'Time (s/M_sun)')
	plt.tight_layout()
	plt.show()
