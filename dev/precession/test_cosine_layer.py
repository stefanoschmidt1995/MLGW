import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.NN_model import CustomLoss, Schedulers
from mlgw.precession_helper import residual_network_angles, angle_params_keeper, angle_manager
import numpy as np
import tensorflow as tf
import tensorflow.data
import time
import numpy as np
import os
from keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler #If this proves to be useful, you need to implement this by yourself
from itertools import combinations, permutations, product
from scipy.stats import binned_statistic_2d
from tqdm import tqdm
from tqdm.keras import TqdmCallback

import joblib

from train_reduced_angles_NN import CosinesLayer

######################

def loss(y_true, y_pred):
	squared_difference = tf.math.square(y_true - y_pred[...,1,tf.newaxis])
	return tf.math.reduce_mean(squared_difference, axis=-1)


def get_model(layers, in_shape, out_shape, cosines = None, lr = 1e-3):
	layers_regression = [tf.keras.Input((in_shape,))]
	if cosines: layers_regression.append(CosinesLayer(units = cosines))

	layers_regression.extend([tf.keras.layers.Dense(units= l, activation='sigmoid') for l in layers])

	model = tf.keras.Sequential(layers_regression)
	
	print(model.summary())

	#loss = tf.keras.losses.MeanSquaredError() #tf.keras.losses.MeanAbsoluteError())	
		
	optimizer = tf.keras.optimizers.Adam(lr)
	model.compile(optimizer=optimizer, loss = loss, metrics = None)

	return model


	

############################

in_shape = 1
freqs = np.array([3])
ph = 3
f = lambda x: np.cos(np.matmul(x, freqs)+ph)

model = get_model([], in_shape, 1, cosines = 1, lr = 1e-3)

t = np.linspace(-3,3, 1000)[:,None]
fs = np.fft.rfft(f(t))

id_ = np.argmax(np.abs(fs))

f_guess = np.fft.rfftfreq(len(t), t[1]-t[0])[id_]*(2*np.pi)
ph_guess = np.pi-np.angle(np.conj(fs[id_]))

if False:
	X_grid = np.stack(np.meshgrid(np.linspace(-3,3, 100),np.linspace(-3,3, 100)), axis = -1)

	Y_grid = f(X_grid)
	print(X_grid.shape, Y_grid.shape)
	FS = np.fft.rfftn(Y_grid, Y_grid.shape)

	ids_ = np.argmax(np.abs(np.fft.fftshift(FS)))
	ids_x, ids_y = ids_//100, ids_-(ids_//100)*100

	f_guess = [np.fft.rfftfreq(100, 6/100)[[ids_x, ids_y]]]
	print(f_guess) 

	plt.imshow(np.log(np.abs(np.fft.fftshift(FS))**2))
	plt.show()

	quit()

train_theta = np.random.uniform(-3, 3, (100000, in_shape))
test_theta = np.random.uniform(-3, 3, (10000, in_shape))
train_y = f(train_theta)
test_y = f(test_theta)

f_guess, ph_guess = 2.2, np.random.uniform(0, 2*np.pi)
model.layers[0].set_weights([np.array([[f_guess]]), np.array([ph_guess]) ])

print('FFT guess: ', f_guess, ph_guess)
print('True: ', freqs, ph)
for layer in model.layers: print(layer.get_weights()[0].T, layer.get_weights()[1])

model.fit(train_theta, train_y,
	epochs=200, shuffle=True, verbose = 0, batch_size = 2048*4,
	callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
				LearningRateScheduler(Schedulers('exponential', exp=-0.0003, min_lr = 5e-5).scheduler),
				TqdmCallback(verbose = 0)],
	validation_data=(test_theta, test_y))

for layer in model.layers: print(layer.get_weights()[0].T, layer.get_weights()[1])


if in_shape == 1:
	plt.plot(t[:,0], model(t)[:,0])
	plt.plot(t[:,0], model(t)[:,1], label = 'fit')
	plt.plot(t[:,0], f(t), label = 'true')
	plt.legend()
	plt.show()
	quit()

quit()

x = np.linspace(0, 10, 1000)
plt.plot(x, np.cos(x))
plt.plot(x, np.arccos(np.cos(x)))
plt.show();quit()































