###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from DenseMoE import *
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#from try_tf_mismatch import *

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset.dat", shuffle = True) #loading dataset

	#adding extra features for non linear regression
extra_features = np.stack((np.multiply(theta_vector[:,0], theta_vector[:,1]), np.multiply(theta_vector[:,2], theta_vector[:,1]),  np.multiply(theta_vector[:,0], theta_vector[:,2])))
theta_vector = np.concatenate((theta_vector, extra_features.T), axis = 1)

print("Loaded "+ str(theta_vector.shape[0])+" data")

	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

		#DOING PCA
print("#####PCA#####")
K_ph = 5 #30 apparently works well for PCA...
print("   K = ",K_ph, " | N_grid = ", test_ph.shape[1])
	#phase
ph_PCA = PCA_model()
E = ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
print("PCA eigenvalues: ", E)

red_train_ph = ph_PCA.reduce_data(train_ph)
red_test_ph = ph_PCA.reduce_data(test_ph)
rec_PCA_test_ph = ph_PCA.reconstruct_data(red_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_test_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase with PCA: ",error_ph)

plt.figure(1)
plt.title("Phase with PCA")
for i in range(1):
	plt.plot(frequencies, test_ph[i,:], label = 'true')
	plt.plot(frequencies, rec_PCA_test_ph[i,:], label = 'reconstructed')
plt.legend()

F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_PCA_test_ph)
#print("Mismatch PCA: ",F_PCA)
print("Mismatch PCA avg: ",np.mean(F_PCA))

#plt.show()
#quit()

	#preprocessing data
logreg_ph = logreg_model(test_theta.shape[1],red_train_ph.shape[1], False)

red_train_ph, weights = logreg_ph.preprocess_data(red_train_ph)
red_test_ph = logreg_ph.preprocess_data(red_test_ph)[0]

	#defining loss function (trying to...)
input_pred = tf.compat.v1.placeholder(tf.float64)
input_true = tf.compat.v1.placeholder(tf.float64)
out = tf.compat.v1.placeholder(tf.float32)

tf_F_loss = mismatch_function(logreg_ph.get_prep_constants(), ph_PCA.get_PCA_params(), train_amp)
#tf_F_loss = tf.compat.v1.numpy_function(np_mse, [input_true, input_pred], [tf.float64])
#tf_F_loss = tf.compat.v1.numpy_function(np_mismatch_function(logreg_ph.get_prep_constants(), ph_PCA.get_PCA_params(), test_amp), [input_true, input_pred], [tf.float64])

#quit()

different_regressions = True

	#doing MoE
n_experts = 100
N_epochs = 200

if different_regressions:
	in_layers = []
	out_layers = []
	models = []
	red_fit_ph = np.zeros(red_test_ph.shape)
	print("Doing MoE model for each component")
	for i in range(K_ph):
		in_layers.append(Input(shape=(train_theta.shape[1],)))
		out_layers.append(DenseMoE(1, n_experts, expert_activation='linear', gating_activation='softmax',
							gating_kernel_initializer_scale=np.std(red_train_ph[:,i])/np.sqrt(train_theta.shape[1]),
							expert_kernel_initializer_scale=np.std(red_train_ph[:,i])/np.sqrt(train_theta.shape[1]))(in_layers[i]))
		models.append(Model(inputs=in_layers[i], outputs=out_layers[i]))
		models[i].compile(optimizer = 'rmsprop', loss = 'mse')
		models[i].fit(x=train_theta, y=red_train_ph[:,i], batch_size=64, epochs=N_epochs, validation_split = 0.1,shuffle=True, verbose=0)
		print("\ttrain model loss for comp "+str(i)+":", models[i].evaluate(train_theta, red_train_ph[:,i], verbose=0))
		print("\ttest model loss for comp "+str(i)+" :", models[i].evaluate(test_theta, red_test_ph[:,i], verbose =0))
		red_fit_ph[:,i]= np.reshape(models[i].predict(test_theta), (red_fit_ph.shape[0],))
	red_fit_ph = logreg_ph.un_preprocess_data(red_fit_ph)

if not different_regressions:
	print("Doing MoE model")
	inputs = Input(shape=(train_theta.shape[1],))
	hidden2 = DenseMoE(K_ph, n_experts, expert_activation='linear', gating_activation='softmax',
							             expert_kernel_initializer_scale=np.std(red_train_ph[:,0]))(inputs)

	model = Model(inputs=inputs, outputs=hidden2)
	model.compile(optimizer = 'rmsprop', loss = 'mse')#tf_F_loss)
	model.summary()
	#model.compile(optimizer = 'rmsprop',
	#		loss = (lambda item1, item2: tf.numpy_function(np_mismatch_function, [item1, item2],
	#														[tf.float32] ) ) ) 
	history = model.fit(x=train_theta, y=red_train_ph[:,:], batch_size=64, epochs=N_epochs, validation_split = 0.1,shuffle=True, verbose=0)
	print("train model loss ", model.evaluate(train_theta, red_train_ph, verbose=0))
	print("test model loss ", model.evaluate(test_theta, red_test_ph, verbose =0))
		#un_preprocessing data
	red_fit_ph = logreg_ph.un_preprocess_data(model.predict(test_theta)) #for single model

red_test_ph = logreg_ph.un_preprocess_data(red_test_ph) #un-preprocessing test labels

for i in range(3):
	plt.figure(i+3)
	comp = 0
	plt.title("Data component #"+str(comp)+" vs param "+str(i))
	plt.plot(test_theta[:,i], red_fit_ph[:,comp], 'o',label = 'fitted', ms = 4)
	plt.plot(test_theta[:,i], red_test_ph[:,comp], 'o',label = 'true', ms = 4)
	#plt.plot(test_theta[:,i], red_fit_ph[:,1]-red_test_ph[:,1], 'o',label = 'difference')
	plt.legend()

#plt.show()
#plt.quit()

#apparently here there is a serious problem of underfitting. According to noise estimator error_ph noise is around 0.1 (and mismatch is consistent with that). The problem is that the proper noise est doesn't give a meaningful noise (and also residuals are not 0 mean) probably residuals are not gaussians.

noise_est = np.divide(red_test_ph - red_fit_ph, red_test_ph)
print("Fit reconstruction error for reduced coefficients: ", np.mean(noise_est), np.std(noise_est))

rec_fit_ph = ph_PCA.reconstruct_data(red_fit_ph)
error_ph = np.linalg.norm(test_ph - rec_fit_ph, ord= 'fro')/(test_ph.shape[0])#*np.std(test_ph))
print("Fit reconstruction error for phase: ", error_ph)

plt.figure(2)
plt.title("Phase with FIT")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true |' + str(np.round(test_theta[i,0],2))+","+ str(np.round(test_theta[i,1],2))+","+ str(np.round(test_theta[i,2],2)))
	plt.plot(frequencies, rec_fit_ph[i,:], label = 'fit')
plt.legend()

F = compute_mismatch(train_amp[0,:], test_ph, train_amp[0,:], rec_fit_ph) #ty if it's the same F as test!!!
#F = compute_mismatch(test_amp, test_ph, test_amp, rec_fit_ph)
#F = compute_mismatch(np.ones(test_ph.shape), test_ph, np.ones(test_ph.shape), rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

plt.show()






























