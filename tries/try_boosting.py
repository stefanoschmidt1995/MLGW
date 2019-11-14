###################
#	Some tries of fitting GW generation model using PCA+ boosting
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from DenseMoE import *
from keras.models import Model
from keras.layers import Input, Dense
import keras.backend as K
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("./datasets/GW_std_dataset_small_grid.dat", shuffle = False) #loading dataset

print("#####Pre-processing#####")
print("Loaded "+ str(theta_vector.shape[0])+" data")

	#making data with a less frequent sampling
indices = np.arange(0, ph_dataset.shape[1], ph_dataset.shape[1]/(512)).astype(int)
#print(indices)
frequencies = frequencies[indices]
ph_dataset = ph_dataset[:,indices]
amp_dataset = amp_dataset[:,indices]
print("New shape: ", amp_dataset.shape)

#amp_dataset = process_amplitudes(frequencies, theta_vector[:,0], amp_dataset, False)
#ph_dataset = process_phases(frequencies, theta_vector[:,0], ph_dataset, False)
#amp_dataset = process_amplitudes(frequencies, theta_vector[:,0],amp_dataset, True)
#ph_dataset = process_phases(frequencies, theta_vector[:,0],ph_dataset, True)

	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

		#DOING PCA
print("#####PCA#####")
K_ph = 30 #30 apparently works well for PCA...
print("   K = ",K_ph, " | N_grid = ", test_ph.shape[1])
	#phase
ph_PCA = PCA_model()
E = ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
print("PCA eigenvalues: ", E)

red_train_ph = ph_PCA.reduce_data(train_ph)
red_test_ph = ph_PCA.reduce_data(test_ph)
rec_PCA_test_ph = ph_PCA.reconstruct_data(red_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_test_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase: ",error_ph)

plt.figure(0)
plt.title("Phase with PCA")
for i in range(1):
	plt.plot(frequencies, test_ph[i,:], label = 'true')
	plt.plot(frequencies, rec_PCA_test_ph[i,:], label = 'reconstructed')
plt.legend()

F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_PCA_test_ph)
print("Mismatch PCA avg: ",np.mean(F_PCA))


print("#####BOOSTING#####")
#Here we perform boosting applying sequentially a MoE model using keras layer in DenseMoe.py

M = 5 #number of iteration
n_experts = 30 #n_experts (for every model)
N_epochs = 100 #training epochs (for every model)
gamma = .99 #damping factor for prediction... the m-th prediction is weighted with a factor gamma^m

	#layers for the model
prep_list = [] #list of preprocessing routines
model_list = [] #list of keras models
input_layer_list = []
hidden_layer_list = []

y_train = red_train_ph
y_test = red_test_ph

	#doing training
for m in range(M):
	prep_list.append(logreg_model(test_theta.shape[1],red_train_ph.shape[1], False) )
	y_train = prep_list[m].preprocess_data(y_train)[0]
	y_test = prep_list[m].preprocess_data(y_test)[0]

	#tf_F_loss = mismatch_function(prep_list[m].get_prep_constants(), ph_PCA.get_PCA_params(), train_amp) #custom loss function

		#creating model...
	input_layer_list.append(Input(shape=(train_theta.shape[1],)) )
	hidden_layer_list.append(DenseMoE(K_ph, n_experts, expert_activation='linear', gating_activation='softmax')(input_layer_list[m]) )
	model_list.append(Model(inputs=input_layer_list[m], outputs=hidden_layer_list[m]))
	model_list[m].compile(optimizer = 'rmsprop', loss = 'mse')
	model_list[m].fit(x=train_theta, y=y_train, batch_size=64, epochs=N_epochs, validation_split = 0.,shuffle=True, verbose=0)
	print("  train model loss at iteration "+str(m)+": ", model_list[m].evaluate(train_theta, y_train, verbose=0))
	print("  test model loss at iteration "+str(m)+": ", model_list[m].evaluate(test_theta, y_test, verbose =0))

		#plotting fit results
	comp = 0
	plt.figure(1)
	plt.title("Fit result for every iteration")
	plt.plot(test_theta[:,0], y_test[:,comp], 'o',label = 'true it '+str(m), ms = 4)
	plt.plot(test_theta[:,0], model_list[m].predict(test_theta)[:,comp], 'o',label = 'fitted it '+str(m), ms = 4)
	plt.legend()
	#plt.show()

		#updating training set
	y_train = y_train - model_list[m].predict(train_theta)
	y_test = y_test - model_list[m].predict(test_theta)

	#doing test
y_test = np.zeros(red_test_ph.shape)
for m in range(M-1,-1,-1):
	#y_test = y_test + prep_list[m].un_preprocess_data(model_list[m].predict(test_theta))
	y_test = prep_list[m].un_preprocess_data(gamma * y_test + model_list[m].predict(test_theta))

	#computing reconstruction error and mismatch
error_ph = np.linalg.norm(red_test_ph - y_test, ord= 'fro')/(test_ph.shape[0]*np.std(y_test))
noise_est = np.divide(y_test - red_test_ph, y_test)
print("Fit reconstruction error for reduced coefficients: ", error_ph,np.mean(noise_est), np.std(noise_est))

rec_fit_ph = ph_PCA.reconstruct_data(y_test)

error_ph = np.linalg.norm(test_ph - rec_fit_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Fit reconstruction error for phase: ", error_ph)

	#doing un-preprocessing
#test_amp = process_amplitudes(frequencies, test_theta[:,0],test_amp, True)
#test_ph = process_phases(frequencies, test_theta[:,0],test_ph, True)
#rec_fit_ph = process_phases(frequencies, test_theta[:,0], rec_fit_ph, True)

F = compute_mismatch(test_amp, test_ph, test_amp, rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

plt.figure(50)
plt.title("Phase with fit")
for i in range(3):
	plt.plot(frequencies, test_ph[i,:], label = 'true')
	plt.plot(frequencies, rec_fit_ph[i,:], label = 'reconstructed')
	#plt.xscale("log")
	#plt.yscale("log")
plt.legend()

for i in range(3):
	plt.figure(i+100)
	comp = 0
	plt.title("Data component #"+str(comp)+" vs param "+str(i))
	plt.plot(test_theta[:,i], y_test[:,comp], 'o',label = 'fitted', ms = 4)
	plt.plot(test_theta[:,i], red_test_ph[:,comp], 'o',label = 'true', ms = 4)
	plt.legend()

plt.show()




