import numpy as np
import matplotlib.pyplot as plt

import precession_helper as p

import sys
sys.path.insert(0,'../mlgw_v2')
import os
from GW_helper import *
from fit_model import *
from ML_routines import *
from GW_generator import *

def fit_create_PCA_angle_dataset(K, dataset_file, out_folder, train_frac = 0.75):
	if not os.path.isdir(out_folder): #check if out_folder exists
		try:
			os.mkdir(out_folder)
		except:
			raise RuntimeError("Impossible to create output folder "+str(out_folder)+". Please, choose a valid folder.")
			return

	if not out_folder.endswith('/'):
		out_folder = out_folder + "/"

	print("Loading dataset from: ", dataset_file)
	print("Saving PCA dataset to: ", PCA_dataset_folder)
	#FIXME: shuffle must be set to True ultimately
	theta_vector, alpha_dataset, beta_dataset, times = load_dataset(dataset_file, shuffle=False, n_params = 6) #loading dataset
	print("Loaded datataset with shape: "+ str(beta_dataset.shape))

	train_theta, test_theta, train_alpha, test_alpha = make_set_split(theta_vector, alpha_dataset, train_frac, 1.)
	train_theta, test_theta, train_beta, test_beta   = make_set_split(theta_vector, beta_dataset, train_frac, 1.)
	
	if type(K) is int:
		K = (K,K)
	if type(K) is not tuple:
		raise RuntimeError("Wrong format for number of component K. Tuple expected but got "+str(type(K)))

	PCA_beta = PCA_model()
	E_beta = PCA_beta.fit_model(train_beta, K[1], scale_PC=True)
	print("PCA eigenvalues for beta: ", E_beta)
	red_train_beta = PCA_beta.reduce_data(train_beta)			#(N,K) to save in train dataset 
	red_test_beta = PCA_beta.reduce_data(test_beta)			#(N,K) to save in test dataset
	rec_test_beta = PCA_beta.reconstruct_data(red_test_beta) 	#(N,D) for computing mismatch

		#amplitude
	PCA_alpha = PCA_model()
	E_alpha = PCA_alpha.fit_model(train_alpha, K[0], scale_PC=True)
	print("PCA eigenvalues for alpha: ", E_alpha)
	red_train_alpha = PCA_alpha.reduce_data(train_alpha)			#(N,K) to save in train dataset 
	red_test_alpha = PCA_alpha.reduce_data(test_alpha)			#(N,K) to save in test dataset
	rec_test_alpha = PCA_alpha.reconstruct_data(red_test_alpha) 	#(N,D) for computing mismatch
	
			#saving to files
	PCA_alpha.save_model(out_folder+"alpha_PCA_model")				#saving amp PCA model
	PCA_beta.save_model(out_folder+"beta_PCA_model")				#saving beta PCA model
	np.savetxt(out_folder+"PCA_train_theta.dat", train_theta)	#saving train theta
	np.savetxt(out_folder+"PCA_test_theta.dat", test_theta)		#saving test theta
	np.savetxt(out_folder+"PCA_train_alpha.dat", red_train_alpha)	#saving train reduced amplitudes
	np.savetxt(out_folder+"PCA_test_alpha.dat", red_test_alpha)		#saving test reduced amplitudes
	np.savetxt(out_folder+"PCA_train_beta.dat", red_train_beta)		#saving train reduced phases
	np.savetxt(out_folder+"PCA_test_beta.dat", red_test_beta)		#saving test reduced phases
	np.savetxt(out_folder+"times", times)						#saving times
	
	return

def create_PCA_angle_dataset(PCA_model_alpha, PCA_model_beta, N_angles, out_type = "train", out_folder = './'):
	batch = 100
	i = 0
	print(out_folder+'PCA_{}_theta'.format(out_type))
	while i < N_angles:
		print("We are at: ",i)
		p.create_dataset_alpha_beta(N_angles = batch, filename="temp.dat", N_grid = 1000, tau_min = 20., q_range= (1.1,10.), smooth_oscillation = False, verbose = False)
		theta_vector, alpha_dataset, beta_dataset, times = load_dataset("temp.dat", shuffle=False, n_params = 6)
		
		red_alpha = PCA_model_alpha.reduce_data(alpha_dataset)
		red_beta = PCA_model_beta.reduce_data(beta_dataset)

		print(theta_vector.shape)

		with open(out_folder+'PCA_{}_theta.dat'.format(out_type),'ab') as f_theta:
			np.savetxt(f_theta,theta_vector)	
		with open(out_folder+'PCA_{}_alpha.dat'.format(out_type),'ab') as f_alpha:
			np.savetxt(f_alpha,red_alpha)
		with open(out_folder+'PCA_{}_beta.dat'.format(out_type),'ab') as f_beta:
			np.savetxt(f_beta,red_beta)
		
		i += batch

		os.remove("temp.dat")
	return
	

def generate_3rd_order(N_obj):
	l = []
	for i in range(N_obj):
		for j in range(i, N_obj):
			for k in range(j, N_obj):
				l.append("{}{}{}".format(i,j,k))

	return l

def generate_4th_order(N_obj):
	l = []
	for i in range(N_obj):
		for j in range(i, N_obj):
			for k in range(j, N_obj):
				for m in range(k, N_obj):
					l.append("{}{}{}{}".format(i,j,k,m))

	return l

#############
#START OF THE CODE HERE

#creating datasets

dataset_file = "PCA_fit_dataset.dat"
PCA_dataset_folder = "alpha_beta_model/"
model_folder = "alpha_beta_model/22"

if True:
	p.create_dataset_alpha_beta(N_angles = 2000, filename="PCA_fit_dataset.dat", N_grid = 1000, tau_min = 20., q_range= (1.1,10.), smooth_oscillation = False, verbose = False)

if True:
	fit_create_PCA_angle_dataset((3,3), dataset_file, PCA_dataset_folder, train_frac = 0.9)	

if True: #adding to PCA dataset
	PCA_alpha = PCA_model(PCA_dataset_folder+"alpha_PCA_model")
	PCA_beta = PCA_model(PCA_dataset_folder+"beta_PCA_model")

	create_PCA_angle_dataset(PCA_alpha, PCA_beta, 1000, out_type = "test", out_folder = PCA_dataset_folder)	
	create_PCA_angle_dataset(PCA_alpha, PCA_beta, 10000, out_type = "train", out_folder = PCA_dataset_folder)

	

quit()

#loading datasets
N_train = 10000
times = np.loadtxt(PCA_dataset_folder+"times")
train_theta = np.loadtxt(PCA_dataset_folder+"PCA_train_theta.dat")[:N_train,:]		#(N,3)
test_theta = np.loadtxt(PCA_dataset_folder+"PCA_test_theta.dat")					#(N',3)
train_alpha = np.loadtxt(PCA_dataset_folder+"PCA_train_alpha.dat")[:N_train,:]	#(N,K)
test_alpha = np.loadtxt(PCA_dataset_folder+"PCA_test_alpha.dat")				#(N',K)
train_beta = np.loadtxt(PCA_dataset_folder+"PCA_train_beta.dat")[:N_train,:]	#(N,K)
test_beta = np.loadtxt(PCA_dataset_folder+"PCA_test_beta.dat")					#(N',K)
PCA_alpha = PCA_model(PCA_dataset_folder+"alpha_PCA_model")
PCA_beta = PCA_model(PCA_dataset_folder+"beta_PCA_model")

val_data = test_theta, np.concatenate([test_alpha, test_beta], axis =1)
train_vals = np.concatenate([train_alpha,train_beta],axis =1)

#import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Neural network
model = Sequential()
model.add(Dense(300, input_dim=6, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(train_vals.shape[1]	, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])


history = model.fit(train_theta, train_vals, validation_data = val_data, epochs=100, batch_size=1000)

prediction = model(test_theta)

rec_beta = PCA_beta.reconstruct_data(prediction[:,3:])
true_beta = PCA_beta.reconstruct_data(val_data[1][:,3:])
plt.plot(times, rec_beta[:10,:].T, c = 'r')
plt.plot(times, true_beta[:10,:].T, c= 'b')
plt.show()




















#############
#Below an attempt of fitting with MoE
#The whole conclusion is as follows:
#	PCA does kind of a good job if we do not care about oscillation (this must be motivated somehow physically)
#	MoE doesn't handle correcly the regression: a NN is expected to perform way better: build it and see
##############
quit()
	#features to use for the basis function expansion
second_order = ["00", "11","22", "33","44","55" "01", "02", "03", "04", "05", "12", "13", "14", "15", "23", "24", "25", "34", "35", "45"]

third_order = generate_3rd_order(6) + second_order
fourth_order = generate_4th_order(6) + third_order
print(fourth_order)

if False:
	print("Saving MoE model to: ", model_folder)
	print("Fitting beta")
	fit_MoE("ph", PCA_dataset_folder, model_folder, experts = 4, comp_to_fit = None, features = third_order, EM_threshold = 1e-2, args = None, N_train = 14000, verbose = False)
	print("Fitting alpha")
	fit_MoE("amp", PCA_dataset_folder, model_folder, experts = 4, comp_to_fit = None, features = third_order, EM_threshold = 1e-2, args = 	None, N_train = 14000, verbose = False)

	#loading test set
theta_vector, alpha_dataset, beta_dataset, times = load_dataset(dataset_file, shuffle=False, n_params = 6) #loading dataset
train_theta, test_theta, train_alpha, test_alpha = make_set_split(theta_vector, alpha_dataset, 0.85, 1.)
train_theta, test_theta, train_beta, test_beta   = make_set_split(theta_vector, beta_dataset, .85, 1.)

g = mode_generator("22", "./alpha_beta_model/22/")

rec_alpha, rec_beta = g.get_raw_mode(test_theta)

print("Mse alpha: ",np.mean(np.square(test_alpha- rec_alpha)))
print("Mse beta: ",np.mean(np.square(test_beta- rec_beta)))

N_plot = 10
plt.figure()
plt.plot(times, rec_beta[:N_plot,:].T, c = 'r')
plt.plot(times, test_beta[:N_plot,:].T, c = 'b')

plt.figure()
plt.plot(times, rec_alpha[:N_plot,:].T, c = 'r')
plt.plot(times, test_alpha[:N_plot,:].T, c = 'b')

plt.show()
	


	
	
	

