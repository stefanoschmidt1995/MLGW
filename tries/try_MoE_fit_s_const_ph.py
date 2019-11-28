import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
import keras

    #loading PCA datasets
train_theta = np.loadtxt("../datasets/PCA_train_theta_s_const.dat")
test_theta = np.loadtxt("../datasets/PCA_test_theta_s_const.dat")
PCA_train_ph = np.loadtxt("../datasets/PCA_train_s_const.dat")
PCA_test_ph = np.loadtxt("../datasets/PCA_test_s_const.dat")

    #removing spins from training data (spins must be the same for all train and test example)
s = train_theta[0,1:]
train_theta = np.reshape(train_theta[:,0], (train_theta.shape[0],1))
test_theta = np.reshape(test_theta[:,0], (test_theta.shape[0],1))

	#adding some basis function
train_theta = np.concatenate((train_theta, np.square(train_theta), np.power(train_theta,4)), axis = 1)
test_theta = np.concatenate((test_theta, np.square(test_theta), np.power(test_theta,4)), axis = 1)

print(train_theta.shape)

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+
      " data with ",PCA_train_ph.shape[1]," PCA components")
print("Spins fixed at s= ", s)

   #preprocessing data
max_ph = np.max(np.abs(PCA_train_ph), axis = 0)
PCA_train_ph = np.divide(PCA_train_ph,max_ph)
PCA_test_ph = np.divide(PCA_test_ph,max_ph)


   #setting up an EM model for each component
MoE_models = []
K = 10 #number of experts
D = train_theta.shape[1] #number of independent variables

for k in range(PCA_train_ph.shape[1]):
		#useless variables for sake of clariness
	y_train = PCA_train_ph[:,k]
	y_test = PCA_test_ph[:,k]

		#building keras gating model (useless so far...)
	gat_model = keras.Sequential()
	gat_model.add(keras.layers.Dense(K, activation='softmax'))
	gat_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	MoE_models.append(MoE_model(D,K))
	args = ["adam", 1e-4, False, 150, 1e-2,] #for softmax
	#args = [None,5,0]
	MoE_models[-1].fit(train_theta, y_train, N_iter = None, args = args, verbose = True)

		#doing some test
	y_pred = MoE_models[-1].predict(test_theta)
	y_exp = MoE_models[-1].experts_predictions(test_theta)
	y_gat = MoE_models[-1].get_gating_probs(test_theta)
	print("Test square loss for comp "+str(k)+": ",np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]))

	for i in range(1):#(D):
		plt.figure(i*K+k, figsize=(20,10))
		plt.title("Component #"+str(k)+" vs q | s = "+str(s))
		plt.plot(test_theta[:,i], y_test, 'o', ms = 3,label = 'true')
		plt.plot(test_theta[:,i], y_pred, 'o', ms = 3, label = 'pred')
		#plt.plot(test_theta[:,i], y_exp, 'o', ms = 1,label = 'exp_pred')
		plt.plot(test_theta[:,i], y_gat, 'o', ms = 1)
		plt.legend()
		if i ==0:
			#pass
			plt.savefig("../pictures/PCA_comp_s_const/fit_"+str(k)+".jpeg")

	#plt.show()


############Comparing mismatch for test waves
N_waves = 100

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset(N_waves, N_grid = 2048, filename = None,
                q_range = (1.,5.), s1_range = s[0], s2_range = s[1],
				log_space = True,
                f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRPhenomPv2")

	#preprocessing theta
theta_vector_test = np.reshape(theta_vector_test[:,0], (theta_vector_test.shape[0],1))
theta_vector_test = np.concatenate((theta_vector_test, np.square(theta_vector_test), np.power(theta_vector_test,4)), axis = 1)

ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/PCA_model_s_const.dat")

red_ph_dataset_test = ph_PCA.reduce_data(ph_dataset_test)
#* np.random.normal(1,5e-3,size=(ph_dataset_test.shape[0], ph_PCA.get_PCA_params()[0].shape[1]))
red_ph_dataset_test[:,-1] = 0
F_PCA = compute_mismatch(amp_dataset_test, ph_PCA.reconstruct_data(red_ph_dataset_test),
						 amp_dataset_test, ph_dataset_test)
print("Avg PCA mismatch: ", np.mean(F_PCA))

rec_PCA_dataset = np.zeros((N_waves, PCA_train_ph.shape[1]))
for k in range(len(MoE_models)):
	rec_PCA_dataset[:,k] = MoE_models[k].predict(theta_vector_test)

rec_PCA_dataset = np.multiply(rec_PCA_dataset, max_ph)
rec_ph_dataset = ph_PCA.reconstruct_data(rec_PCA_dataset)

F = compute_mismatch(amp_dataset_test, rec_ph_dataset, amp_dataset_test, ph_dataset_test)
print("Avg fit mismatch: ", np.mean(F))

plt.figure(100)
plt.plot(frequencies_test, rec_ph_dataset[0,:], label = "Rec")
plt.plot(frequencies_test, ph_dataset_test[0,:], label = "True")
plt.legend()
plt.show()



























