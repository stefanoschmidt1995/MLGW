###################
#	Some tries of fitting GW generation model using PCA + MoE
###################

import sys
#sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
sys.path.insert(1, '../routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from EM_MoE import *
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset_s0.dat", shuffle = False) #loading dataset
if np.all(theta_vector[:,1] == 0):
	theta_vector = np.reshape(theta_vector[:,0], (theta_vector.shape[0],1))
PCA_train_ph = np.loadtxt("../datasets/PCA_train_s0.dat")
PCA_test_ph = np.loadtxt("../datasets/PCA_test_s0.dat")

train_frac = .8

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

del amp_dataset, ph_dataset, theta_vector

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+" data with ",PCA_test_ph.shape[1]," features")

		#DOING PCA
print("#####PCA#####")
K_ph = PCA_train_ph.shape[1]
ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/PCA_std_model_s0.dat")

rec_PCA_test_ph = ph_PCA.reconstruct_data(PCA_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_test_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase with PCA: ",error_ph)

F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_PCA_test_ph)
print("Mismatch PCA avg: ",np.mean(F_PCA))

	#preprocessing data
max_ph = np.max(np.abs(PCA_train_ph), axis = 0)
max_ph[np.where(max_ph > 100)] /= 10.
PCA_train_ph = np.divide(PCA_train_ph,max_ph)
PCA_test_ph = np.divide(PCA_test_ph,max_ph)

	#doing MoE
mu_guess = [1.,1.01,1.03,1.05,1.07,1.09,1.10,1.12,1.14,1.17,1.19,1.21,1.24,1.26,1.29,1.31,1.34,1.37,1.4,1.43,1.46,1.5,1.53,1.57,1.6,1.64,1.69,1.73,
1.77,1.82,1.87,1.92,1.97,2.03,2.09,2.16,2.22,2.27,2.36,2.43,2.5,2.59,2.68,2.77,2.85,2.97,3.07,3.18,3.3,3.29,3.43,3.56,3.69,3.84,3.99,4.15, 4.33,4.52,4.7,4.89]
N_experts = len(mu_guess)
mu_guess = np.reshape(mu_guess, (1,N_experts))

MoE_models = []
gat_models = []
PCA_fit_ph = np.zeros(PCA_test_ph.shape)
print("Doing MoE model for each component")
print("Indipendent variables = ", train_theta.shape[1])
for k in range(1):
	print("Fitting component #",str(k))
		#creating model for gating function
	gat_models.append(GDA(train_theta.shape[1], N_experts, hard_clustering = True, same_weights = True))
	gat_models[k].init_centroids(mu_guess)

		#creating model for MoE & doing fitting
	MoE_models.append(MoE_model(train_theta.shape[1], N_experts, bias = True, gating_function=gat_models[k]))
	MoE_models[k].fit(train_theta, PCA_train_ph[:,k], N_iter=1, args=[], threshold = 5e-3)

		#making predictions
	PCA_fit_ph[:,k] = MoE_models[k].predict(test_theta)

	#MoE_models[k].save("./saved_model/"+str(k)+"_MoE_s0.dat","./saved_model/"+str(k)+"_gat_s0.h5") #saving model for future use
		
	noise_est_train = np.divide(PCA_train_ph[:,k] - MoE_models[k].predict(train_theta), PCA_train_ph[:,k])
	noise_est_test = np.divide(PCA_test_ph[:,k] - PCA_fit_ph[:,k], PCA_test_ph[:,k])
	print("Train reconstruction error for comp #"+str(k)+": ", np.mean(noise_est_train), np.std(noise_est_train))
	print("Test reconstruction error for comp #"+str(k)+": ", np.mean(noise_est_test), np.std(noise_est_test))
	print("test square loss (norm): ",np.sum(np.square(PCA_fit_ph[:,k]-PCA_test_ph[:,k]))/(PCA_test_ph.shape[0]*np.std(PCA_test_ph[:,k])))
	print("test square loss (not norm): ",np.sum(np.square(PCA_fit_ph[:,k]-PCA_test_ph[:,k]))/(PCA_test_ph.shape[0]))

	#for i in range(1):
#		plt.figure(k*3+i)
	plt.figure(k, figsize = (15,10))
	comp = k
	i=0
	plt.title("Data component #"+str(comp)+" vs param "+str(i))
	plt.plot(test_theta[:,i], PCA_test_ph[:,comp], 'o',label = 'true', ms = 2)
	plt.plot(test_theta[:,i], PCA_fit_ph[:,comp], 'o',label = 'fitted', ms = 2)
	plt.legend()
	plt.show()
	plt.savefig("../pictures/MoE_pic/comp_"+str(k)+"_s0.jpeg")
	plt.close(k)

#plt.show()
quit()

PCA_test_ph = np.multiply(PCA_test_ph, max_ph)
PCA_fit_ph = np.multiply(PCA_fit_ph, max_ph)

noise_est = np.divide(PCA_test_ph - PCA_fit_ph, PCA_test_ph)
print("Test reconstruction error for reduced coefficients: ", np.mean(noise_est), np.std(noise_est))

rec_fit_ph = ph_PCA.reconstruct_data(PCA_fit_ph)
error_ph = np.linalg.norm(test_ph - rec_fit_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Fit reconstruction error for phase: ", error_ph)

plt.figure(2)
plt.title("Phase with FIT")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true |' + str(np.round(test_theta[i,0],2))+","+ str(np.round(test_theta[i,1],2))+","+ str(np.round(test_theta[i,2],2)))
	plt.plot(frequencies, rec_fit_ph[i,:], label = 'fit')
plt.legend()

F = compute_mismatch(test_amp, test_ph, test_amp, rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

plt.show()






























