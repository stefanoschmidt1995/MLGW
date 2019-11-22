###################
#	Some tries of fitting GW generation model using PCA + MoE
###################

#probably a better approach would be that of many 1D fits with GP on top...

import sys
#sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
sys.path.insert(1, '../routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from EM_KM import *
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	#loading datasets...
train_theta = np.loadtxt("../datasets/PCA_train_theta_s2_0.dat")[:,:]
test_theta = np.loadtxt("../datasets/PCA_test_theta_s2_0.dat")
if np.all(train_theta[:,2] == 0):
	train_theta = np.reshape(train_theta[:,0:2], (train_theta.shape[0],2))
	test_theta = np.reshape(test_theta[:,0:2], (test_theta.shape[0],2))
#train_theta = np.log(train_theta)
PCA_train_ph = np.loadtxt("../datasets/PCA_train_s2_0.dat")[:,:]
PCA_test_ph = np.loadtxt("../datasets/PCA_test_s2_0.dat")

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+" data with ",PCA_test_ph.shape[1]," features")

	#preprocessing data
max_ph = np.max(np.abs(PCA_train_ph), axis = 0)
max_ph[np.where(max_ph > 100)] /= 10.
PCA_train_ph = np.divide(PCA_train_ph,max_ph)
PCA_test_ph = np.divide(PCA_test_ph,max_ph)
PCA_fit_ph = np.zeros(PCA_test_ph.shape)


	#plotting with curves
N_step = 30
q_step = 1.6/float(N_step)
q_start = -0.8
plt.figure()
for k in range(N_step):
	q_min = k*q_step + q_start
	q_max = (k+1)*q_step + q_start
	indices = np.where(np.logical_and(train_theta[: ,1]>q_min, train_theta[: ,1]<q_max))[0]
	plt.plot(train_theta[indices,0], PCA_train_ph[indices,0] , 'o', ms =2,label = str(q_min))
q_step = 4./float(N_step)
q_start = 1.
plt.legend()
plt.figure()
for k in range(N_step):
	q_min = k*q_step + q_start
	q_max = (k+1)*q_step + q_start
	indices = np.where(np.logical_and(train_theta[: ,0]>q_min, train_theta[: ,0]<q_max))[0]
	plt.plot(train_theta[indices,1], PCA_train_ph[indices,0] , 'o', ms =2,label = str(q_min))

plt.show()
#train_theta = train_theta[:,1].reshape((train_theta.shape[0],1))

##############################
#	Model with K means + linear fit...
# This is a good idea but doesn't work well in practise: too difficoult to find good clustering divisions...
N_experts = 300

for comp in range(1):
	print(train_theta.shape)
	model = K_means_linfit(train_theta.shape[1], sigma = 1e-3)
	D = 1
	#K = N_experts
	model.fit(train_theta, PCA_train_ph[:,comp], K_0 = N_experts ,N_iter=5)
	mu=model.get_params()[2]
	#print(mu)

		#uncomment for test	
	#train_theta = test_theta
	#PCA_train_ph = PCA_test_ph

	train_labels = model.predict(train_theta, PCA_train_ph[:,comp],  hard_clustering = True)
	y = model.predict_y(train_theta, get_labels = False)

		#plotting predictions
	plt.figure(0)
	plt.title("q vs PC")
	plt.plot(train_theta[:,0], y , 'o', ms =2,label = "pred")
	plt.plot(train_theta[:,0],  PCA_train_ph[:,comp] , 'o', ms =2,label = "true")
	plt.legend()
	plt.figure(1)
	plt.title("s1 vs PC")
	#plt.plot(train_theta[:,1], y , 'o', ms =2,label = "pred")
	#plt.plot(train_theta[:,1],  PCA_train_ph[:,comp] , 'o', ms =2,label = "true")
	plt.legend()

		#3D plot
	#from mpl_toolkits import mplot3d
	#fig = plt.figure(figsize=(15,10))
	#ax = plt.axes(projection='3d')
	#ax.scatter(train_theta[:,0],train_theta[:,1], PCA_train_ph[:,0],'o', label = "true")
	#ax.scatter(train_theta[:,0],train_theta[:,1], y,'o', label = "pred")
	#ax.view_init(60, 35)
	#plt.show()



	print("Reconstruction error: ",np.mean(np.square(y-PCA_train_ph[:,comp])))

	#plt.figure(comp, figsize = (15,10))
	plt.title("Data component #"+str(comp)+" vs q ")
	for k_cl in range(model.get_params()[1]):
		train_theta_k = train_theta[np.where(train_labels == k_cl)[0]]
		PCA_train_ph_k = PCA_train_ph[np.where(train_labels == k_cl)[0]]
		#print(k, type(train_theta_k))
		if len(train_theta_k) !=0:
			plt.plot(train_theta_k[:,1], PCA_train_ph_k[:,comp], 'o', label = str(k_cl), ms = 2)
			#y = model.predict_y(train_theta_k[:,0])
			#plt.plot(train_theta_k[:,0],y , 'o', c = 'b', ms = 2, label = "pred")
			plt.plot(mu[2,k_cl],mu[0,k_cl],'o', c = 'b', ms = 3)
	#plt.legend()
	plt.xlabel("q")
	plt.ylabel("PC projection")
	plt.show()


quit()

##############################
#	Model with K means

	#doing MoE
mu_guess = [1.,1.01, 1.03,1.05,1.07,1.09,1.10,1.12, 1.14,1.17,1.19,1.21,1.24, 1.26, 1.29,1.31, 1.34,1.37, 1.4, 1.43, 1.46,1.5,1.53, 1.57,1.6,1.64, 1.69, 1.73, 1.77, 1.82, 1.87,1.92,1.97,2.03, 2.09,2.16,2.22, 2.27,2.36,2.43,2.5,2.59,2.68,2.77,2.85, 2.97, 3.07, 3.18, 3.3, 3.29, 3.43, 3.56,3.69,3.84,3.99, 4.15, 4.33, 4.52,4.7,4.89]
N_experts = len(mu_guess)+100
#mu_guess = np.reshape(mu_guess, (1,N_experts))
#mu_guess = np.concatenate((mu_guess, np.zeros((1,N_experts))), axis =0)

	#fitting the model with 1st component
train_data_0 = np.concatenate((train_theta, np.reshape(PCA_train_ph[:,0], (PCA_train_ph.shape[0],1))), axis = 1)
test_data_0 = np.concatenate((test_theta, np.reshape(PCA_test_ph[:,0], (PCA_test_ph.shape[0],1))), axis = 1)

		#creating model for clusters & doing fitting
model = K_means_model(train_data_0.shape[1], N_experts, sigma = [.1, 1e-3], fit_sigma = [0])
model.fit(train_data_0, N_iter=6)

PCA_fit_ph = np.zeros(PCA_test_ph.shape)

print("Doing MoE model for each component")
print("Indipendent variables = ", train_theta.shape[1])
for k in range(10):
	print("Fitting component #",str(k))
		#creating data
	train_data = np.concatenate((train_theta, np.reshape(PCA_train_ph[:,k], (PCA_train_ph.shape[0],1))), axis = 1)
	test_data = np.concatenate((test_theta, np.reshape(PCA_test_ph[:,k], (PCA_test_ph.shape[0],1))), axis = 1)

	lin_fit_model = predictor_lin_fit_cluster(train_theta.shape[1], model.get_params()[1], model)
	lin_fit_model.fit(train_theta, PCA_train_ph[:,k], train_data_0, regularizer =0., loss = "L1")
	PCA_fit_ph[:,k] = lin_fit_model.predict(test_theta)

	if True:
		plt.plot(test_theta,PCA_fit_ph[:,k], 'o', label = "fitted")
		plt.plot(test_theta,PCA_test_ph[:,k], 'o', label = "true")
		plt.legend()
		plt.show()
		print("Test error component "+str(k)+": ", np.sum(np.square(PCA_fit_ph[:,k]-PCA_test_ph[:,k]))/PCA_fit_ph.shape[0])

		#making predictions
	#train_data = test_data #debug... to check if at test time nothing changes
	#train_data_0 = test_data_0
	#train_data = np.random.uniform([1.,-10.],[5.,4.],size =(100000,2))
	train_labels = model.predict(train_data_0, hard_clustering = True, dim_list = [0])
	#print(model.accuracy(train_data_0,train_labels))
	#train_labels = model.predict(train_theta, hard_clustering = True)


	#MoE_models[k].save("./saved_model/"+str(k)+"_MoE_s0.dat","./saved_model/"+str(k)+"_gat_s0.h5") #saving model for future use

	w = model.get_params()[2]
	sigma = model.get_params()[3]

	plt.figure(k, figsize = (15,10))
	plt.title("Data component #"+str(k)+" vs q ")
	for k_cl in range(N_experts):
		train_data_k = train_data[np.where(train_labels == k_cl)[0]]
		if len(train_data_k !=0):
			pass
			plt.plot(train_data_k[:,0], train_data_k[:,1], 'o', label = str(k_cl), ms = 2)
			#plt.errorbar(w[0,k_cl],w[1,k_cl], xerr=sigma[0,k_cl])#, ps = 15)
			#print(w[0,k_cl],w[1,k_cl],sigma[0,k_cl])
	plt.legend()
	plt.show()
	plt.savefig("../pictures/cluster_pic/comp_"+str(k)+"_s0.jpeg")
	plt.close(k)

#plt.show()
quit()

	#processing back coefficients
train_frac = .8
theta_vector_bis, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset_s0.dat", shuffle = False) #loading
train_theta_bis, test_theta_bis, train_amp, test_amp = make_set_split(theta_vector_bis, amp_dataset, train_frac, 1e-21)
train_theta_bis, test_theta_bis, train_ph, test_ph   = make_set_split(theta_vector_bis, ph_dataset, train_frac, 1.)

ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/PCA_std_model_s0.dat")

PCA_test_ph = np.multiply(PCA_test_ph, max_ph)
PCA_fit_ph = np.multiply(PCA_fit_ph, max_ph)

rec_fit_ph = ph_PCA.reconstruct_data(PCA_fit_ph)
#test_ph = ph_PCA.reconstruct_data(PCA_test_ph)

F = compute_mismatch(test_amp, test_ph, test_amp, rec_fit_ph)
print("Mismatch fit avg: ",np.mean(F))

quit()


















