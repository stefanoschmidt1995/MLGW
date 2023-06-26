###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

to_fit = "h"

#theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_TD_dataset_long/GW_TD_dataset_long.dat", shuffle=False, N_grid = None) #loading dataset
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/datasets/GW_TD_dataset_mtotconst/GW_TD_dataset_mtotconst.dat",
shuffle=False, N_grid = None) #loading dataset
print("Loaded data with shape: "+ str(ph_dataset.shape))

	#aligning to merger
cut_off = np.where(frequencies ==0)[0][0]
amp_dataset = amp_dataset
ph_dataset= (ph_dataset.T - ph_dataset[:, cut_off]).T
frequencies=frequencies


	#splitting into train and test set
	#to make data easier to deal with
train_frac = .98


train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

print(np.max(train_theta[:,1]), np.min(train_theta[:,1]))
print(np.max(train_theta[:,2]), np.min(train_theta[:,2]))

if to_fit == "amp":
	indices = range(train_ph.shape[1])
	train_data = train_amp
	test_data = test_amp
if to_fit == "ph":
#	indices = np.where(frequencies>=0)[0]
	#indices = range(train_ph.shape[1])
	train_data = train_ph
	test_data = test_ph
if to_fit == "h":
	train_data = (train_amp*np.exp(1j*train_ph)).real
	test_data = (test_amp*np.exp(1j*test_ph)).real

		#DOING PCA
print("##### PCA of "+to_fit+" #####")
K_ph = 241 #choose here number of PC
print("   K = ",K_ph, " | N_grid = ", test_data.shape[1])

	#phase
PCA = PCA_model()
E = PCA.fit_model(train_data, K_ph, scale_PC=True)
print("PCA eigenvalues: ", E)
PCA.save_model("../datasets/PCA_model_full_"+to_fit+".dat")
PCA.load_model("../datasets/PCA_model_full_"+to_fit+".dat")
red_train_data = PCA.reduce_data(train_data)
red_test_data = PCA.reduce_data(test_data)
rec_test_data = PCA.reconstruct_data(red_test_data)

#print(PCA.get_PCA_params())
#print(PCA.get_PCA_params()[3])

np.savetxt("../datasets/PCA_train_theta_full.dat", train_theta)
np.savetxt("../datasets/PCA_test_theta_full.dat", test_theta)
np.savetxt("../datasets/PCA_train_full_"+to_fit+".dat", red_train_data)
np.savetxt("../datasets/PCA_test_full_"+to_fit+".dat", red_test_data)
np.savetxt("../datasets/times", frequencies)

	#plotting PC projection vs. q
for k in range(K_ph):
	plt.figure(k, figsize=(15,10))
	plt.title("Component of "+to_fit+" # "+str(k)+"| full dataset")#s = "+str(train_theta[0,1:]))
	plt.plot(train_theta[:,0], red_train_data[:,k], 'o', ms = 1)
	plt.xlabel("q")
	plt.ylabel("PC value")
#	plt.legend()
	#plt.savefig("../pictures/PCA_comp_full_"+to_fit+"/comp_"+str(k)+".jpeg")
	plt.close(k)

	#computing mismatch
if to_fit == "amp":
	F_PCA = compute_mismatch(test_data, test_ph, rec_test_data, test_ph)
if to_fit == "ph":
	F_PCA = compute_mismatch(test_amp, test_data, test_amp, rec_test_data)
if to_fit == "h":
	F_PCA = compute_mismatch(np.abs(test_data), np.unwrap(np.angle(test_data)), np.abs(rec_test_data), np.unwrap(np.angle(rec_test_data)) )
print("Mismatch PCA avg: ",np.mean(F_PCA))


quit()

####checking behaviour with noise

plt.figure(200)
plt.title("Features value")
for i in range(4):
	plt.plot(frequencies[indices], test_data[i,:], label = 'true')
	#plt.plot(frequencies, rec_PCA_ph[i,:], label = 'reconstructed')
	#plt.plot(frequencies, rec_PCA_ph[i,:]-test_data[i,:], label = str(test_theta[i,0]))
	#plt.xscale('log')
	#plt.yscale('log')
plt.legend()

plt.figure(600)
plt.title("Phase principal components")
for i in range(7):#PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies[indices], PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

#plt.show()

plt.figure(100)
plt.plot(train_theta[:,0], train_data[:,0],'o', ms = 1)
plt.plot(train_theta[:,0], train_data[:,235],'o', ms = 1)
plt.plot(train_theta[:,0], train_data[:,1535],'o', ms = 1)
plt.plot(train_theta[:,0], train_data[:,1800],'o', ms = 1)
plt.plot(train_theta[:,0], train_data[:,1999],'o', ms = 1)
#plt.show()

plt.show()
quit()








