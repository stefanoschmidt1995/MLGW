###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

to_fit = "amp"

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_TD_dataset_long/GW_TD_dataset_long.dat", shuffle=False, N_grid = None) #loading dataset

print("Loaded data with shape: "+ str(ph_dataset.shape))

	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85


train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

print(np.max(train_theta[:,1]), np.min(train_theta[:,1]))

if to_fit == "amp":
	train_data = train_amp
	test_data = test_amp
if to_fit == "ph":
	train_data = train_ph
	test_data = test_ph

		#DOING PCA
print("#####PCA of "+to_fit+" #####")
K_ph = 7 #choose here number of PC
noise = 0.0
print("   K = ",K_ph, " | N_grid = ", test_data.shape[1]," | noise ", str(noise))

	#phase
PCA = PCA_model()
E = PCA.fit_model(train_data, K_ph, scale_PC=True)
print("PCA eigenvalues: ", E)
PCA.save_model("../datasets/PCA_model_full_"+to_fit+".dat")
#PCA.load_model("../datasets/PCA_model_s2_const.dat")
red_train_data = PCA.reduce_data(train_data)
red_test_data = PCA.reduce_data(test_data)
rec_test_data = PCA.reconstruct_data(red_test_data)

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
	plt.savefig("../pictures/PCA_comp_full_"+to_fit+"/comp_"+str(k)+".jpeg")
	plt.close(k)
plt.show()


	#computing mismatch
if to_fit == "amp":
	F_PCA = compute_mismatch(test_amp, test_ph, rec_test_data, test_ph)
if to_fit == "ph":
	F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_test_data)
print("Mismatch PCA avg: ",np.mean(F_PCA))

quit()

####checking behaviour with noise

	#adding some noise to PCA values
red_PCA_ph_old = red_PCA_ph
red_PCA_ph = np.multiply(red_PCA_ph, np.random.normal(1, noise,red_PCA_ph.shape))
error_ph = np.linalg.norm(red_PCA_ph - red_PCA_ph_old, ord= 'fro')/(test_data.shape[0]*np.std(red_PCA_ph))
noise_est = np.divide(red_PCA_ph - red_PCA_ph_old, red_PCA_ph_old)
#print("Fit reconstruction error for reduced coefficients: ", error_ph, np.mean(noise_est), np.std(noise_est))

rec_PCA_ph = PCA.reconstruct_data(red_PCA_ph) #reconstructed data for phase

error_ph = np.linalg.norm(test_data - rec_PCA_ph, ord= 'fro')/(test_data.shape[0])#*np.std(test_data))
print("Reconstruction error for phase: ",error_ph) #apparently this is the most accurate estimator for the noise added. Why?? And why not the rec error for reduced coefficients?

plt.figure(2)
plt.title("Phase with PCA")
for i in range(2):
	plt.plot(frequencies, test_data[i,:], label = 'true')
	plt.plot(frequencies, rec_PCA_ph[i,:], label = 'reconstructed')
	#plt.plot(frequencies, rec_PCA_ph[i,:]-test_data[i,:], label = str(test_theta[i,0]))
	#plt.xscale('log')
	#plt.yscale('log')
plt.legend()


F_PCA = compute_mismatch(test_amp, test_data, test_amp, rec_PCA_ph)
#print("Mismatch PCA: ",F_PCA)
print("Mismatch PCA avg: ",np.mean(F_PCA))

plt.figure(6)
plt.title("Phase principal components")
for i in range(10):#PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.show()


plt.show()
quit()








