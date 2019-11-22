###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("../datasets/GW_std_dataset_s_const.dat", shuffle=False, N_grid = None) #loading dataset

print("Loaded data with shape: "+ str(ph_dataset.shape))

	#splitting into train and test set
	#to make data easier to deal with
train_frac = .8

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)

		#DOING PCA
print("#####PCA#####")
K_ph = 6 #30 apparently works well for PCA...
noise = 0.0
print("   K = ",K_ph, " | N_grid = ", test_ph.shape[1]," | noise ", str(noise))

	#phase
ph_PCA = PCA_model()
E = ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
print("PCA eigenvalues: ", E)
ph_PCA.save_model("../datasets/PCA_model_s_const.dat")
ph_PCA.load_model("../datasets/PCA_model_s_const.dat")
red_train_ph = ph_PCA.reduce_data(train_ph)
red_PCA_ph = ph_PCA.reduce_data(test_ph)

np.savetxt("../datasets/PCA_train_theta_s_const.dat", train_theta)
np.savetxt("../datasets/PCA_test_theta_s_const.dat", test_theta)
np.savetxt("../datasets/PCA_train_s_const.dat", red_train_ph)
np.savetxt("../datasets/PCA_test_s_const.dat", red_PCA_ph)

	#plotting PC projection vs. q
for k in range(K_ph):
	plt.figure(k, figsize=(15,10))
	plt.title("Component # "+str(k)+"| s = "+str(train_theta[0,1:]))
	plt.plot(train_theta[:,0], red_train_ph[:,k], 'o', ms = 1)
	plt.xlabel("q")
	plt.ylabel("PC value")
#	plt.legend()
	plt.savefig("../pictures/PCA_comp_s_const/comp_"+str(k)+".jpeg")
	plt.close(k)
plt.show()

quit()

	#adding some noise to PCA values
red_PCA_ph_old = red_PCA_ph
red_PCA_ph = np.multiply(red_PCA_ph, np.random.normal(1, noise,red_PCA_ph.shape))
error_ph = np.linalg.norm(red_PCA_ph - red_PCA_ph_old, ord= 'fro')/(test_ph.shape[0]*np.std(red_PCA_ph))
noise_est = np.divide(red_PCA_ph - red_PCA_ph_old, red_PCA_ph_old)
#print("Fit reconstruction error for reduced coefficients: ", error_ph, np.mean(noise_est), np.std(noise_est))

rec_PCA_ph = ph_PCA.reconstruct_data(red_PCA_ph) #reconstructed data for phase

error_ph = np.linalg.norm(test_ph - rec_PCA_ph, ord= 'fro')/(test_ph.shape[0])#*np.std(test_ph))
print("Reconstruction error for phase: ",error_ph) #apparently this is the most accurate estimator for the noise added. Why?? And why not the rec error for reduced coefficients?

plt.figure(2)
plt.title("Phase with PCA")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true')
	plt.plot(frequencies, rec_PCA_ph[i,:], label = 'reconstructed')
	#plt.plot(frequencies, rec_PCA_ph[i,:]-test_ph[i,:], label = str(test_theta[i,0]))
	#plt.xscale('log')
	#plt.yscale('log')
plt.legend()


F_PCA = compute_mismatch(test_amp, test_ph, test_amp, rec_PCA_ph)
#print("Mismatch PCA: ",F_PCA)
print("Mismatch PCA avg: ",np.mean(F_PCA))

plt.figure(6)
plt.title("Phase principal components")
for i in range(10):#ph_PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, ph_PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.show()


plt.show()
quit()

#####################################à
	#OLD TRIES....

##
	#checking PCA behaviour on changing frequency scale
old_frequencies = frequencies
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("./datasets/GW_std_dataset_small_grid.dat", shuffle=False, N_grid =1000)
train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph   = make_set_split(theta_vector, ph_dataset, train_frac, 1.)
#ph_PCA.change_grid(old_frequencies, frequencies)
E = ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
red_PCA_ph = ph_PCA.reduce_data(test_ph)
print(red_PCA_ph[0,:])


	#doing "inverse logreg"
max_theta = np.max(theta_vector[:,0])
max_q = np.max([np.max(theta_vector[:,1]),np.max(theta_vector[:,2])])
train_theta[:,0] = train_theta[:,0] / max_theta
train_theta[:,1] = (train_theta[:,1] - max_q) / (2*max_q)
train_theta[:,2] = (train_theta[:,2] - max_q) / (2*max_q)
test_theta[:,0] = test_theta[:,0] / max_theta
test_theta[:,1] = (test_theta[:,1] - max_q) / (2*max_q)
test_theta[:,2] = (test_theta[:,2] - max_q) / (2*max_q)


logreg_theta = logreg_model(red_train_ph.shape[1], test_theta.shape[1], False)
logreg_theta.fit_gradopt(red_train_ph, train_theta, 0.01)

fit_theta = (logreg_theta.get_predictions(red_PCA_ph))
error_ph = np.linalg.norm(fit_theta - test_theta, ord= 'fro')/(test_ph.shape[0])
print("Fit reconstruction error for phase in \"inverse fit\": ",error_ph)


quit()

	#solving for red_ph given theta
W, bb = logreg_theta.get_weights()
b = logit(test_theta[0,:]) - bb
print(W,bb)
red_test_ph = np.linalg.lstsq(W,b, rcond = 1)

print(red_test_ph)

rec_test_ph = ph_PCA.reconstruct_data(np.reshape(red_test_ph, (1,len(red_test_ph))))
F = compute_mismatch(test_amp[0,:],red_test_ph, test_amp[0,:],rec_PCA_ph[0,:])
print("Fit mismatch: ", F)

plt.figure(6)
plt.plot(frequencies, rec_test_ph, label='fit')
plt.plot(frequencies, rec_PCA_ph[0,:], label='true')
plt.legend()

plt.show()
quit()

	#plotting principal components
plt.figure(5)
plt.title("Amplitude principal components")
for i in range(amp_PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, amp_PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.figure(6)
plt.title("Phase principal components")
for i in range(ph_PCA.get_V_matrix().shape[1]):
	plt.plot(frequencies, ph_PCA.get_V_matrix()[:,i], label=str(i))
plt.legend()

plt.show()

quit()

	#snippet for visualizing WFs in different resolutions
for i in [100000,10000,1000,500,256]:
	plt.figure(i)
	wave = test_ph[0,:]
	indices = np.arange(1, wave.shape[0], wave.shape[0]/i).astype(int)
	plt.title("Phase with N_grid = "+str(i))
	for i in range(1):
		plt.plot(frequencies[indices], wave[indices], label = str(i))
	plt.legend()
plt.show()
quit()

############# PN tries...
M_c = np.divide(np.power(theta_vector[:,0],3./5.), np.power(1.+theta_vector[:,0],1./5.)) #chirp mass
m = 1.+theta_vector[:,0]
nu = np.divide(theta_vector[:,0], np.power(m,2.))
prefactor = 6.6e-11 * 2e30 /(3e8)**3 #G*M_sun/c**3
M = prefactor*m

	#PN coefficients... (p.298 Maggiore)
t_0 = np.divide(np.power(M*np.pi,-5./3.), nu) * (5./(256*np.pi))
t_1 = np.multiply(np.power(np.multiply(M*np.pi,nu),-1.), 743./336 + 11/4. * nu) * (5./(192*np.pi))
t_1_5 = np.divide(np.power(M*np.pi,-2./3.), nu) / 8.
t_2 =  np.multiply(np.divide(np.power(M*np.pi,-1./3.), nu), 3058673./1016064. + 5429/1008 * nu+ 617/144 * np.square(nu))  * (5./(128*np.pi))


plt.figure(0)
plt.title("Phase")
for i in range(5):
	#offset = (3/5. * t_0[i] * np.power(frequencies,-5/3.) + t_1[i] * np.power(frequencies,-1.) - 1.5 *t_1_5[i] * np.power(frequencies,-2/3.) + 3 * t_2[i] * np.power(frequencies,-1./3.) )* 2*np.pi

#	plt.plot(frequencies*np.power(1+theta_vector[i,0],1.)/2.2e3, ph_dataset[i,:]+offset-np.max(ph_dataset[i,:])+100)
#,label = str(theta_vector[i,0])+" "+str(theta_vector[i,1])+" "+str(theta_vector[i,2]))
	#plt.plot(frequencies*np.power(1+theta_vector[i,0],1.)/2.2e3, offset-np.max(ph_dataset[i,:])+100)

	#plt.plot(frequencies*np.power(1+theta_vector[i,0],1.)/2.2e3, process_phases(frequencies, theta_vector[:,0],ph_dataset, False)[i,:])
	plt.plot(frequencies, ph_dataset[i,:], label = str(theta_vector[i,0]))
	#plt.yscale('log')
	#plt.xscale('log')
	plt.legend()

plt.figure(1)
plt.title("Amp")
for i in range(5):
	plt.plot(frequencies*np.power(1+theta_vector[i,0],1.)/2.2e3, np.multiply(amp_dataset[i,:], np.multiply(np.power(frequencies, 7./6.),np.power(M_c[i], -5./6.)))*1e19, label = str(theta_vector[i,0])+" "+str(theta_vector[i,1])+" "+str(theta_vector[i,2]))
	#plt.plot(frequencies*np.power(1+theta_vector[i,0],1.)/2.2e3, process_amplitudes(frequencies, theta_vector[:,0],amp_dataset, False)[i,:])
	#plt.plot(frequencies, process_amplitudes(frequencies, theta_vector[:,0],amp_dataset, False)[i,:])
	plt.legend()
	#plt.yscale('log')
#plt.show()







