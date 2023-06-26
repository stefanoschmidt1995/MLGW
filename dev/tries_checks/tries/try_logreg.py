###################
#	Some tries of fitting GW generation model using PCA+ logistic regression
#	Apparently it works quite well
###################

from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *

#theta_vector, amp_dataset, ph_dataset, frequencies = create_dataset(250, N_grid=450, q_max =10, spin_mag_max = 0.6, f_step=.01, f_high = 1024, f_min = 50, f_max = 300) #for generating dataset from scratch
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("./datasets/GW_dataset_small_f_step.gz") #loading dataset
	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85
ph_scale_factor = 1. #np.std(ph_dataset) #phase must be rescaled back before computing mismatch index beacause F strongly depends on an overall phase... (why so strongly????)

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, ph_scale_factor)

		#DOING PCA
print("#####PCA#####")
K_amp = 5
K_ph = 20
	#amplitude
amp_PCA = PCA_model()
amp_PCA.fit_model(train_amp, K_amp)
red_train_amp = amp_PCA.reduce_data(train_amp)
res_test_amp = amp_PCA.reduce_data(test_amp)
rec_PCA_amp = amp_PCA.reconstruct_data(res_test_amp) #reconstructed data for amplitude
error_amp = np.linalg.norm(test_amp - rec_PCA_amp, ord= 'fro')/(test_amp.shape[0]*np.std(test_amp))
print("Reconstruction error for amplitude: ",error_amp)

plt.figure(0)
plt.title("Amplitude with PCA")
for i in range(2):
	plt.plot(frequencies, test_amp[0,:], label = 'true')
	plt.plot(frequencies, rec_PCA_amp[0,:], label = 'reconstructed')
plt.legend()

	#phase
ph_PCA = PCA_model()
ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
red_train_ph = ph_PCA.reduce_data(train_ph)
red_test_ph = ph_PCA.reduce_data(test_ph)
rec_PCA_ph = ph_PCA.reconstruct_data(red_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_PCA_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase: ",error_ph)

plt.figure(1)
plt.title("Phase with PCA")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true')
	plt.plot(frequencies, rec_PCA_ph[i,:], label = 'reconstructed')
plt.legend()

#scaling back phase (??)
#rec_PCA_ph = rec_PCA_ph*ph_scale_factor
#test_ph = test_ph*ph_scale_factor

F_PCA = compute_mismatch(test_amp, test_ph*ph_scale_factor, rec_PCA_amp, rec_PCA_ph*ph_scale_factor)
print("Mismatch PCA: ",F_PCA)
print("Mismatch PCA avg: ",np.mean(F_PCA))

#plt.show()

		#DOING LOGREG
print("#####LOGREG#####")
	#amplitude
max_amp = np.max(red_train_amp)
min_amp = np.min(red_train_amp)
yy = (red_train_amp - min_amp)/np.abs(max_amp-min_amp) #labels set scaled in [0,1]

logreg_amp = logreg_model(test_theta.shape[1],red_train_amp.shape[1],False)
logreg_amp.fit_gradopt(train_theta, yy, 0.01)

red_fit_amp = (logreg_amp.get_predictions(test_theta) * np.abs(max_amp-min_amp) ) + min_amp #making predictions
#red_fit_amp = logreg_amp.get_predictions(test_theta)
rec_fit_amp = amp_PCA.reconstruct_data(red_fit_amp)
error_amp = np.linalg.norm(test_amp - rec_fit_amp, ord= 'fro')/(test_amp.shape[0]*np.std(test_amp))
print("Fit reconstruction error for amplitude: ",error_amp)

plt.figure(3)
for i in range(2):
	plt.plot(frequencies, test_amp[i,:], label = 'true |' + str(np.round(test_theta[i,0],2))+","+ str(np.round(test_theta[i,1],2))+","+ str(np.round(test_theta[i,2],2)))
	plt.plot(frequencies, rec_fit_amp[i,:], label = 'fit')
plt.title("Amplitude with FIT")
plt.legend()

	#phase
logreg_ph = logreg_model(test_theta.shape[1],red_train_ph.shape[1], False)

red_train_ph = logreg_ph.preprocess_data(red_train_ph)[0]
red_test_ph = logreg_ph.preprocess_data(red_test_ph)[0]

logreg_ph.fit_gradopt(train_theta,red_train_ph, 0.0)

red_fit_ph = logreg_ph.get_predictions(test_theta) #making predictions

plt.figure(8)
plt.title("Fitted reduced phases")
for i in range(1):
	plt.plot(red_test_ph[i,:], 'o', label = 'true')
	plt.plot(red_fit_ph[i,:], 'o', label = 'fit')
plt.legend()

red_fit_ph = logreg_ph.un_preprocess_data(red_fit_ph)
red_test_ph = logreg_ph.un_preprocess_data(red_test_ph)
rec_fit_ph = ph_PCA.reconstruct_data(red_fit_ph) * ph_scale_factor
error_ph = logreg_ph.get_reconstruction_error(test_theta, red_test_ph)
print("Fit reconstruction error for phase: ",error_ph)


plt.figure(20)
plt.title("Data to fit...")
plt.plot(test_theta[:,0], red_fit_ph[:,0], 'o',label = 'fitted')
plt.plot(test_theta[:,0], red_test_ph[:,0], 'o',label = 'true')
plt.legend()


plt.figure(4)
plt.title("Phase with FIT")
for i in range(2):
	plt.plot(frequencies, test_ph[i,:], label = 'true |' + str(np.round(test_theta[i,0],2))+","+ str(np.round(test_theta[i,1],2))+","+ str(np.round(test_theta[i,2],2)))
	plt.plot(frequencies, rec_fit_ph[i,:], label = 'fit')

plt.legend()

#computing mismatch
F = compute_mismatch(test_amp, test_ph, rec_fit_amp, rec_fit_ph)
#F = compute_mismatch(test_amp, test_ph, rec_fit_amp, test_ph) #to compute amp mismatch
print("Mismatch fit: ",F)
print("Mismatch fit avg: ",np.mean(F))

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









