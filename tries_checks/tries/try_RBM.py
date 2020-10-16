###################
#	Some tries of fitting GW generation model using RBM
#	RBM seems to be useless!!!!
###################

import sys
sys.path.insert(1, '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/routines')
sys.path.insert(1, '/home/stefano/Documents/Stefano/scuola/uni/miscellanea/RBM/')
from GW_helper import *
import matplotlib.pyplot as plt
from ML_routines import *
from RBM_class import *

	#loading dataset
theta_vector, amp_dataset, ph_dataset, frequencies = load_dataset("GW_dataset.gz")
print("Dataset loaded with "+str(theta_vector.shape[0])+" data")

	#splitting into train and test set
	#to make data easier to deal with
train_frac = .85
ph_scale_factor = 1. #np.std(ph_dataset) #phase must be rescaled back before computing mismatch index beacause F strongly depends on an overall phase... (why so strongly????)

train_theta, test_theta, train_amp, test_amp = make_set_split(theta_vector, amp_dataset, train_frac, 1e-21)
train_theta, test_theta, train_ph, test_ph = make_set_split(theta_vector, ph_dataset, train_frac, ph_scale_factor)

	#DOING PCA
print("#####PCA#####")
K_ph = 30

	#phase
ph_PCA = PCA_model()
ph_PCA.fit_model(train_ph, K_ph, scale_data=False)
red_train_ph = ph_PCA.reduce_data(train_ph)
red_test_ph = ph_PCA.reduce_data(test_ph) #reduced test data PCA
rec_test_ph = ph_PCA.reconstruct_data(red_test_ph) #reconstructed data for phase
error_ph = np.linalg.norm(test_ph - rec_test_ph, ord= 'fro')/(test_ph.shape[0]*np.std(test_ph))
print("Reconstruction error for phase: ",error_ph)

	#preprocessing phases and thetas...
logreg_ph = logreg_model(test_theta.shape[1],red_train_ph.shape[1], False) #for preprocessing only
y_train = logreg_ph.preprocess_data(red_train_ph)[0]
y_test = logreg_ph.preprocess_data(red_test_ph)[0]

for i in range(train_theta.shape[1]):
	max_theta = np.max(train_theta[:,i])
	min_theta = np.min(train_theta[:,i])
	test_theta[:,i] = (test_theta[:,i] - min_theta)/np.abs(max_theta-min_theta) #params set [0,1]
	train_theta[:,i] = (train_theta[:,i] - min_theta)/np.abs(max_theta-min_theta) #params set [0,1]

print(train_theta, y_train)

	#trying RBM
hidden_layers = 200
print(train_theta.shape, train_ph.shape)

train_data = np.concatenate((train_theta, y_train), axis = 1) #merging training data for a training set suitable for RBM
machine = RBM(train_data.shape[1], hidden_layers, v_type="cont")
#history = machine.CD_fit(3000, train_data, K=2, minibatch_size=.5, alpha = 0.0001, persistent = True)
#machine.save_W("./RBM_model.dat")
machine.load_W("./RBM_model.dat")

	#doing predictions...
error = 0 
for i in range(1):
	print(i)
	index_to_sample = np.random.randint(0,test_theta.shape[0])
	v_cond_sam = machine.sample_from_conditional(test_theta[index_to_sample,:], N_samples = 50)
	avg = np.average(v_cond_sam, axis=0)

	rec_fit_ph = ph_PCA.reconstruct_data(avg[train_theta.shape[1]:])
	error = error + np.linalg.norm(rec_fit_ph - test_ph[index_to_sample,:])

	plt.plot(avg[train_theta.shape[1]:], 'o', label = 'reconstructed')
	plt.plot(y_test[index_to_sample,:], 'o', label = 'true')
	print(avg[0],test_theta[index_to_sample,0])
	plt.legend()
error = error / (10*np.std(test_ph))
print("Phase reconstruction error: ", error)
plt.show()






quit()

#computing mismatch
print(np.std(test_ph))
F = compute_mismatch(test_amp, test_ph, rec_fit_amp, rec_fit_ph)
#F = compute_mismatch(test_amp,test_ph, test_amp, rec_fit_ph) #to compute phase mismatch
print("Mismatch: ",F)
print("Mismatch avg: ",np.mean(F))


plt.show()









