import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
#import keras

	#adding extra features for basis function regression
new_features = ["00", "11","22", "01", "02", "12", "0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

folder = "GW_std_dataset/"
PCA_train_amp = np.loadtxt("../datasets/"+folder+"PCA_train_full_amp.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_full_ph.dat")

   #setting up an EM model for each component
MoE_models_amp = 	[]
MoE_models_ph = 	[]

amp_folder = "./saved_models_full_amp/"
ph_folder = "./saved_models_full_ph/"
amp_files = os.listdir(amp_folder)
ph_files = os.listdir(ph_folder)

D = 3+len(new_features) #number of independent variables

K_PCA_amp = 10
K_PCA_ph = 11

	#loading models
for k in range(np.maximum(K_PCA_amp,K_PCA_ph)):

	if "amp_exp_"+str(k) in amp_files:
		MoE_models_amp.append(MoE_model(D,1))
		MoE_models_amp[-1].load(amp_folder+"amp_exp_"+str(k),amp_folder+"amp_gat_"+str(k))
		print("Loaded amplitude model for comp: ", k)

	if "ph_exp_"+str(k) in ph_files:
		MoE_models_ph.append(MoE_model(D,1))
		MoE_models_ph[-1].load(ph_folder+"ph_exp_"+str(k),ph_folder+"ph_gat_"+str(k))
		print("Loaded phase model for comp: ", k)


############Comparing mismatch for test waves
N_waves = 200

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset(N_waves, N_grid = 2048, filename = None,
                q_range = (1.,5.), s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
				log_space = True,
                f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRphenomPv2")
amp_dataset_test = 1e21*amp_dataset_test


	#preprocessing theta
theta_vector_test = add_extra_features(theta_vector_test, new_features)

amp_PCA = PCA_model()
amp_PCA.load_model("../datasets/"+folder+"PCA_model_full_amp.dat")
ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/"+folder+"PCA_model_full_ph.dat")

red_amp_dataset_test = amp_PCA.reduce_data(amp_dataset_test)
if K_PCA_amp < PCA_train_amp.shape[1]:
	red_amp_dataset_test[:,K_PCA_amp:] = 0

red_ph_dataset_test = ph_PCA.reduce_data(ph_dataset_test)
if K_PCA_ph < PCA_train_ph.shape[1]:
	red_ph_dataset_test[:,K_PCA_ph:] = 0

F_PCA = compute_mismatch(amp_dataset_test, ph_dataset_test,
						 amp_PCA.reconstruct_data(red_amp_dataset_test), ph_PCA.reconstruct_data(red_ph_dataset_test))
print("Avg PCA mismatch: ", np.mean(F_PCA))

	###Now it's time to make predictions and test results
rec_PCA_dataset_amp = np.zeros((N_waves, PCA_train_amp.shape[1]))
rec_PCA_dataset_ph = np.zeros((N_waves, PCA_train_ph.shape[1]))

	#making predictions for amplitude
for k in range(len(MoE_models_amp)):
	rec_PCA_dataset_amp[:,k] = MoE_models_amp[k].predict(theta_vector_test)

	#making predictions for phase
for k in range(len(MoE_models_ph)):
	rec_PCA_dataset_ph[:,k] = MoE_models_ph[k].predict(theta_vector_test)

rec_amp_dataset = amp_PCA.reconstruct_data(rec_PCA_dataset_amp)
rec_ph_dataset = ph_PCA.reconstruct_data(rec_PCA_dataset_ph)

F = compute_mismatch(amp_dataset_test, ph_dataset_test, amp_dataset_test, rec_ph_dataset)
print("Avg phase mismatch: ", np.mean(F))
F = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, rec_ph_dataset)
print("Avg fit mismatch: ", np.mean(F), np.max(F),np.min(F))


	#looking at bad points...
bad_points = np.where(F>1e-3)[0]
print("P(F>1e-3): ", len(bad_points)/float(N_waves))
print("bad points q: ", np.concatenate((theta_vector_test[bad_points,:3], F[bad_points, np.newaxis]), axis =1))

print("low q mismatchs: ", F[np.where(theta_vector_test[:,0]<1.5)[0]])


true_h = np.multiply(amp_dataset_test,np.exp(1j*ph_dataset_test))
rec_h = np.multiply(rec_amp_dataset,np.exp(1j*rec_ph_dataset))

N_plots = 10
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
for i in range(N_plots):
	plt.figure(i, figsize=(15,10))
	plt.title("(q,s1,s2) = "+str(theta_vector_test[indices[i],0:3]))
	plt.plot(frequencies_test, rec_h[indices[i]].real, label = "Rec")
	plt.plot(frequencies_test, true_h[indices[i]].real, label = "True")
	plt.xscale("log")
	plt.legend()
	plt.savefig("../pictures/rec_WFs/WF_"+str(i)+".jpeg")

plt.show()



























