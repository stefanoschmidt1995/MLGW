import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
#import keras

	#adding extra features for basis function regression
new_features_amp = ["00", "11","22", "01", "02", "12", "0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]
new_features_ph = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222"
,"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]

folder = "GW_TD_dataset/"
PCA_train_amp = np.loadtxt("../datasets/"+folder+"PCA_train_full_amp.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_full_ph.dat")

   #setting up an EM model for each component
MoE_models_amp = 	[]
MoE_models_ph = 	[]

amp_folder = "./saved_models_full_amp_TD/"
ph_folder = "./saved_models_full_ph_TD/"
amp_files = os.listdir(amp_folder)
ph_files = os.listdir(ph_folder)

D_amp = 3+len(new_features_amp) #number of independent variables
D_ph = 3+len(new_features_ph) #number of independent variables

K_PCA_amp = 7
K_PCA_ph = 15

	#loading models
for k in range(np.maximum(K_PCA_amp,K_PCA_ph)):

	if "amp_exp_"+str(k) in amp_files:
		MoE_models_amp.append(MoE_model(D_amp,1))
		MoE_models_amp[-1].load(amp_folder+"amp_exp_"+str(k),amp_folder+"amp_gat_"+str(k))
		print("Loaded amplitude model for comp: ", k)

	if "ph_exp_"+str(k) in ph_files:
		MoE_models_ph.append(MoE_model(D_ph,1))
		MoE_models_ph[-1].load(ph_folder+"ph_exp_"+str(k),ph_folder+"ph_gat_"+str(k))
		print("Loaded phase model for comp: ", k)


############Comparing mismatch for test waves
N_waves = 30
print("Generating "+str(N_waves)+" waves")

amp_PCA = PCA_model()
amp_PCA.load_model("../datasets/"+folder+"PCA_model_full_amp.dat")
ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/"+folder+"PCA_model_full_ph.dat")

theta_vector_test, amp_dataset_test, ph_dataset_test, test_times = create_dataset_TD(N_waves, N_grid = ph_PCA.get_V_matrix().shape[0], filename = None,
                t_coal = .5, q_range = (1.,7.), m2_range = 10., s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
                t_step = 5e-5, lal_approximant = "SEOBNRv2_opt")
amp_dataset_test = 1e21*amp_dataset_test


	#preprocessing theta
theta_vector_test_amp = add_extra_features(theta_vector_test, new_features_amp)
theta_vector_test_ph = add_extra_features(theta_vector_test, new_features_ph)

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
print(len(MoE_models_amp))

for k in range(len(MoE_models_amp)):
	rec_PCA_dataset_amp[:,k] = MoE_models_amp[k].predict(theta_vector_test_amp)

	#making predictions for phase
for k in range(len(MoE_models_ph)):
	rec_PCA_dataset_ph[:,k] = MoE_models_ph[k].predict(theta_vector_test_ph)

rec_amp_dataset = amp_PCA.reconstruct_data(rec_PCA_dataset_amp)
rec_ph_dataset = ph_PCA.reconstruct_data(rec_PCA_dataset_ph)

F = compute_mismatch(amp_dataset_test, ph_dataset_test, amp_dataset_test, rec_ph_dataset)
print("Avg phase mismatch: ", np.mean(F))
F = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, rec_ph_dataset)
print("Avg fit mismatch: ", np.mean(F), np.max(F),np.min(F))


	#looking at bad points...
bad_points = np.where(F>1e-2)[0]
print("P(F>1e-3): ", len(bad_points)/float(N_waves))
#print("bad points q: ", np.concatenate((theta_vector_test_amp[bad_points,:3], F[bad_points, np.newaxis]), axis =1))

#print("low q mismatchs: ", F[np.where(theta_vector_test_amp[:,0]<1.5)[0]])


true_h = np.multiply(amp_dataset_test,np.exp(1j*ph_dataset_test))
rec_h = np.multiply(rec_amp_dataset,np.exp(1j*rec_ph_dataset))

N_plots = 2
indices = np.random.choice(range(N_plots), size=N_plots ,replace = False)
indices = bad_points[:2]
for i in range(N_plots):
	plt.figure(i, figsize=(15,10))
	plt.title("(q,s1,s2) = "+str(theta_vector_test[indices[i],0:3])+" | F = "+str(F[indices[i]]) )
	plt.plot(test_times, rec_h[indices[i]].real, label = "Rec")
	plt.plot(test_times, true_h[indices[i]].real, label = "True")
	#plt.xscale("log")
	plt.legend()
	plt.savefig("../pictures/rec_WFs/WF_"+str(i)+".jpeg")

		#plotting phases
plt.figure(N_plots+1)
for i in range(1):
	plt.plot(test_times, ph_dataset_test[i,:], label = "Rec")
	plt.plot(test_times, rec_ph_dataset[i,:], label = "True")
	plt.legend()

plt.show()



























