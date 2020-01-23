import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
#import keras

folder = "GW_TD_dataset_mtotconst/"

    #loading PCA datasets
N_train = -1
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta_full.dat")[:N_train,:]
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta_full.dat")
PCA_train_amp = np.loadtxt("../datasets/"+folder+"PCA_train_full_amp.dat")[:N_train,:]
PCA_test_amp = np.loadtxt("../datasets/"+folder+"PCA_test_full_amp.dat")
K_PCA_to_fit = 4

	#adding extra features for basis function regression
new_features = ["00", "11","22", "01", "02", "12", "0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]
outfile = open("./saved_models_full_amp_TD/amp_feat", "w+")
outfile.write("\n".join(new_features))
outfile.close()

train_theta = add_extra_features(train_theta, new_features)
test_theta = add_extra_features(test_theta, new_features)

print(train_theta.shape, test_theta.shape)

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+
      " data with ", PCA_train_amp.shape[1]," PCA components")
print("Spins are allowed to vary within domain [-0.8,0.8]x[-0.8,0.8]")

   #setting up an EM model for each component
MoE_models = 	[]
load_list = 	[]#[0   ,1   ,2   ,3   ,4   ,5   ,6   ,7   ,8   ,9   ]#,10  ,11  ,12  ,13  ,14  ]  #indices of models to be loaded from file

K = [5 for i in range(K_PCA_to_fit)] 
#K = 			[15  ,15  ,30  ,20  ,15  ,20  ,25  ,20  ,15  ,15  ,15  ,25  ,25  ,25  ,25  ] #number of experts for each model
#epochs_list  = 	[150 ,200 ,200 ,300 ,400 ,400 ,400 ,300 ,300 ,300 ,400 ,400 ,400 ,400 ,400 ] #number of epochs for gating function fit
#step_list =		[1e-2,5e-3,5e-3,5e-3,2e-3,2e-3,1e-3,2e-3,2e-3,2e-3,1e-3,1e-3,1e-3,1e-3,1e-3] #number of steps for gating function fit

D = train_theta.shape[1] #number of independent variables

for k in range(0,K_PCA_to_fit):
	print("Fitting comp ", k)
		#useless variables for sake of clariness
	y_train = PCA_train_amp[:,k]
	y_test = PCA_test_amp[:,k]

	MoE_models.append(MoE_model(D,K[k]))
			#opt	val_set reg verbose threshold N_it	step_size
	args = ["adam", None,   0., False,  1e-4,     None, 2e-3]#, epochs_list[k], step_list[k]] #for softmax
	#args = [None,5,0]

	if k in load_list:
		MoE_models[-1].load("./saved_models_full_amp_TD/amp_exp_"+str(k),"./saved_models_full_amp_TD/amp_gat_"+str(k))
		print("Loaded model for comp: ", k)
	else:	
		MoE_models[-1].fit(train_theta, y_train, threshold = 1e-2, args = args, verbose = True, val_set = (test_theta, y_test))
		MoE_models[-1].save("./saved_models_full_amp_TD/amp_exp_"+str(k),"./saved_models_full_amp_TD/amp_gat_"+str(k))

		#doing some test
	y_pred = MoE_models[-1].predict(test_theta)
	y_exp = MoE_models[-1].experts_predictions(test_theta)
	y_gat = MoE_models[-1].get_gating_probs(test_theta)
	print("Test square loss for comp "+str(k)+": ",np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]))

	for i in range(3):#(D):
		plt.figure(i*K[k]+k, figsize=(20,10))
		plt.title("Component #"+str(k)+" vs q/s1/s2 | index "+str(i))
		plt.plot(test_theta[:,i], y_test, 'o', ms = 3,label = 'true')
		plt.plot(test_theta[:,i], y_pred, 'o', ms = 3, label = 'pred')
		#plt.plot(test_theta[:,i], y_gat, 'o', ms = 1) 						#plotting gating predictions
		#plt.plot(test_theta[:,i], y_exp, 'o', ms = 1,label = 'exp_pred')	#plotting experts predictions
		plt.legend()
		if i ==0 or i==1 or i ==2:
			#pass
			plt.savefig("../pictures/PCA_comp_full_amp/fit_"+str(k)+"_vs"+str(i)+".jpeg")
		plt.close(i*K[k]+k)
	#plt.show()


############Comparing mismatch for test waves
N_waves = 50

amp_PCA = PCA_model()
amp_PCA.load_model("../datasets/"+folder+"PCA_model_full_amp.dat")

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset_TD(N_waves, N_grid = amp_PCA.get_V_matrix().shape[0], filename = None,
                t_coal = .4, q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
                t_step = 1e-5, lal_approximant = "SEOBNRv2_opt")
amp_dataset_test = 1e21*amp_dataset_test


	#preprocessing theta
theta_vector_test = add_extra_features(theta_vector_test, new_features)

red_amp_dataset_test = amp_PCA.reduce_data(amp_dataset_test)
if K_PCA_to_fit < PCA_train_amp.shape[1]:
	red_amp_dataset_test[:,K_PCA_to_fit:] = 0
#* np.random.normal(1,5e-3,size=(amp_dataset_test.shape[0], amp_PCA.get_PCA_params()[0].shape[1]))
F_PCA = compute_mismatch(amp_dataset_test, ph_dataset_test,
						 amp_PCA.reconstruct_data(red_amp_dataset_test), ph_dataset_test)
print("Avg PCA mismatch: ", np.mean(F_PCA))

rec_PCA_dataset = np.zeros((N_waves, PCA_train_amp.shape[1]))
for k in range(len(MoE_models)):
	rec_PCA_dataset[:,k] = MoE_models[k].predict(theta_vector_test)

rec_amp_dataset = amp_PCA.reconstruct_data(rec_PCA_dataset)

F = compute_mismatch(amp_dataset_test, ph_dataset_test, rec_amp_dataset, ph_dataset_test)
print("Avg fit mismatch: ", np.mean(F))

plt.figure(100)
plt.plot(frequencies_test, rec_amp_dataset[0,:], label = "Rec")
plt.plot(frequencies_test, amp_dataset_test[0,:], label = "True")
plt.legend()
plt.show()



























