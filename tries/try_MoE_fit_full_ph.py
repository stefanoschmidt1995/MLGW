import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model

folder = "GW_std_dataset/"
    #loading PCA datasets
N_train = 7000
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta_full.dat")[:N_train,:]
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta_full.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_full_ph.dat")[:N_train,:]
PCA_test_ph = np.loadtxt("../datasets/"+folder+"PCA_test_full_ph.dat")
K_PCA_to_fit = 11

	#adding extra features for basis function regression
new_features = ["00", "11","22", "01", "02", "12"
#,"000", "001", "002", "011", "012", "022", "111", "112", "122", "222"]
#,"000","111","222", "001"
,"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]
outfile = open("./saved_models_full_ph/ph_feat", "w+")
outfile.write("\n".join(new_features))
outfile.close()


train_theta = add_extra_features(train_theta, new_features)
test_theta = add_extra_features(test_theta, new_features)

print(train_theta.shape, test_theta.shape)

print("Loaded "+ str(train_theta.shape[0]+test_theta.shape[0])+
      " data with ", PCA_train_ph.shape[1]," PCA components")
print("Spins are allowed to vary within domain [-0.8,0.8]x[-0.8,0.8]")


   #setting up an EM model for each component
MoE_models = 	[]
load_list =		[0   ,1   ,2   ,3   ,4   ,5   ,6   ,7   ,8   ,9   ,10  ,11  ,12  ,13  ,14  ]  #indices of models to be loaded from file

#for 4-th only
K = 			[15  ,20  ,30  ,20  ,20  ,30  ,20  ,20  ,20  ,20  ,25  ,25  ,25  ,25  ,25  ]  #number of experts for each model


D = train_theta.shape[1] #number of independent variables

for k in range(0,K_PCA_to_fit):
	print("### Comp ", k, " | K = ", K[k])
		#useless variables for sake of clariness
	y_train = PCA_train_ph[:,k]
	y_test = PCA_test_ph[:,k]

	MoE_models.append(MoE_model(D,K[k]))
			#opt	val_set reg verbose threshold	N_it     step
	args = ["adam", None,   0e-4, False,  1e-4,		150,    2e-3]
	#args = [None,5,0]

	if k in load_list:
		MoE_models[-1].load("./saved_models_full_ph/ph_exp_"+str(k),"./saved_models_full_ph/ph_gat_"+str(k))
		print("Loaded model for comp: ", k)
	else:
		MoE_models[-1].fit(train_theta, y_train, threshold = 1e-2, args = args, verbose = True, val_set = (test_theta, y_test))
		MoE_models[-1].save("./saved_models_full_ph/ph_exp_"+str(k),"./saved_models_full_ph/ph_gat_"+str(k))

		#doing some test
	y_pred = MoE_models[-1].predict(test_theta)
	y_exp = MoE_models[-1].experts_predictions(test_theta)
	y_gat = MoE_models[-1].get_gating_probs(test_theta)
	print("Test square loss for comp "+str(k)+": ",np.sum(np.square(y_pred-y_test))/(y_pred.shape[0]))
	print("LL for comp "+str(k)+" (train,val): ", (MoE_models[-1].log_likelihood(train_theta,y_train),MoE_models[-1].log_likelihood(test_theta,y_test)))

	for i in range(3):
		plt.figure(i*K[k]+k, figsize=(20,10))
		plt.title("Component #"+str(k)+" vs q/s1/s2 | index "+str(i))
		plt.plot(test_theta[:,i], y_test, 'o', ms = 3,label = 'true')
		plt.plot(test_theta[:,i], y_pred, 'o', ms = 3, label = 'pred')
		#plt.plot(test_theta[:,i], y_gat, 'o', ms = 1) 						#plotting gating predictions
		#plt.plot(test_theta[:,i], y_exp, 'o', ms = 1,label = 'exp_pred')	#plotting experts predictions
		plt.legend()
		if i ==0 or i==1 or i ==2:
			#pass
			plt.savefig("../pictures/PCA_comp_full_ph/fit_"+str(k)+"_vs"+str(i)+".jpeg")
		plt.close(i*K[k]+k)
	#plt.show()


############Comparing mismatch for test waves
N_waves = 200

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset(N_waves, N_grid = 2048, filename = None,
                q_range = (1.,5.), s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
				log_space = True,
                f_high = 1000, f_step = 5e-2, f_max = None, f_min =None, lal_approximant = "IMRPhenomPv2")



	#preprocessing theta
theta_vector_test = add_extra_features(theta_vector_test, new_features)

ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/"+folder+"PCA_model_full_ph.dat")

red_ph_dataset_test = ph_PCA.reduce_data(ph_dataset_test)
if K_PCA_to_fit < PCA_train_ph.shape[1]:
	red_ph_dataset_test[:,K_PCA_to_fit:] = 0
#* np.random.normal(1,5e-3,size=(ph_dataset_test.shape[0], ph_PCA.get_PCA_params()[0].shape[1]))
F_PCA = compute_mismatch(amp_dataset_test, ph_PCA.reconstruct_data(red_ph_dataset_test),
						 amp_dataset_test, ph_dataset_test)
print("Avg PCA mismatch: ", np.mean(F_PCA))

rec_PCA_dataset = np.zeros((N_waves, PCA_train_ph.shape[1]))
for k in range(len(MoE_models)):
	rec_PCA_dataset[:,k] = MoE_models[k].predict(theta_vector_test)

rec_ph_dataset = ph_PCA.reconstruct_data(rec_PCA_dataset)

F = compute_mismatch(amp_dataset_test, rec_ph_dataset, amp_dataset_test, ph_dataset_test)
print("Avg fit mismatch: ", np.mean(F))

plt.figure(100)
plt.plot(frequencies_test, rec_ph_dataset[0,:], label = "Rec")
plt.plot(frequencies_test, ph_dataset_test[0,:], label = "True")
plt.legend()
plt.show()



























