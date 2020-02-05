import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model

folder = "GW_TD_dataset_mtotconst/"
    #loading PCA datasets
N_train = 7000
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta.dat")[:N_train,:]
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_ph.dat")[:N_train,:]
PCA_test_ph = np.loadtxt("../datasets/"+folder+"PCA_test_ph.dat")
K_PCA_to_fit = 7

print("Using "+str(PCA_train_ph.shape[0])+" train data")

	#adding extra features for basis function regression
new_features = ["00", "11","22", "01", "02", "12"#, "111", "110", "112","1111", "1122", "1100", "1120"
,"000", "001", "002", "011", "012", "022", "111", "112", "122", "222"
#,"000","111","222", "001"
,"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]
#   ,"00000", "00010","00020", "00110", "00220","00120","01110","01120", "01220", "02220","11110", "11120", "11220", "12220", "22220" 
#   ,"00001", "00011","00021", "00111", "00221","00121","01111","01121", "01221", "02221","11111", "11121", "11221", "12221", "22221" 
#   ,"00002", "00012","00022", "00112", "00222","00122","01112","01122", "01222", "02222","11112", "11122", "11222", "12222", "22222" ]

new_features = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222", #2nd/3rd order
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222", #4th order
"00000", "00001", "00002", "00011", "00012", "00022", "00111", "00112","00122", "00222", #5th order
"01111", "01112", "01122", "01222", "02222", "11111", "11112", "11122","11222", "12222", "22222"] #5th order


outfile = open("./saved_models_full_ph_TD/ph_feat", "w+")
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
load_list =		[]#[0   ,1   ,2   ,3   ,4   ,5   ,6   ,7   ,8   ,9   ,10  ,11  ,12  ,13  ,14  ]  #indices of models to be loaded from file
K = [5 for i in range(K_PCA_to_fit)] 
#for 4-th only
#K = 			[1  ,10  ,15  ,30  ,10  ,30  ,20  ,10  ,15  ,15  ,15  ,15  ,15  ,20  ,20  ]  #number of experts for each model


D = train_theta.shape[1] #number of independent variables

for k in range(0,K_PCA_to_fit):
	print("### Comp ", k, " | K = ", K[k])
		#useless variables for sake of clariness
	y_train = PCA_train_ph[:,k]
	y_test = PCA_test_ph[:,k]

	MoE_models.append(MoE_model(D,K[k]))
			#opt	val_set reg verbose threshold	N_it     step
	args = ["adam", None,   0e-4, False,  1e-2,		150,    2e-3]
	#args = [None,5,0]

	if k in load_list:
		MoE_models[-1].load("./saved_models_full_ph_TD/ph_exp_"+str(k),"./saved_models_full_ph_TD/ph_gat_"+str(k))
		print("Loaded model for comp: ", k)
	else:
		MoE_models[-1].fit(train_theta, y_train, threshold = 1e-2, args = args, verbose = False, val_set = (test_theta, y_test))
		MoE_models[-1].save("./saved_models_full_ph_TD/ph_exp_"+str(k),"./saved_models_full_ph_TD/ph_gat_"+str(k))

		#doing some test
	y_pred = MoE_models[-1].predict(test_theta)
	y_exp = MoE_models[-1].experts_predictions(test_theta)
	y_gat = MoE_models[-1].get_gating_probs(test_theta)
	resp = MoE_models[-1].get_responsibilities(test_theta, y_test)
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

	#plt.figure(1000)
	#plt.plot(test_theta[:,1], y_gat[:,0], 'o', ms = 1, label = "gating")
	#plt.plot(test_theta[:,1], resp[:,0], 'o', ms = 1, label = "resp") 	
	#plt.legend()
	#plt.show()

############Comparing mismatch for test waves
N_waves = 50

ph_PCA = PCA_model()
ph_PCA.load_model("../datasets/"+folder+"ph_PCA_model")

theta_vector_test, amp_dataset_test, ph_dataset_test, frequencies_test = create_dataset_TD(N_waves, N_grid = ph_PCA.get_V_matrix().shape[0], filename = None,
                t_coal = .4, q_range = (1.,10.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
                t_step = 1e-5, lal_approximant = "SEOBNRv2_opt")

	#preprocessing theta
theta_vector_test = add_extra_features(theta_vector_test, new_features)

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

F = compute_mismatch(amp_dataset_test, rec_ph_dataset, amp_dataset_test, ph_PCA.reconstruct_data(red_ph_dataset_test))
print("Avg fit mismatch vs PCA: ", np.mean(F))

mse = np.sum(np.square( rec_ph_dataset[:,290]-ph_dataset_test[:,290]))/(ph_dataset_test.shape[0])#*ph_dataset_test.shape[1])
print("Reconstruction mse: ", mse)

feat = 0 #1,2 index of feature to plot everything against
plt.figure(0)
plt.plot(theta_vector_test[:,feat], ph_dataset_test[:,20], 'o', label="true")
plt.plot(theta_vector_test[:,feat], rec_ph_dataset[:,20] , 'o', label="pred")
plt.legend()


plt.figure(100)
plt.plot(frequencies_test, rec_ph_dataset[0,:], label = "Rec")
plt.plot(frequencies_test, ph_dataset_test[0,:], label = "True")
plt.legend()

plt.figure(20)
for i in range(rec_ph_dataset.shape[0]):
	plt.plot(frequencies_test, rec_ph_dataset[i,:]-ph_dataset_test[i,:])
plt.show()

