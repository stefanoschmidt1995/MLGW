import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from ML_routines import *	#PCA model
from fit_model import *

fifth_order = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222", #2nd/3rd order
"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"] #4th order

N_train_list = [50, 100, 200,300, 500,700, 1000, 2000, None]

load = False

if load:
	res = np.load("MoE_models/res_ngrid.npy") #(len(N_train), 4, 2)

if not load:
	res = np.zeros((len(N_train_list),4, 2))
	for i in range(len(N_train_list)):
		args = ["adam", None,   1e-4, False, 1e-2,		150, 2e-3] #default arguments for softmax fit routine
		temp_F_train, temp_F_test, temp_mse_train, temp_mse_test = fit_MoE("ph", "../datasets/GW_TD_dataset_merger/", "MoE_models", 4, comp_to_fit = 7, features = fifth_order, EM_threshold = 1e-2, args = args, N_train = N_train_list[i], verbose = False, train_mismatch = True, test_mismatch = True)

		res[i,:,0] = [temp_F_train, *temp_mse_train[0:3]]
		res[i,:,1] = [temp_F_test, *temp_mse_test[0:3]]
		print(res[i,:,0], res[i,:,0])

	np.save("MoE_models/res_ngrid.npy", res)


labels_list = [r'$\mathcal{F}$',"mse 1st PC","mse 2nd PC","mse 3rd PC"]

fig, axs = plt.subplots(res.shape[1], figsize = (10,5), sharex=True)#, sharey=True)
plt.suptitle("MoE fit performances vs Number of training points")
for i in range(res.shape[1]):
	axs[i].plot(N_train_list, res[:,i,0],'x',c = 'k', label = "train")
	axs[i].plot(N_train_list, res[:,i,1],'o',c = 'k', label = "test")
	axs[i].legend()
	axs[i].set_ylabel(labels_list[i])
	axs[i].set_yscale('log')
	if i ==0:
		axs[i].set_ylim([np.min(res[:,0,:]),1])
axs[-1].set_xscale('log')
axs[-1].set_xlabel(r'$N_{train}$')


plt.show()











