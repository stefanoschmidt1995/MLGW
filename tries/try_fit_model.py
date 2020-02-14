import numpy as np

import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved
import scipy.optimize

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model
from fit_model import *

folder = "GW_TD_dataset_mtotconst/"

    #loading PCA datasets
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta.dat")
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta.dat")
y_train = np.loadtxt("../datasets/"+folder+"PCA_train_ph.dat")[:,0]
y_test = np.loadtxt("../datasets/"+folder+"PCA_test_ph.dat")[:,0]

N_exp = 3
softmax_args = ["adam", None,   1e-4, False,  1e-2,		150,    2e-3] #args for softmax model
features = ["00", "01", "02", "11","22","12"]
features = ["00", "11","22", "01", "02", "12","000", "001", "002", "011", "012", "022", "111", "112", "122", "222"] #2nd/3rd order


	#fitting with fit model routine
temp_F, temp_mse = fit_MoE("ph", "../datasets/"+folder, "saved_models_full_ph_TD",N_exp, comp_to_fit = 3, features = features, EM_threshold = 1e-2, args = softmax_args, verbose = True, test_mismatch = True)
print("fit_model: ", temp_mse)



train_theta = add_extra_features(train_theta, features, [1,1,1])
test_theta = add_extra_features(test_theta, features, [1,1,1])

model = MoE_model(train_theta.shape[1],N_exp)

#model.fit(train_theta, y_train, threshold = 1e-2, args = softmax_args, verbose = False)
y_pred = model.predict(test_theta)
mse = np.mean(np.square(y_pred-y_test))
print("normal fit: ", mse)



















