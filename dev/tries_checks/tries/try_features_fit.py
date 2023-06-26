import numpy as np

import matplotlib.pyplot as plt 
import numpy as np
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved
import scipy.optimize

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model

def add_features(X, alpha):
	"""
	Add features specified by alpha
	Input:
		X (N,D_0) 			train matrix
		alpha (D,D_0*n)		exponents
	Output:
		X_new (N,D)		augmented data ready for basis function expansion
	"""
	D_0 = X.shape[1]
	X_new = np.ones((X.shape[0],alpha.shape[0]))
	for i in range(X_new.shape[1]):
		for j in range(alpha.shape[1]):
			X_new[:,i] = np.multiply(X_new[:,i], np.power(np.abs(X[:,j%D_0]), alpha[i,j])) #(N,)
			if alpha[i,j] != 0:
				X_new[:,i] = np.multiply(X_new[:,i],np.sign(X[:,j%D_0])) #(N,)
			#print(X_new[0,i], np.power(np.abs(X[0,j%D_0]), alpha[i,j]))
	return np.concatenate((X,X_new), axis =1)

def MoE_loss(alpha, X_train, y_train, X_test, y_test, alpha_shape, N_exp = 1):
	"""
	Return test mse loss of a trained MoE, given the features alpha.
	Input:
		alpha (D,D_0*n)
		X_train/X_test (N,D_0)		train/test dataset for independent variables
		y_train/y_test (N,)			train/test dataset for targets
		N_exp 						number of experts to use in model fitting
	Output:
		mse		mse of fitted MoE model on test set
	"""
	alpha_temp = np.reshape(alpha, alpha_shape)

	X_train_new = (add_features(X_train, alpha_temp))
	X_test_new = (add_features(X_test, alpha_temp))
	#X_train_new = X_train
	#X_test_new = X_test

	features = ["00", "11","22", "01", "02", "12"
	,"000", "001", "002", "011", "012", "022", "111", "112", "122", "222"
	,"0000", "0001","0002", "0011", "0022","0012","0111","0112", "0122", "0222","1111", "1112", "1122", "1222", "2222"]
	X_train_new = add_extra_features(X_train, features, [1,1,1])
	X_test_new = add_extra_features(X_test, features, [1,1,1])
	#print(X_train_check.shape ,"###########\n", X_train_new.shape)

	model = MoE_model(X_train_new.shape[1],N_exp)
	softmax_args = ["adam", None,   1e-4, False,  1e-2,		150,    2e-3] #args for softmax model

	model.fit(X_train_new, y_train, threshold = 1e-2, args = softmax_args, verbose = False)
	y_pred = model.predict(X_test_new)
	
	mse = np.mean(np.square(y_pred-y_test))
	print(mse)

	return mse

###################starting optimization###########################

folder = "GW_TD_dataset_mtotconst/"

    #loading PCA datasets
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta.dat")
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_ph.dat")[:,0]
PCA_test_ph = np.loadtxt("../datasets/"+folder+"PCA_test_ph.dat")[:,0]

D = train_theta.shape[1]*2 #"second order" polynomial

#alpha_0 = np.zeros((6,D))
alpha_0 = np.array([[1,0,0,1,0,0],[1,0,0,0,1,0], [1,0,0,0,0,1], [0,1,0,0,1,0], [0,1,0,0,0,1], [0,0,1,0,0,1]])

loss_args = (train_theta, PCA_train_ph, test_theta, PCA_test_ph, alpha_0.shape, 1)
print("Loss at beginning: ", MoE_loss(alpha_0, *loss_args))

#scipy.optimize.basinhopping(MoE_loss, np.ravel(alpha_0), disp = True, minimizer_kwargs = {"args":loss_args})


















