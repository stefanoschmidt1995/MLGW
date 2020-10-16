# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved

from GW_helper import * 	#routines for dealing with datasets
from ML_routines import *	#PCA model
from EM_MoE import *		#MoE model

# Loading dataset
folder = "GW_TD_dataset_mtotconst/"
    #loading PCA datasets
N_train = 70000
train_theta = np.loadtxt("../datasets/"+folder+"PCA_train_theta.dat")[:N_train,:]
test_theta = np.loadtxt("../datasets/"+folder+"PCA_test_theta.dat")
PCA_train_ph = np.loadtxt("../datasets/"+folder+"PCA_train_ph.dat")[:N_train,:]
PCA_test_ph = np.loadtxt("../datasets/"+folder+"PCA_test_ph.dat")
K_PCA_to_fit = 8


# Fit regression model
comp_to_fit = 0
print("Fitting component: ", comp_to_fit)
regr_1 = DecisionTreeRegressor(max_depth=15)
regr_2 = RandomForestRegressor(n_estimators = 100, max_depth=15)
regr_1.fit(train_theta, PCA_train_ph[:,comp_to_fit])
regr_2.fit(train_theta, PCA_train_ph[:,comp_to_fit])

# Predict
y_1_train = regr_1.predict(train_theta)
y_2_train = regr_2.predict(train_theta)
y_1_test = regr_1.predict(test_theta)
y_2_test = regr_2.predict(test_theta)

print("Train error 1: ", np.sum(np.square(y_1_train-PCA_train_ph[:,comp_to_fit]))/len(y_1_train) )
print("Train error 2: ", np.sum(np.square(y_2_train-PCA_train_ph[:,comp_to_fit]))/len(y_2_train) )
print("Test error 1: ", np.sum(np.square(y_1_test-PCA_test_ph[:,comp_to_fit]))/len(y_1_test) )
print("Test error 2: ", np.sum(np.square(y_2_test-PCA_test_ph[:,comp_to_fit]))/len(y_1_test) )
print(regr_1.feature_importances_, regr_1.get_depth())

# Plot test results
plt.figure()
plt.plot(test_theta[:,0],PCA_test_ph[:,comp_to_fit],'o', ms=2,
            c="darkorange", label="true")
#plt.plot(test_theta[:,0], y_1_test,'o', color="cornflowerblue",label="max_depth= "+str(regr_1.get_depth()), ms=2)
plt.plot(test_theta[:,0], y_2_test,'o', color="yellowgreen", label="max_depth= "+str(regr_1.get_depth()), ms=2)
plt.xlabel("q")
plt.ylabel("PC")
plt.title("Decision Tree Regression - Test")
plt.legend()

plt.figure()
plt.plot(train_theta[:,0],PCA_train_ph[:,comp_to_fit],'o', ms=2,
            c="darkorange", label="true")
#plt.plot(train_theta[:,0], y_1_train,'o', color="cornflowerblue", label="max_depth= "+str(regr_1.get_depth()), ms=2)
plt.plot(train_theta[:,0], y_2_train,'o', color="yellowgreen", label="max_depth= "+str(regr_1.get_depth()), ms=2)
plt.xlabel("q")
plt.ylabel("PC")
plt.title("Decision Tree Regression - Train ")
plt.legend()
plt.show()

plt.show()



