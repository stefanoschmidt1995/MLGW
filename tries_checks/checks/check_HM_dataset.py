import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../routines') #folder in which every relevant routine is saved
import scipy.optimize

from GW_helper import * 	#routines for dealing with datasets

load_file = "./shift_dataset"

#theta, shifts = 
if False:
	create_shift_dataset(N_data = 3000, modes = [[3,2]], filename = load_file,
			q_range = (1.,5.), m2_range = None, s1_range = (-0.8,0.8), s2_range = (-0.8,0.8),
           path_TEOBResumS = '/home/stefano/Desktop/Stefano/scuola/uni/tesi_magistrale/code/TEOBResumS/Python')


#do the fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt(load_file, skiprows = 2)
X = data[:,:3]
Y = data[:,3]

train_fraction = 0.8 #fraction of training points
deg = 4 #degree of the polynomiale

        #preprocessing data
polynomial_features = PolynomialFeatures(degree=deg)
X = polynomial_features.fit_transform(X)

print(X[0,:])

N_data = X.shape[0]

X_train = X[:int(N_data*train_fraction),:]
Y_train = Y[:int(N_data*train_fraction)]
X_test = X[int(N_data*train_fraction):,:]
Y_test = Y[int(N_data*train_fraction):]

        #building and fitting the model
model = LinearRegression()
model.fit(X_train, Y_train)

        #testing
Y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test,Y_pred))

print("Average rmse ",rmse)
print("Average relative rmse ",rmse/np.mean(Y_test))

        #plotting
plt.title("{}th order polynomial\nmse = {} ".format(deg,rmse))
plt.plot(X_test[:,1], Y_test, 'o', label = 'true', ms = 2)
plt.plot(X_test[:,1], Y_pred, 'o', label = 'pred', ms = 2)
plt.legend()
plt.show()











