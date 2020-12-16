import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *
from ML_routines import *

#alpha, beta = get_alpha_beta(1.4, .4, .14, .75, 1.34, 3., 500., times)

#create_dataset_alpha_beta(50, "angles.dat", 10000, 1000, (1.1,10.))

params, alpha, beta, times = load_dataset("angles_dataset.dat", N_data = 500, n_params = 6)

print("DATA LOADED")
beta = beta[:,::4] #downsampling
train_p, test_p, train_beta, test_beta = make_set_split(params, beta)

K_max = 100
PCA_beta = PCA_model()
print("fitting", beta.shape)
E_beta = PCA_beta.fit_model(train_beta, K_max, scale_PC=True)
print("PCA eigenvalues for beta: ", E_beta)
red_true_test_beta = PCA_beta.reduce_data(test_beta)
red_approx_test_beta = np.zeros(red_true_test_beta.shape)
mse_list = []
for k in range(K_max):
	red_approx_test_beta[:,k] = red_true_test_beta[:,k] 
	rec_test_beta = PCA_beta.reconstruct_data(red_approx_test_beta) #(N,D)
	mse = np.sum(np.square(rec_test_beta- test_beta))/(beta.shape[0]*beta.shape[1])
	mse_list.append(mse)
	print(mse)
	if k == 99:
		plt.plot(rec_test_beta.T)
		plt.plot(test_beta.T)
		plt.show()


plt.figure()
plt.plot(range(K_max), mse_list)
plt.yscale('log')



plt.figure()
plt.plot(times, alpha.T[:,:100]- alpha.T[0,:100])

plt.figure()
plt.plot(times[::4], beta.T[:,:100])

plt.show()

