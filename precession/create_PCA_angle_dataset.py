import numpy as np
import matplotlib.pyplot as plt

from precession_helper import *

import sys
sys.path.insert(0,'../../mlgw_v2')
from GW_helper import *
from ML_routines import *

#alpha, beta = get_alpha_beta(1.4, .4, .14, .75, 1.34, 3., 500., times)

#create_dataset_alpha_beta(N_angles = 100, "validation_angles.dat", N_grid = 500, tau_min = 20., (1.1,10.))

#quit()
params, alpha, beta, times = load_dataset("starting_dataset.dat", N_data = 1000, n_params = 6)

print("DATA LOADED")
beta = beta
train_p, test_p, train_beta, test_beta = make_set_split(params, beta)

K_max = 10
PCA_beta = PCA_model()
print("fitting", beta.shape)
E_beta = PCA_beta.fit_model(train_beta, K_max, scale_PC=True)
print("PCA eigenvalues for beta: ", E_beta)
red_true_test_beta = PCA_beta.reduce_data(test_beta)
red_approx_test_beta = np.zeros(red_true_test_beta.shape)
mse_list = [0 for i in range(K_max)]
for k in range(K_max):
	break
	red_approx_test_beta[:,k] = red_true_test_beta[:,k] 
	rec_test_beta = PCA_beta.reconstruct_data(red_approx_test_beta) #(N,D)
	mse = np.sum(np.square(rec_test_beta- test_beta))/(beta.shape[0]*beta.shape[1])
	mse_list[i] = mse
	print("mse(k): ",k,mse)

	if k == 10:
		plt.title("Test")
		to_plot = test_beta.T/rec_test_beta.T -1.
		to_plot_fft = np.fft.rfft(to_plot)
		#plt.plot(np.abs(to_plot[:,:10]))

		plt.plot(times,rec_test_beta[:10,:].T)
		plt.plot(times,test_beta[:10,:].T)
		plt.show()


plt.figure()
plt.plot(range(K_max), mse_list)
plt.yscale('log')



plt.figure()
plt.plot(times, alpha.T[:,:100])#- alpha.T[0,:100])

plt.figure()
plt.plot(times, beta.T[:,:100])

plt.show()


