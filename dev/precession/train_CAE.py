import glob
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset
from mlgw.ML_routines import PCA_model
import numpy as np
import tensorflow as tf
import time
import numpy as np



theta, alpha_dataset, beta_dataset, time_grid = load_dataset('tmp_angle_dataset.dat', N_data=None, N_entries = 2, N_grid = None, shuffle = False, n_params = 9)

val_id = 3000


pca = PCA_model()
pca.fit_model(alpha_dataset[:val_id], K = 20)
print(pca.get_eigenvalues())

rec_alpha = pca.reconstruct_data(pca.reduce_data(alpha_dataset[val_id:]))
mse = np.mean(np.square(rec_alpha - alpha_dataset[val_id:]))
print(mse)


plt.figure()
for k in range(1,20):
	rec_alpha = pca.reduce_data(alpha_dataset[val_id:])
	rec_alpha[:,k:] = 0.
	rec_alpha = pca.reconstruct_data(rec_alpha)
	mse = np.mean(np.square(rec_alpha - alpha_dataset[val_id:]))
	plt.scatter(k, mse)

	if False:
		fig, axes = plt.subplots(2,1, sharex = True)
		plt.suptitle('K = {}'.format(k))
		for id_ in range(1,5):
			axes[0].plot(time_grid, rec_alpha[id_], c='orange')
			axes[0].plot(time_grid, alpha_dataset[val_id+id_], c='blue')
			axes[1].plot(time_grid, rec_alpha[id_]-alpha_dataset[val_id+id_])
		plt.show()

plt.xlabel('# PC')
plt.ylabel('Validation mse')
plt.yscale('log')
plt.show()




	

if not True:
	N_plot = 20
	fig, axes = plt.subplots(2,1, sharex = True)
	for alpha_, beta_ in zip(alpha_dataset[:N_plot], beta_dataset[:N_plot]):
		axes[0].plot(time_grid, alpha_)
		axes[1].plot(time_grid, beta_)
		
	plt.xlabel(r'Time (s/M_sun)')
	plt.tight_layout()
	plt.show()
