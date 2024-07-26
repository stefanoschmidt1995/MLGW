"""
Convenience script to fit a PCA model for an angle dataset

Usage:

	python make_pca_model.py --dataset-file dataset_full_fref_const_3000_grid_points.dat --out-dir test_pca --k-pca 4 6 --n-train-pca 3900 --training-fraction 0.8

"""

#TODO: merge this script with the one for the amplitude and phase
#TODO: allow for multiple dataset file support

import numpy as np
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model

import argparse
import sys
import os

def plot_mse_as_k(pca_model, dataset, K_min, K_max, time_grid = None, savedir = None, name = ''):
	reduced_dataset = pca_model.reduce_data(dataset)
	
	plt.figure()
	plt.title(name)
	for k in range(K_min, K_max+1):
		rec_dataset = pca_model.reconstruct_data(reduced_dataset, K = k)
		mse = np.mean(np.square(rec_dataset - dataset))
		plt.scatter(k, mse)

	plt.xlabel('# PC')
	plt.ylabel('Validation mse')
	plt.yscale('log')
	if savedir: plt.savefig('{}/pca_residuals_{}.png'.format(savedir, name))
	
	if time_grid is not None:
		fig, axes = plt.subplots(2,1, sharex = True)
		plt.suptitle('{}\nK = {}'.format(name, K_max))
		for id_ in range(1,9):
			axes[0].plot(time_grid, rec_dataset[id_], c='orange')
			axes[0].plot(time_grid, dataset[id_], c='blue')
			axes[1].plot(time_grid, rec_dataset[id_]-dataset[id_])

		if savedir: plt.savefig('{}/pca_rec_{}.png'.format(savedir, name))

#########################################################################

parser = argparse.ArgumentParser(__doc__)

parser.add_argument(
	"--dataset-file", type = str, required = True,
	help="Dataset file")
parser.add_argument(
	"--out-dir", type = str, required = False, default = './out_pca',
	help="Output directory")
parser.add_argument(
	"--k-pca", type = int, nargs = 2, required = False, default = (2,2),
	help="Number of PCs to consider for alpha and beta")
parser.add_argument(
	"--training-fraction", type = float, required = False, default = 0.8,
	help="Fraction of points to be stored in the PCA training dataset")
parser.add_argument(
	"--align-alpha", action = 'store_true', default = False,
	help="Whether to align the alphas so that they are all zero in at the beginning of the time grid")
parser.add_argument(
	"--n-train-pca", type = int, required = False, default = None,
	help="Number of training data to use for the PCA")


args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok = True)

theta, alpha_dataset, beta_dataset, time_grid = load_dataset(args.dataset_file, N_data=None, N_entries = 2, N_grid = None, shuffle = False, n_params = 9)
beta_dataset = np.arccos(beta_dataset)

	#Setting all the alphas to zero at the beginning of the grid! What are the consequences for this??
if args.align_alpha: alpha_dataset = (alpha_dataset.T - alpha_dataset[:,0]).T

args.n_train = int(theta.shape[0]*args.training_fraction)
if not args.n_train_pca:
	args.n_train_pca = args.n_train
if args.n_train_pca >= args.n_train:
	args.n_train_pca = args.n_train

np.savetxt('{}/time_grid.dat'.format(args.out_dir), time_grid)

theta_train, theta_test = theta[:args.n_train], theta[args.n_train:]
np.savetxt('{}/PCA_train_theta_angles.dat'.format(args.out_dir), theta_train)
np.savetxt('{}/PCA_test_theta_angles.dat'.format(args.out_dir), theta_test)

for (dataset, k, name) in zip([alpha_dataset, beta_dataset], args.k_pca, ['alpha', 'beta']):

	print("Making PCA for {}".format(name))

	pca_file = '{}/pca_{}'.format(args.out_dir, name)

	pca = PCA_model()
	if os.path.isfile(pca_file):
		pca.load_model(pca_file)
	else:
		pca.fit_model(dataset[:args.n_train_pca], K = k)
		pca.save_model(pca_file)
	
	angle_train, angle_test = dataset[:args.n_train], dataset[args.n_train:]
	
	pca_dataset_train = pca.reduce_data(angle_train)
	pca_dataset_test = pca.reduce_data(angle_test)
	
	np.savetxt('{}/PCA_train_{}.dat'.format(args.out_dir, name), pca_dataset_train)
	np.savetxt('{}/PCA_test_{}.dat'.format(args.out_dir, name), pca_dataset_test)
	
	plot_mse_as_k(pca, dataset[args.n_train:], 1, k, time_grid = time_grid, savedir = args.out_dir, name = name)
	
	del pca_dataset_train, pca_dataset_test

plt.show()



