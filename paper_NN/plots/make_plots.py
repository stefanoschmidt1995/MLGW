import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as tck
from matplotlib.lines import Line2D
from matplotlib import rc
rc('font',**{'size': 9})
rc('text', usetex=True)
from scipy.stats import binned_statistic_2d, gaussian_kde

from pathlib import Path
import os
import json
import copy
import pandas as pd

###################################
def dodge_points(points, offset):
	"""
	Dodge every point by a multiplicative offset (multiplier is based on frequency of appearance)
	https://stackoverflow.com/questions/53093560/python-scatter-plot-overlapping-data

	Args:
		points (array-like (2D)): Array containing the points
		offset (float): Offset amount. Effective offset for each point is `index of appearance` * offset

	Returns:
		array-like (2D): Dodged points
	"""

	# Extract uniques points so we can map an offset for each
	uniques, inv, counts = np.unique(
		points, return_inverse=True, return_counts=True, axis=0
	)
	
	assert len(offset)==2

	for i, num_identical in enumerate(counts):
		# Find where the dodge values must be applied, in order
		points_loc = np.where(inv == i)[0]
		
		# Prepare dodge values
		if num_identical <= 5:
			dodge_values = np.array([offset[0] * i for i in range(num_identical)])
			#Apply the dodge values
			points[points_loc, 0] += dodge_values
		else:
			P = (num_identical+1)//2
			dodge_values = np.array(
					[[offset[0] * (i%P) for i in range(num_identical)],
					[points[points_loc[0], 1]*10**((1-int(i/P))*offset[1]) for i in range(num_identical)]]).T
			points[points_loc, :] += dodge_values
			
	return points

def plot_validation(run_folder, title = None, savefile = None, put_legend = True):
	
	if not isinstance(run_folder, Path): file_loc = Path(run_folder)

	trials = {}
	for x in os.listdir(file_loc):
		if x.startswith('trial'):
			with open(file_loc/"{}/trial.json".format(x), "r") as f:
				new_trial = json.load(f)
				if new_trial.pop('status') == 'COMPLETED':
					trials[new_trial.pop('trial_id')] = {'hyperparameters': new_trial['hyperparameters']['values'], 'score': new_trial['score']}

	scores = [v['score'] for k, v in trials.items()]
	scores = np.log10(scores)
	n_layers = np.array([v['hyperparameters']['layers'] for k, v in trials.items()])
	n_units = np.array([v['hyperparameters']['units'] for k, v in trials.items()])
	features = np.array([v['hyperparameters']['feature order'] for k, v in trials.items()])

		#Printing the best models
	ids_sort = np.argsort(scores)[:5]
	print("Best {} models: ".format(len(ids_sort)))
	for id_ in ids_sort:
		print('\t', *[v for i, (k, v) in enumerate(trials.items()) if i == id_])
	
	points = np.stack([n_layers, n_units], axis = -1).astype(np.float64)
	points = dodge_points(points, offset=[0.17, 0.15])
	
	markers = {1: 'o', 2: '*', 3: '^', 4: 's', 5: 'p', 6: 'H'}
	handles = []
	
	plt.figure(figsize = (3.54, 3.54*0.95))
	if title: plt.title(r'$\textrm{'+title+r'}$', fontsize = 9)
	for feat in set(features):
		ids = np.where(features == feat)
		plt.scatter(*points[ids].T, c = scores[ids], marker = markers[feat], label = '{} order'.format(feat))
		h =  Line2D([0], [0], marker=markers[feat], color='k', label='{} order'.format(feat),
                          markerfacecolor='k', lw = 0)
		handles.append(h)
	plt.xlim([0,11])
	plt.gca().xaxis.set_major_locator(tck.MaxNLocator(integer = True))
	plt.xlabel(r'$\textrm{\# layers}$')
	plt.ylabel(r'$\textrm{\# units}$')
	plt.yscale('log')
	
	cbar = plt.colorbar()
	cbar.set_label(r'$\textrm{Validation score}$', rotation=270, labelpad = 25)
	if put_legend: leg = plt.legend(handles = handles, loc = 'lower right')
	plt.tight_layout()

	if savefile: plt.savefig('../tex/img/'+savefile, bbox_inches='tight')

	#plt.show()

def plot_2d_data(data, values, ax = None, statistic= 'mean', bins = 30, vmin = None, vmax = None, cbar = True):
	stat, x_edges, y_edges, binnumber = binned_statistic_2d(*data.T, values = values, statistic = statistic, bins = bins)

	if ax is None:
		fig = plt.figure()
		ax = plt.gca()

	X, Y = np.meshgrid(x_edges,y_edges)
	mesh = ax.pcolormesh(X, Y, stat.T, vmin = vmin, vmax = vmax)
	
	if cbar:
		cbar = plt.colorbar(mesh, ax = ax)
		return ax, mesh, cbar
	return ax, mesh, None

def plot_speed_accuracy_hist(json_file):
	dataset = pd.read_json(json_file, lines = True)
	dataset['M'] = np.array(dataset['m1']+dataset['m2'])
	dataset['q'] = np.array(dataset['m1']/dataset['m2'])
	
	kwargs = {'histtype':'step', 'density': True, 'bins': 100}
	
		#Accuracy histogram
	fig, axes = plt.subplots(2,1, sharex = True, figsize = (3.54, 3.54))
	#plt.suptitle(r"$\textrm{Model accuracy}$")
	axes[0].hist(np.log10(dataset['mismatch']), color = 'k', label = 'overall', **kwargs)
	for k in dataset.keys():
		if k.find('mismatch_')==-1: continue
		mode = k.replace('mismatch_', '') 
		axes[1].hist(np.log10(dataset[k]), label = "$({},{})$".format(*mode), **kwargs)
	plt.xlabel(r'$\log_{10}\mathcal{F}$')
		#Doing annotations
	ann_str = "\\\\{}: {:.2E}\\\\{}: {:.2E}\\\\{}: {:.2E}".format(
		r'\textrm{mean}', np.mean(dataset['mismatch']),
		r'\textrm{median}',np.median(dataset['mismatch']),
		r'90^\textrm{th} \textrm{ perc.}', np.percentile(dataset['mismatch'], 90))
	ann_str = ann_str.replace(r'E-04', r'\times 10^{-4}')
	ann_str = '$'+ann_str+'$'
	print(ann_str)
	axes[0].annotate(ann_str,
		xy = (0.2,0.4),
		xycoords = 'axes fraction',
		fontsize = 7
		)
	axes[0].legend(*axes[1].get_legend_handles_labels(), loc = 'upper right')
	axes[0].set_yscale('log')
	axes[1].set_yscale('log')
	plt.xticks(list(range(-6,1)))
	plt.tight_layout()
	plt.savefig('../tex/img/accuracy.pdf', bbox_inches='tight')
	
		#Countor plots
	print(dataset.keys())
	l_latex = { 'M': r'$M (M_\odot)$', 'q': r'$q$',
		's1z': r'$s_\mathrm{1z}$', 's2z': r'$s_\mathrm{2z}$'}
	
	fig, axes = plt.subplots(2, 3, figsize = (3.54*2, 3.54))
	modes = ['22', '21', '33', '44', '55']
	k1, k2 = 'q', 's1z'
	for lm, ax in zip(modes, axes.flatten()):
		ax.set_title('$({},{})$'.format(*lm))
		ax, mesh, cbar = plot_2d_data(np.array(dataset[[k1,k2]]), np.log10(dataset['mismatch_{}'.format(lm)]),
			ax = ax, statistic= 'mean', bins = 30, cbar = True)
		ax.set_xlabel(l_latex[k1])
		ax.set_ylabel(l_latex[k2], labelpad = None if k2=='q' else 1)
		cbar.set_label(r'$\log_{10}\mathcal{F}$', rotation=270, labelpad = 12)
	axes[1][2].set_visible(False)
	plt.tight_layout()
	fig.subplots_adjust(hspace = 1)
	#fig.subplots_adjust(right=0.85, left = 0.08, top = 0.98, bottom = 0.19, wspace = 0.4)
	#cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.7])
	#cbar = fig.colorbar(mesh, cax=cbar_ax)
	#cbar.set_label(r'$\log_{10}\mathcal{F}$', rotation=270, labelpad = 15)
	plt.savefig('../tex/img/colormesh_modes.pdf', bbox_inches='tight')
	
	fig, axes = plt.subplots(1, 3, figsize = (3.54*2, 3.54/1.5))
	for ax, (k1, k2) in zip(axes, [('M', 'q'), ('q', 's1z'), ('q', 's2z')]):
		print("Making colored plot for {}-{}".format(k1,k2))
		ax, mesh, cbar = plot_2d_data(np.array(dataset[[k1,k2]]), np.log10(dataset['mismatch']),
			ax = ax, statistic= 'mean', bins = 30, cbar = False ,
			vmin = np.percentile(np.log10(dataset['mismatch']), 5), vmax = np.percentile(np.log10(dataset['mismatch']), 95))
		ax.set_xlabel(l_latex[k1])
		ax.set_ylabel(l_latex[k2], labelpad = None if k2=='q' else 1)

	fig.subplots_adjust(right=0.85, left = 0.08, top = 0.98, bottom = 0.19, wspace = 0.4)
	cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.7])
	cbar = fig.colorbar(mesh, cax=cbar_ax)
	cbar.set_label(r'$\log_{10}\mathcal{F}$', rotation=270, labelpad = 15)

	#plt.tight_layout()
	plt.savefig('../tex/img/colormesh.pdf', bbox_inches='tight')

		#Speed up histogram
	plt.figure(figsize = (3.54, 3.54/2))
	#plt.title(r'$\textrm{Timing analysis}$')
	plt.hist(dataset['time_lal']/dataset['time_mlgw'], label = r"$\textrm{no batch}$", **kwargs)
	plt.hist(dataset['time_lal']/dataset['time_mlgw_100'], label = r"$\textrm{batch}$", **kwargs)
	plt.axvline(1, c='k', ls ='dashed')
	plt.xlabel(r'$t_{\texttt{SEOBNRv4PHM}}/t_{\texttt{mlgw}}$')
	plt.legend()
	plt.tight_layout()
	plt.savefig('../tex/img/timing.pdf', bbox_inches='tight')
	plt.show()
	
	
	plt.show()

def mse_table():
	import mlgw.NN_model
	from mlgw.GW_generator import mode_generator_NN
	from mlgw.ML_routines import augment_features
	
	modes = ['22', '21', '33', '44', '55']
	
	df = []
	pca_folder = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/pca_datasets/SEOBNRv4PHM'
	model_folder = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/models_NN/model_SEOBNRv4PHM/{}/'
	#model_folder = '/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/models_NN/model_IMRPhenomTPHM_mc_chieff/{}/'

	#pca_folder = '../pca_datasets/IMRPhenomTPHM'
	#model_folder = '../models_NN/model_IMRPhenomTPHM/{}/'
	
	for mode in modes:
		PCA_data_amp = mlgw.NN_model.PcaData('{}/{}/'.format(pca_folder, mode), [0,1,2,3], 'amp')
		PCA_data_ph = mlgw.NN_model.PcaData('{}/{}/'.format(pca_folder, mode), [0,1], 'ph')
		PCA_data_ph_2345 = mlgw.NN_model.PcaData('{}/{}/'.format(pca_folder, mode), [2,3,4,5], 'ph')
		PCA_data_residual = mlgw.NN_model.PcaData('{}/{}_residual_01'.format(pca_folder, mode), [0,1], 'ph')

		generator = mode_generator_NN((2,2), model_folder.format(mode))

		ph_res_pred = generator.ph_residual_models['01'](augment_features(PCA_data_residual.test_theta, generator.ph_residual_models['01'].features))
		residual_mse = np.sum(np.square(ph_res_pred - PCA_data_residual.test_var), axis =0)/ph_res_pred.shape[0]
		
		#generator.ph_residual_models = {}; print("Removing residual models")
		
		amp_pred, ph_pred = generator.get_red_coefficients(PCA_data_ph.test_theta)
		
		#ph_pred[:,[0,1]] += PCA_data_residual.test_var*generator.ph_res_coefficients['01']

		amp_mse = np.sum(np.square(amp_pred[:,:4] - PCA_data_amp.test_var), axis =0)/amp_pred.shape[0]
		ph_mse = np.sum(np.square(ph_pred[:,:2] - PCA_data_ph.test_var), axis =0)/ph_pred.shape[0]
		ph_2345_mse = np.sum(np.square(ph_pred[:,2:6] - PCA_data_ph_2345.test_var), axis =0)/ph_pred.shape[0]
	
		print('mode {}:\n\tamp: {}\n\tph: {}\n\tph2345: {}\n\tphres: {}'.format(mode, amp_mse, ph_mse, ph_2345_mse, residual_mse))
		
		f = '{:.2e}'.format
		df.append({'mode':mode, 'amp': [f(a) for a in amp_mse],
				'ph': [f(a) for a in ph_mse],
				'ph_2345': [f(a) for a in ph_2345_mse],
				'ph_residual': [f(a) for a in residual_mse]})
	
	df = pd.DataFrame(df)
	#print(df.to_latex(index_names = False))

		

if __name__=='__main__':
	plot_validation('/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_SEOB_22/tuning_amp_22_0123', "Tuning of amplitude", 'tuning_amp.pdf', False)
	plot_validation('/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_SEOB_22/tuning_ph_22_01', "Tuning of phase - PC 0 1", 'tuning_ph_01.pdf', False)
	plot_validation('/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_SEOB_22/tuning_ph_22_2345', "Tuning of phase - PC 2 3 4 5", 'tuning_ph_2345.pdf', False)
	plot_validation('/home/stefano/Dropbox/Stefano/PhD/mlgw_repository/dev/mlgw_NN/bayesian_tuning_SEOB_22/tuning_ph_22_01_residual', "Tuning of residual model for phase", 'tuning_ph_01_residual.pdf', True)

	#mse_table()
	plot_speed_accuracy_hist('model_SEOB.json')
	quit()
	

