import numpy as np
import matplotlib.pyplot as plt
import mlgw
from mlgw.GW_helper import load_dataset, make_set_split
from mlgw.ML_routines import PCA_model
from mlgw.GW_helper import compute_optimal_mismatch
import json
from tqdm import tqdm

###################################################################################################

pca_dir = 'test_pca'
pca_stuff = {'compute_gamma': True}
pca_stuff['model_alpha'] = PCA_model()
pca_stuff['model_alpha'].load_model('{}/pca_alpha'.format(pca_dir))
pca_stuff['model_beta'] = PCA_model()
pca_stuff['model_beta'].load_model('{}/pca_beta'.format(pca_dir))
pca_stuff['t_pca'] = np.loadtxt('{}/time_grid.dat'.format(pca_dir))

print(pca_stuff['t_pca']*20)

gen = mlgw.GW_generator()
modes = [(2,2), (2,1), (3,3), (4,4), (5,5)]

verbose = True
load_json = True
show_plots = True
out_file = 'pca_angles_mismatch_results.json'

res_dict = {
	'theta': [],
	'fref': []
}
for m in modes:
	res_dict['mismatch_{}{}'.format(*m)] = []

np.random.seed(0)

for i in tqdm(range(100), disable = verbose):
	if load_json:
		with open(out_file, 'r') as f:
			res_dict = json.load(f)
		break

	q = np.random.uniform(1,10)
	s1, s2 = np.random.uniform(0,1,2)
	t1, t2 = np.arccos(np.random.uniform(-1,1, 2))
	
	M = 20.
	m1, m2, s1x, s1y, s1z, s2x, s2y, s2z = q*M/(1+q), M/(1+q), s1*np.sin(t1), 0., s1*np.cos(t1), s2*np.sin(t2), 0., s2*np.cos(t2)
	fref = 50.
	theta = [m1, m2, s1x, s1y, s1z, s2x, s2y, s2z]

	t_grid = np.linspace(-10, 0.01, 10*4096)

	hlm_p, hlm_c, alpha_IMR, beta_IMR, gamma_IMR =  gen.get_twisted_modes(theta, t_grid, modes, f_ref = fref, alpha0 = 0., gamma0 = 0., pca_stuff = None)
	hlm_p_pca, hlm_c_pca, alpha_PCA, beta_PCA, gamma_PCA =  gen.get_twisted_modes(theta, t_grid, modes, f_ref = fref, alpha0 = 0., gamma0 = 0., pca_stuff = pca_stuff)

	if verbose: print("#########\nParams: {}".format(theta))
	res_dict['theta'].append(theta)
	res_dict['fref'].append(fref)
	
	id_mismatch = -1 #np.argmin(np.square(t_grid + 0.05))
	
	for i, mode in enumerate(modes):
		mismatch, _ = compute_optimal_mismatch(
				hlm_p[:id_mismatch,i]+1j*hlm_c[:id_mismatch,i],
				hlm_p_pca[:id_mismatch,i]+1j*hlm_c_pca[:id_mismatch,i],
				optimal = True, return_F = True)
		res_dict['mismatch_{}{}'.format(*mode)].append(float(mismatch))
		
		if verbose: print("Mode ({},{}) - {} ".format(*mode, mismatch))

	if show_plots:
		alpha, beta, gamma = gen.get_alpha_beta_gamma(theta, pca_stuff['t_pca']*M, fref)
		rec_alpha = pca_stuff['model_alpha'].reconstruct_data(pca_stuff['model_alpha'].reduce_data(alpha), K =4)[0] if 'model_alpha' in pca_stuff else np.zeros(pca_stuff['t_pca'].shape)
		rec_beta = pca_stuff['model_beta'].reconstruct_data(pca_stuff['model_beta'].reduce_data(beta), K = 6)[0]

		fig, axes = plt.subplots(3,1, figsize = (6.4, 4.8*3/2), sharex = True)
		axes[0].set_title('alpha')
		#axes[0].plot(pca_stuff['t_pca']*M, alpha[0], label = 'true')
		#axes[0].plot(pca_stuff['t_pca']*M, rec_alpha, label = 'PCA')
		axes[0].plot(t_grid, alpha_IMR[0], label = 'true')
		axes[0].plot(t_grid, alpha_PCA[0], label = 'PCA')
		axes[1].set_title('beta')
		#axes[1].plot(pca_stuff['t_pca']*M, beta[0], label = 'true')
		#axes[1].plot(pca_stuff['t_pca']*M, rec_beta, label = 'PCA')
		axes[1].plot(t_grid, beta_IMR[0], label = 'true')
		axes[1].plot(t_grid, beta_PCA[0], label = 'PCA')
		axes[2].set_title('gamma')
		axes[2].plot(t_grid, gamma_IMR[0], label = 'true')
		axes[2].plot(t_grid, gamma_PCA[0], label = 'PCA')
		plt.legend()
		plt.tight_layout()

		for i, mode in enumerate(modes):
			plt.figure()
			plt.title('(l,m) = ({},{})'.format(*mode))
			plt.plot(t_grid, hlm_p[:,i], label = 'true')
			plt.plot(t_grid, hlm_p_pca[:,i], label = 'pca')
			plt.legend()
		plt.tight_layout()
		plt.show()

if not load_json:
	with open(out_file, 'w') as f:
		json.dump(res_dict, f)


res_dict['theta'] = np.array(res_dict['theta'])

fig, axes = plt.subplots(len(modes),1, figsize = (6.4, 4.8*len(modes)/2), sharex = True)

n_bins = 2*int(np.sqrt(len(res_dict['theta'])))
for mode, ax in zip(modes, axes):
	log_mismatch = np.log10(res_dict['mismatch_{}{}'.format(*mode)])
	ax.set_title('(l,m) = ({},{})'.format(*mode))
	ax.hist(log_mismatch, color = 'orange', alpha = 0.8, bins = n_bins)
	ax.axvline(np.median(log_mismatch), ls = '--', c= 'k')
	ax.axvline(np.percentile(log_mismatch, 90), ls = 'dotted', c= 'k')
plt.xlabel(r'$log_{10}(\mathcal{M})$')
plt.tight_layout()


xs = [res_dict['theta'][:,0]/res_dict['theta'][:,1],
	np.linalg.norm(res_dict['theta'][:,[2,3,4]], axis = 1),
	np.linalg.norm(res_dict['theta'][:,[5,6,7]], axis = 1),
	np.arccos(res_dict['theta'][:,2]/res_dict['theta'][:,4])]

for x, label in zip(xs, ['q','s1','s2', 't1']):
	plt.figure()
	for mode in modes:
		plt.scatter(x, np.log10(res_dict['mismatch_{}{}'.format(*mode)]), s = 5, label = '{}{}'.format(*mode))
	plt.legend()
	plt.xlabel(label)

plt.show()


